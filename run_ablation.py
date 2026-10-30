"""Unified, resumable CS-Net ablation training/evaluation entry point."""

import argparse
import contextlib
import csv
import datetime as dt
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ablation_data import PairedRawDataset, validate_independent_test
from ablation_metrics import summarize_metric_rows, volume_metrics
from ablation_models import build_loss, build_model
from ablation_visualization import generate_comparison


CONFIG_PATH = Path(__file__).with_name("ablation_config.json")


def now_iso():
	return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temp = path.with_suffix(path.suffix + ".tmp")
	temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
	os.replace(str(temp), str(path))


def load_config(path):
	with Path(path).open(encoding="utf-8") as stream:
		config = json.load(stream)
	required = {"paths", "data", "training", "model", "evaluation", "experiments"}
	missing = required.difference(config)
	if missing:
		raise ValueError("Config missing sections: {}".format(sorted(missing)))
	return config


def resolve_output_root(config, config_path, override=None):
	if override:
		return Path(override).resolve()
	root = Path(config["paths"]["output_root"])
	return root.resolve() if root.is_absolute() else (Path(config_path).parent / root).resolve()


def select_experiments(config, requested):
	if requested == ["all"]:
		return list(config["experiments"])
	unknown = set(requested).difference(config["experiments"])
	if unknown:
		raise ValueError("Unknown experiments: {}".format(sorted(unknown)))
	return requested


def environment_info(deterministic):
	info = {
		"timestamp": now_iso(), "python": sys.version, "platform": platform.platform(),
		"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
		"cuda_available": torch.cuda.is_available(),
		"cudnn": torch.backends.cudnn.version(),
		"deterministic_algorithms": bool(deterministic),
		"deterministic_policy": "torch.use_deterministic_algorithms(..., warn_only=True); warnings are retained in console/run logs",
		"cudnn_benchmark": torch.backends.cudnn.benchmark,
		"cudnn_deterministic": torch.backends.cudnn.deterministic,
	}
	if torch.cuda.is_available():
		info["gpu"] = torch.cuda.get_device_name(0)
		info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
	return info


def set_seed(seed, deterministic=True):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = bool(deterministic)
	torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def rng_state():
	state = {
		"python": random.getstate(), "numpy": np.random.get_state(),
		"torch": torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		state["cuda"] = torch.cuda.get_rng_state_all()
	return state


def restore_rng_state(state):
	if not state:
		return
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch"].cpu())
	if torch.cuda.is_available() and "cuda" in state:
		torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def gpu_memory_summary():
	if not torch.cuda.is_available():
		return {"available": False}
	return {
		"available": True,
		"allocated_bytes": torch.cuda.memory_allocated(),
		"reserved_bytes": torch.cuda.memory_reserved(),
		"max_allocated_bytes": torch.cuda.max_memory_allocated(),
		"summary": torch.cuda.memory_summary(abbreviated=True),
	}


def initial_status(experiment, seed):
	return {
		"experiment": experiment, "seed": seed, "state": "pending",
		"created_at": now_iso(), "started_at": None, "ended_at": None,
		"last_completed_epoch": 0, "best_val_loss": None,
		"last_checkpoint": None, "best_checkpoint": None,
		"latest_metrics": {}, "error": None,
	}


def load_or_create_status(path, experiment, seed):
	path = Path(path)
	if path.is_file():
		return json.loads(path.read_text(encoding="utf-8"))
	status = initial_status(experiment, seed)
	atomic_json(path, status)
	return status


def autocast_context(device, enabled):
	if device.type == "cuda":
		return torch.amp.autocast("cuda", enabled=enabled)
	return contextlib.nullcontext()


def pad_input(tensor, data_config):
	front, back = data_config["z_pad"]
	return F.pad(tensor, (0, 0, 0, 0, front, back), mode="reflect")


def crop_output(tensor, data_config):
	front = data_config["z_pad"][0]
	depth = data_config["volume_shape"][0]
	return tensor[:, :, front:front + depth]


def make_loader(dataset, batch_size, workers, shuffle, seed, epoch=0):
	generator = torch.Generator()
	generator.manual_seed(int(seed) + int(epoch))
	return DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
		pin_memory=torch.cuda.is_available(), generator=generator,
		persistent_workers=workers > 0,
	)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, config_summary):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"model": model.state_dict(), "optimizer": optimizer.state_dict(),
		"scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
		"epoch": int(epoch), "best_val_loss": float(best_val),
		"rng_state": rng_state(), "config_summary": config_summary,
		"saved_at": now_iso(),
	}
	temp = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, str(temp))
	os.replace(str(temp), str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
	checkpoint = torch.load(str(path), map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model"])
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint["optimizer"])
	if scheduler is not None:
		scheduler.load_state_dict(checkpoint["scheduler"])
	if scaler is not None and "scaler" in checkpoint:
		scaler.load_state_dict(checkpoint["scaler"])
	restore_rng_state(checkpoint.get("rng_state"))
	return checkpoint


def append_epoch_csv(path, row):
	path = Path(path)
	exists = path.is_file()
	with path.open("a", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(row))
		if not exists:
			writer.writeheader()
		writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config):
	model.train()
	total, count = 0.0, 0
	accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
	optimizer.zero_grad(set_to_none=True)
	for batch_index, (dirty, clean) in enumerate(loader):
		dirty = dirty.to(device, non_blocking=True)
		clean = clean.to(device, non_blocking=True)
		with autocast_context(device, config["training"]["amp"]):
			prediction = crop_output(model(pad_input(dirty, config["data"])), config["data"])
			loss = criterion(prediction, clean)
		scaler.scale(loss / accumulation).backward()
		if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
		total += float(loss.item()) * dirty.shape[0]
		count += dirty.shape[0]
	return total / count


def validate_epoch(model, loader, criterion, device, config):
	model.eval()
	total, count, rows = 0.0, 0, []
	clip_min, clip_max = config["evaluation"]["clip_predictions"]
	with torch.no_grad():
		for dirty, clean in loader:
			dirty = dirty.to(device, non_blocking=True)
			clean = clean.to(device, non_blocking=True)
			with autocast_context(device, config["training"]["amp"]):
				prediction = crop_output(model(pad_input(dirty, config["data"])), config["data"])
				loss = criterion(prediction, clean)
			total += float(loss.item()) * dirty.shape[0]
			count += dirty.shape[0]
			for pred_item, clean_item in zip(prediction, clean):
				pred_np = np.clip(pred_item.squeeze().float().cpu().numpy(), clip_min, clip_max)
				clean_np = clean_item.squeeze().float().cpu().numpy()
				rows.append(volume_metrics(pred_np, clean_np, None))
	summary = summarize_metric_rows(rows)
	summary["loss"] = total / count
	return summary


def plot_training_curve(csv_path, output_path):
	if not Path(csv_path).is_file():
		return
	with Path(csv_path).open(encoding="utf-8") as stream:
		rows = list(csv.DictReader(stream))
	if not rows:
		return
	epochs = [int(r["epoch"]) for r in rows]
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))
	axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
	axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label="validation")
	axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training/validation loss")
	axes[0].legend()
	axes[1].plot(epochs, [float(r["val_gradient_mae"]) for r in rows])
	axes[1].set(xlabel="Epoch", ylabel="Gradient MAE", title="Validation structure error")
	fig.tight_layout()
	fig.savefig(str(output_path), dpi=160)
	plt.close(fig)


def train_experiment(config, experiment_name, seed, run_dir, device, resume=True,
					 max_epochs_override=None, subset_size=None, simulate_failure=False):
	run_dir = Path(run_dir)
	run_dir.mkdir(parents=True, exist_ok=True)
	status_path = run_dir / "status.json"
	status = load_or_create_status(status_path, experiment_name, seed)
	if status["state"] == "completed" and resume:
		return status
	if not resume and any((run_dir / name).exists() for name in ("last.pt", "best_val.pt", "epoch_metrics.csv")):
		raise FileExistsError("--no-resume refuses to overwrite an existing run: {}".format(run_dir))
	experiment = config["experiments"][experiment_name]
	snapshot = {
		"experiment_name": experiment_name, "experiment": experiment,
		"seed": seed, "common": {k: config[k] for k in ("paths", "data", "training", "model", "evaluation")},
	}
	atomic_json(run_dir / "config_snapshot.json", snapshot)
	set_seed(seed, config["training"]["deterministic"])
	status.update({"state": "running", "started_at": status.get("started_at") or now_iso(), "ended_at": None, "error": None})
	atomic_json(status_path, status)
	log_path = run_dir / "run.log"
	try:
		train_dataset = PairedRawDataset(config["paths"]["dirty_train"], config["paths"]["clean_train"], config["data"]["volume_shape"])
		val_dataset = PairedRawDataset(config["paths"]["dirty_val"], config["paths"]["clean_val"], config["data"]["volume_shape"])
		if subset_size:
			train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
			val_dataset = Subset(val_dataset, range(min(subset_size, len(val_dataset))))
		atomic_json(run_dir / "dataset_roles.json", {
			"train": {"dirty": config["paths"]["dirty_train"], "clean": config["paths"]["clean_train"], "samples": len(train_dataset), "role": "optimization only"},
			"validation": {"dirty": config["paths"]["dirty_val"], "clean": config["paths"]["clean_val"], "samples": len(val_dataset), "role": "checkpoint selection/early stopping"},
			"independent_test": {"dirty": config["paths"]["dirty_independent_test"], "clean": config["paths"]["clean_independent_test"], "role": "final evaluation only; never loaded by training"},
		})
		model = build_model(experiment, config["model"]).to(device)
		criterion = build_loss(experiment["loss"])
		optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
		epochs = int(max_epochs_override or config["training"]["epochs"])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
		scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config["training"]["amp"]))
		last_path, best_path = run_dir / "last.pt", run_dir / "best_val.pt"
		start_epoch, best_val = 1, float("inf")
		if resume and last_path.is_file():
			checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
			start_epoch = int(checkpoint["epoch"]) + 1
			best_val = float(checkpoint["best_val_loss"])
		else:
			# Epoch-zero checkpoint guarantees a clean recovery point even if the
			# first epoch is interrupted before any batch/epoch can be committed.
			save_checkpoint(last_path, model, optimizer, scheduler, scaler, 0, best_val, snapshot)
		with log_path.open("a", encoding="utf-8") as log:
			log.write("{} start/resume at epoch {} on {}\n".format(now_iso(), start_epoch, device))
			for epoch in range(start_epoch, epochs + 1):
				started = time.perf_counter()
				loader = make_loader(train_dataset, config["training"].get("micro_batch_size", config["training"]["batch_size"]), config["training"]["num_workers"], True, seed, epoch)
				val_loader = make_loader(val_dataset, config["training"]["eval_batch_size"], config["training"]["num_workers"], False, seed)
				train_loss = train_one_epoch(model, loader, criterion, optimizer, scaler, device, config)
				val = validate_epoch(model, val_loader, criterion, device, config)
				scheduler.step()
				row = {
					"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
					"val_loss": val["loss"], "val_rmse": val["rmse_mean"], "val_psnr": val["psnr_mean"],
					"val_ssim": val["ssim_mean"], "val_gradient_mae": val["gradient_mae_mean"],
					"epoch_seconds": time.perf_counter() - started,
				}
				append_epoch_csv(run_dir / "epoch_metrics.csv", row)
				best_val = min(best_val, val["loss"])
				save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_val, snapshot)
				if val["loss"] <= best_val:
					save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_val, snapshot)
				status.update({"last_completed_epoch": epoch, "best_val_loss": best_val,
					"last_checkpoint": str(last_path.resolve()), "best_checkpoint": str(best_path.resolve()), "latest_metrics": row})
				atomic_json(status_path, status)
				log.write(json.dumps(row) + "\n"); log.flush()
				if simulate_failure:
					raise RuntimeError("Intentional smoke-test exception")
		plot_training_curve(run_dir / "epoch_metrics.csv", run_dir / "training_curves.png")
		status.update({"state": "running", "training_completed": True, "training_completed_at": now_iso()})
		atomic_json(status_path, status)
		return status
	except KeyboardInterrupt as exc:
		write_failure(run_dir, status, status_path, exc, "interrupted")
		raise
	except Exception as exc:
		write_failure(run_dir, status, status_path, exc, "failed")
		raise


def write_failure(run_dir, status, status_path, exc, state):
	error = {
		"time": now_iso(), "exception_type": type(exc).__name__, "message": str(exc),
		"current_epoch": status.get("last_completed_epoch", 0),
		"latest_metrics": status.get("latest_metrics", {}), "gpu_memory": gpu_memory_summary(),
		"traceback": traceback.format_exc(),
	}
	Path(run_dir, "error.log").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
	status.update({"state": state, "ended_at": now_iso(), "error": error})
	atomic_json(status_path, status)


def benchmark_model(model, sample_padded, device, config):
	model.eval()
	params = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	result = {"params": params, "trainable_params": trainable,
		"flops_definition": "2 * THOP MACs for one padded [1,1,64,256,256] forward; executed path only",
		"vram_definition": "peak allocated CUDA bytes during one AMP forward; model/input allocations reset before measurement"}
	try:
		from thop import profile
		with torch.no_grad():
			macs, _ = profile(model, inputs=(sample_padded,), verbose=False)
		result["macs"] = float(macs); result["flops"] = float(2.0 * macs)
	except Exception as exc:
		result.update({"macs": None, "flops": None, "flops_error": "{}: {}".format(type(exc).__name__, exc)})
	if device.type == "cuda":
		torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
	with torch.no_grad(), autocast_context(device, config["training"]["amp"]):
		_ = model(sample_padded)
	if device.type == "cuda":
		torch.cuda.synchronize(); result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
	else:
		result["peak_vram_bytes"] = None
	warmup = int(config["evaluation"]["warmup_runs"])
	runs = int(config["evaluation"]["timed_runs"])
	with torch.no_grad():
		for _ in range(warmup):
			with autocast_context(device, config["training"]["amp"]): _ = model(sample_padded)
		if device.type == "cuda": torch.cuda.synchronize()
		times = []
		for _ in range(runs):
			if device.type == "cuda": torch.cuda.synchronize()
			start = time.perf_counter()
			with autocast_context(device, config["training"]["amp"]): _ = model(sample_padded)
			if device.type == "cuda": torch.cuda.synchronize()
			times.append((time.perf_counter() - start) * 1000.0)
	result["warmup_runs"] = warmup; result["timed_runs"] = runs
	result["inference_ms_mean"] = float(np.mean(times)); result["inference_ms_std"] = float(np.std(times))
	return result


def evaluate_experiment(config, experiment_name, seed, run_dir, device):
	run_dir = Path(run_dir); status_path = run_dir / "status.json"
	status = load_or_create_status(status_path, experiment_name, seed)
	best_path = run_dir / "best_val.pt"
	if not best_path.is_file():
		raise FileNotFoundError("No best validation checkpoint: {}".format(best_path))
	validation_report = validate_independent_test(
		config["paths"]["dirty_independent_test"], config["paths"]["clean_independent_test"],
		config["data"], run_dir / "independent_test_validation.json")
	dataset = PairedRawDataset(config["paths"]["dirty_independent_test"], config["paths"]["clean_independent_test"], config["data"]["volume_shape"], return_name=True)
	loader = make_loader(dataset, 1, config["training"]["num_workers"], False, seed)
	model = build_model(config["experiments"][experiment_name], config["model"], checkpoint_override=False).to(device)
	load_checkpoint(best_path, model, device=device)
	model.eval(); rows = []; prediction_dir = run_dir / "predictions"; prediction_dir.mkdir(exist_ok=True)
	clip_min, clip_max = config["evaluation"]["clip_predictions"]
	first_padded = None
	with torch.no_grad():
		for dirty, clean, names in loader:
			dirty = dirty.to(device); padded = pad_input(dirty, config["data"])
			if first_padded is None: first_padded = padded
			with autocast_context(device, config["training"]["amp"]):
				prediction = crop_output(model(padded), config["data"])
			pred_np = np.clip(prediction[0, 0].float().cpu().numpy(), clip_min, clip_max)
			clean_np = clean[0, 0].numpy()
			row = {"filename": names[0]}; row.update(volume_metrics(pred_np, clean_np, config["evaluation"]["copper_metrics"])); rows.append(row)
			if config["evaluation"]["save_predictions"]:
				pred_np.astype(np.float32).tofile(str(prediction_dir / names[0].replace("_dirty.raw", "_pred.raw")))
	metrics = summarize_metric_rows(rows)
	benchmark = benchmark_model(model, first_padded, device, config)
	result = {"experiment": experiment_name, "id": config["experiments"][experiment_name]["id"], "seed": seed,
		"best_val_loss": status.get("best_val_loss"), "test_metrics": metrics, "efficiency": benchmark,
		"test_validation": validation_report, "evaluated_at": now_iso()}
	atomic_json(run_dir / "test_results.json", result)
	with (run_dir / "test_per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
	status.update({"state": "completed", "ended_at": now_iso(), "test_results": str((run_dir / "test_results.json").resolve())})
	atomic_json(status_path, status)
	return result


def collect_results(config, output_root):
	output_root = Path(output_root); summary_dir = output_root / "summary"; summary_dir.mkdir(parents=True, exist_ok=True)
	rows, failures = [], []
	for name, exp in config["experiments"].items():
		base = output_root / name
		for seed_dir in sorted(base.glob("seed_*")) if base.is_dir() else []:
			status_path = seed_dir / "status.json"
			if not status_path.is_file(): continue
			status = json.loads(status_path.read_text(encoding="utf-8"))
			row = {"id": exp["id"], "experiment": name, "seed": status.get("seed"), "status": status.get("state"), "best_val_loss": status.get("best_val_loss")}
			result_path = seed_dir / "test_results.json"
			if result_path.is_file():
				result = json.loads(result_path.read_text(encoding="utf-8"))
				row.update(result["test_metrics"]); row.update(result["efficiency"])
			rows.append(row)
			if status.get("state") in ("failed", "interrupted"):
				failures.append((name, status.get("seed"), status.get("error", {})))
	columns = sorted({key for row in rows for key in row}) if rows else ["id", "experiment", "seed", "status"]
	with (summary_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
	metric_cols = ["rmse_mean", "psnr_mean", "ssim_mean", "gradient_mae_mean", "copper_dice_mean", "copper_iou_mean", "params", "flops", "peak_vram_bytes", "inference_ms_mean"]
	lines = ["| ID | Experiment | Seed | Status | " + " | ".join(metric_cols) + " |", "|" + "---|" * (4 + len(metric_cols))]
	for row in rows:
		lines.append("| {} | {} | {} | {} | {} |".format(row.get("id", ""), row.get("experiment", ""), row.get("seed", ""), row.get("status", ""), " | ".join(str(row.get(c, "")) for c in metric_cols)))
	(summary_dir / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	failure_lines = ["# Failed or interrupted runs", ""]
	for name, seed, error in failures:
		failure_lines.extend(["## {} / seed {}".format(name, seed), "", "- Error: `{}: {}`".format(error.get("exception_type", "unknown"), error.get("message", "")), "- Resume: `python run_ablation.py --experiments {} --seeds {}`".format(name, seed), ""])
	if not failures: failure_lines.append("No failed or interrupted runs were found.")
	(summary_dir / "failures.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
	# Multi-seed aggregate is kept separate to avoid mixing per-run and paper rows.
	aggregates = []
	for name in config["experiments"]:
		complete = [r for r in rows if r.get("experiment") == name and r.get("status") == "completed"]
		if not complete: continue
		agg = {"experiment": name, "n_seeds": len(complete)}
		for key in metric_cols:
			values = [float(r[key]) for r in complete if r.get(key) is not None]
			if values: agg[key + "_seed_mean"], agg[key + "_seed_std"] = float(np.mean(values)), float(np.std(values))
		aggregates.append(agg)
	atomic_json(summary_dir / "multi_seed_summary.json", aggregates)
	if aggregates:
		agg_columns = sorted({key for row in aggregates for key in row})
		with (summary_dir / "multi_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
			writer = csv.DictWriter(stream, fieldnames=agg_columns); writer.writeheader(); writer.writerows(aggregates)
	else:
		(summary_dir / "multi_seed_summary.csv").write_text("experiment,n_seeds\n", encoding="utf-8-sig")
	return rows


def generate_qualitative_if_ready(config, output_root):
	output_root = Path(output_root); summary = output_root / "summary"
	test = PairedRawDataset(config["paths"]["dirty_independent_test"], config["paths"]["clean_independent_test"], config["data"]["volume_shape"], True)
	dirty, clean = test.pairs[0]
	predictions = {}
	for name in config["experiments"]:
		seed_dirs = sorted((output_root / name).glob("seed_*"))
		if seed_dirs:
			predictions[name] = seed_dirs[0] / "predictions" / dirty.name.replace("_dirty.raw", "_pred.raw")
	return generate_comparison(clean, dirty, predictions, summary / "qualitative_rois.json", summary / "qualitative_comparison.png")


def smoke_test(config, output_root, device):
	"""Small spatial/data smoke test; never writes into formal result directories."""
	root = Path(output_root) / "_smoke_test"; root.mkdir(parents=True, exist_ok=True)
	report = {"started_at": now_iso(), "device": str(device), "checks": {}}
	set_seed(config["training"]["seed"], config["training"]["deterministic"])
	# 64x64x128 keeps every Swin stage valid after CS-Net's 2x compression.
	dummy = torch.rand(1, 1, 64, 64, 128, device=device)
	for name, experiment in config["experiments"].items():
		try:
			model = build_model(experiment, config["model"], checkpoint_override=False).to(device)
			model.train(); target = torch.rand_like(dummy)
			prediction = model(dummy); loss = build_loss(experiment["loss"])(prediction, target); loss.backward()
			report["checks"][name] = {"ok": True, "output_shape": list(prediction.shape), "loss": float(loss.item()), "params": sum(p.numel() for p in model.parameters())}
			del model, prediction, loss
			if device.type == "cuda": torch.cuda.empty_cache()
		except Exception as exc:
			report["checks"][name] = {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
	# Real independent-test integrity check.
	try:
		report["dataset_validation"] = validate_independent_test(config["paths"]["dirty_independent_test"], config["paths"]["clean_independent_test"], config["data"], root / "independent_test_validation.json")
	except Exception as exc:
		report["dataset_validation"] = {"valid": False, "error": str(exc)}
	# Checkpoint round-trip with a tiny model, including optimizer/scheduler/RNG.
	tiny = torch.nn.Conv3d(1, 1, 1).to(device); opt = torch.optim.AdamW(tiny.parameters()); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2); scaler = torch.amp.GradScaler("cuda", enabled=False)
	checkpoint = root / "roundtrip.pt"; save_checkpoint(checkpoint, tiny, opt, sched, scaler, 1, 0.5, {"smoke": True}); loaded = load_checkpoint(checkpoint, tiny, opt, sched, scaler, device)
	report["checkpoint_roundtrip"] = {"ok": loaded["epoch"] == 1 and loaded["best_val_loss"] == 0.5}
	# Exercise exact failure/status writer without launching a formal training run.
	failure_dir = root / "intentional_failure"; failure_dir.mkdir(exist_ok=True); status = initial_status("intentional_failure", config["training"]["seed"]); status_path = failure_dir / "status.json"
	try:
		raise RuntimeError("Intentional smoke-test exception")
	except RuntimeError as exc:
		write_failure(failure_dir, status, status_path, exc, "failed")
	stored = json.loads(status_path.read_text(encoding="utf-8")); report["failure_logging"] = {"ok": stored["state"] == "failed" and (failure_dir / "error.log").is_file()}
	collect_results(config, root); report["summary_generation"] = {"ok": (root / "summary" / "ablation_results.csv").is_file()}
	report["finished_at"] = now_iso(); report["ok"] = all(v.get("ok", False) for v in report["checks"].values()) and report["dataset_validation"].get("valid", False) and report["checkpoint_roundtrip"]["ok"] and report["failure_logging"]["ok"] and report["summary_generation"]["ok"]
	atomic_json(root / "smoke_report.json", report)
	return report


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default=str(CONFIG_PATH))
	parser.add_argument("--experiments", nargs="+", default=["all"])
	parser.add_argument("--seeds", nargs="+", type=int)
	parser.add_argument("--mode", choices=("train-eval", "eval-only", "summary-only", "smoke-test"), default="train-eval")
	parser.add_argument("--output-root")
	parser.add_argument("--no-resume", action="store_true")
	parser.add_argument("--fail-fast", action="store_true")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	return parser.parse_args()


def main():
	args = parse_args(); config = load_config(args.config); output_root = resolve_output_root(config, args.config, args.output_root); output_root.mkdir(parents=True, exist_ok=True)
	seeds = args.seeds or [config["training"]["seed"]]; experiments = select_experiments(config, args.experiments); device = torch.device(args.device)
	manifest_path = output_root / "run_manifest.json"
	invocation = {"time": now_iso(), "experiments": experiments, "seeds": seeds, "mode": args.mode, "command": sys.argv}
	if manifest_path.is_file():
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		manifest.setdefault("invocations", []).append(invocation)
	else:
		manifest = {"created_at": now_iso(), "config": str(Path(args.config).resolve()), "output_root": str(output_root), "environment": environment_info(config["training"]["deterministic"]), "invocations": [invocation]}
	atomic_json(manifest_path, manifest)
	if args.mode == "smoke-test":
		report = smoke_test(config, output_root, device); print(json.dumps(report, indent=2, ensure_ascii=False)); return 0 if report["ok"] else 2
	if args.mode == "summary-only":
		collect_results(config, output_root); generate_qualitative_if_ready(config, output_root); return 0
	for name in experiments:
		for seed in seeds:
			run_dir = output_root / name / "seed_{}".format(seed)
			try:
				if args.mode == "train-eval":
					status = train_experiment(config, name, seed, run_dir, device, resume=not args.no_resume)
					if status.get("state") == "completed": continue
				evaluate_experiment(config, name, seed, run_dir, device)
			except KeyboardInterrupt:
				print("Interrupted {} seed {}".format(name, seed), file=sys.stderr)
				status_path = run_dir / "status.json"
				status = load_or_create_status(status_path, name, seed)
				if status.get("state") not in ("failed", "interrupted"):
					write_failure(run_dir, status, status_path, KeyboardInterrupt(), "interrupted")
				if args.fail_fast: raise
			except Exception:
				traceback.print_exc()
				status_path = run_dir / "status.json"
				status = load_or_create_status(status_path, name, seed)
				if status.get("state") not in ("failed", "interrupted"):
					exc = sys.exc_info()[1]
					write_failure(run_dir, status, status_path, exc, "failed")
				if args.fail_fast: raise
	collect_results(config, output_root); generate_qualitative_if_ready(config, output_root)
	return 0


if __name__ == "__main__":
	torch.multiprocessing.freeze_support()
	raise SystemExit(main())
