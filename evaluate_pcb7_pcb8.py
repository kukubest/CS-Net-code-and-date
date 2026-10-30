"""Evaluate all formal PCB reconstruction baselines on independent PCB7/PCB8.

This inference-only runner never changes checkpoints or source datasets.  Each
board/method writes an independent CSV and JSON result, so a stopped run can be
resumed by launching the same command again.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
import astra
from monai.networks.nets import UNet, UNETR

from ablation_data import validate_independent_test
from ablation_metrics import summarize_metric_rows, volume_metrics
from ablation_models import build_model
from generate_fdk_baseline import create_cl_vector_geometry


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(r"G:\CT_RE\dataset")
CONFIG_PATH = ROOT / "ablation_config.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "pcb7_pcb8_evaluation_20261019"
SHAPE = (40, 256, 256)
PAD = (0, 0, 0, 0, 12, 12)
RAW_BYTES = int(np.prod(SHAPE)) * 4
SEED = 20260916
COPPER = {"enabled": True, "prediction_threshold": 0.91}


def now_iso():
	return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
	path = Path(path)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	tmp.replace(path)


def load_config():
	return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def paired_files(board_id):
	dirty_dir = DATASET_ROOT / "Final_Adaptive_Dirty" / f"independent_test_pcb{board_id}"
	clean_dir = DATASET_ROOT / "Final_Adaptive_GT" / f"independent_test_pcb{board_id}"
	pairs = []
	for dirty in sorted(dirty_dir.glob("*_dirty.raw")):
		clean = clean_dir / dirty.name.replace("_dirty.raw", ".raw")
		if not clean.is_file():
			raise FileNotFoundError(clean)
		if dirty.stat().st_size != RAW_BYTES or clean.stat().st_size != RAW_BYTES:
			raise ValueError(f"Unexpected RAW byte count: {dirty} / {clean}")
		pairs.append((dirty, clean))
	if len(pairs) != 200:
		raise RuntimeError(f"PCB{board_id}: expected 200 pairs, found {len(pairs)}")
	return dirty_dir, clean_dir, pairs


def load_raw(path):
	return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


def safe_name(value):
	return value.lower().replace(" ", "_").replace("-", "_")


def write_method_result(board_dir, method, rows, metadata):
	stem = safe_name(method)
	csv_path = board_dir / f"{stem}_per_sample.csv"
	with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	payload = {
		"method": method,
		"samples": len(rows),
		"summary": summarize_metric_rows(rows),
		"metadata": metadata,
		"finished_at": now_iso(),
	}
	atomic_json(board_dir / f"{stem}_summary.json", payload)
	return payload


def evaluate_iterator(board_dir, method, iterator, total, metadata):
	rows = []
	started = time.perf_counter()
	for index, (filename, prediction, target) in enumerate(iterator, 1):
		prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
		row = {"filename": filename}
		row.update(volume_metrics(prediction, target, COPPER))
		rows.append(row)
		if index == 1 or index % 20 == 0:
			print(
				f"[PCB{metadata['board_id']}][{method}] {index}/{total} "
				f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.5f}",
				flush=True,
			)
	metadata = dict(metadata)
	metadata["wall_time_seconds"] = time.perf_counter() - started
	return write_method_result(board_dir, method, rows, metadata)


def sirt_iterator(pairs):
	for dirty, clean in pairs:
		yield clean.name, load_raw(dirty), load_raw(clean)


def add_poisson_noise(sinogram, i0=5e4):
	photons = i0 * np.exp(-sinogram)
	noisy = np.random.poisson(photons)
	noisy[noisy == 0] = 1
	return -np.log(noisy / i0)


def fdk_iterator(pairs):
	np.random.seed(SEED)
	vol_geom = astra.create_vol_geom(SHAPE[2], SHAPE[1], SHAPE[0])
	vectors = create_cl_vector_geometry(180, 45.0, 500.0, 500.0, 1.0)
	proj_geom = astra.create_proj_geom("cone_vec", 300, 300, vectors)
	for _, clean in pairs:
		target = load_raw(clean)
		sino_id = noisy_id = recon_id = alg_id = None
		try:
			sino_id, sinogram = astra.create_sino3d_gpu(target, proj_geom, vol_geom)
			noisy = add_poisson_noise(sinogram, 5e4)
			noisy_id = astra.data3d.create("-sino", proj_geom, noisy)
			recon_id = astra.data3d.create("-vol", vol_geom)
			cfg = astra.astra_dict("FDK_CUDA")
			cfg["ReconstructionDataId"] = recon_id
			cfg["ProjectionDataId"] = noisy_id
			alg_id = astra.algorithm.create(cfg)
			astra.algorithm.run(alg_id)
			prediction = astra.data3d.get(recon_id)
		finally:
			if alg_id is not None:
				astra.algorithm.delete(alg_id)
			for object_id in (recon_id, sino_id, noisy_id):
				if object_id is not None:
					astra.data3d.delete(object_id)
		prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
		yield clean.name, prediction.astype(np.float32), target


def baseline_model(name):
	if name == "3D U-Net":
		return UNet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=(16, 32, 64, 128, 256),
			strides=(2, 2, 2, 2),
			num_res_units=2,
			norm="instance",
		)
	if name == "UNETR":
		return UNETR(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			img_size=(64, 256, 256),
			feature_size=16,
			hidden_size=768,
			mlp_dim=3072,
			num_heads=12,
			norm_name="instance",
			res_block=True,
		)
	raise KeyError(name)


def learned_specs(config):
	specs = [
		{
			"method": "3D U-Net",
			"kind": "baseline",
			"checkpoint": ROOT / "checkpoints" / "3d_unet_best.pth",
		},
		{
			"method": "UNETR",
			"kind": "baseline",
			"checkpoint": ROOT / "checkpoints" / "unetr_baseline_best.pth",
		},
	]
	seed_map = {
		"swin_unetr_baseline": (20260916, 20260917, 20260918),
		"csnet_no_compression": (20260916,),
		"csnet_compressed_no_hr_skip": (20260916, 20260917, 20260918),
		"csnet_l1": (20260916, 20260917, 20260918),
		"csnet_mse": (20260916,),
	}
	for experiment, seeds in seed_map.items():
		exp = config["experiments"][experiment]
		for seed in seeds:
			specs.append(
				{
					"method": f"{exp['id']}_{experiment}_seed_{seed}",
					"kind": "ablation",
					"experiment": experiment,
					"seed": seed,
					"checkpoint": ROOT / "outputs" / "ablation" / experiment / f"seed_{seed}" / "best_val.pt",
				}
			)
	return specs


def load_learned_model(spec, config, device):
	if not spec["checkpoint"].is_file():
		raise FileNotFoundError(spec["checkpoint"])
	if spec["kind"] == "baseline":
		model = baseline_model(spec["method"])
		state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
	else:
		model = build_model(
			config["experiments"][spec["experiment"]],
			config["model"],
			checkpoint_override=False,
		)
		checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
		state = checkpoint["model"]
	model.load_state_dict(state, strict=True)
	model.to(device).eval()
	return model


def learned_iterator(spec, config, pairs, device, timing):
	model = load_learned_model(spec, config, device)
	times = []
	peaks = []
	try:
		for dirty, clean in pairs:
			x = torch.from_numpy(load_raw(dirty)).unsqueeze(0).unsqueeze(0).to(device)
			target = load_raw(clean)
			x = F.pad(x, PAD, mode="reflect")
			if device.type == "cuda":
				torch.cuda.reset_peak_memory_stats()
				torch.cuda.synchronize()
			started = time.perf_counter()
			with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
				prediction = model(x)[:, :, 12:52]
			if device.type == "cuda":
				torch.cuda.synchronize()
			times.append((time.perf_counter() - started) * 1000.0)
			peaks.append(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
			yield clean.name, prediction[0, 0].float().cpu().numpy(), target
	finally:
		timing.update(
			{
				"per_sample_forward_ms_mean": float(np.mean(times)) if times else None,
				"per_sample_forward_ms_std": float(np.std(times)) if times else None,
				"peak_allocated_bytes_max": int(max(peaks)) if peaks else None,
			}
		)
		del model
		gc.collect()
		if device.type == "cuda":
			torch.cuda.empty_cache()


def result_complete(board_dir, method):
	stem = safe_name(method)
	json_path = board_dir / f"{stem}_summary.json"
	csv_path = board_dir / f"{stem}_per_sample.csv"
	if not json_path.is_file() or not csv_path.is_file():
		return False
	try:
		payload = json.loads(json_path.read_text(encoding="utf-8"))
		with csv_path.open(encoding="utf-8-sig") as stream:
			row_count = sum(1 for _ in csv.DictReader(stream))
		return payload.get("samples") == 200 and row_count == 200
	except Exception:
		return False


def collect_summary(output_root):
	rows = []
	for board_dir in sorted(output_root.glob("pcb*")):
		if not board_dir.is_dir():
			continue
		board_id = int(board_dir.name.replace("pcb", ""))
		for path in sorted(board_dir.glob("*_summary.json")):
			payload = json.loads(path.read_text(encoding="utf-8"))
			row = {"board_id": board_id, "method": payload["method"], "samples": payload["samples"]}
			row.update(payload["summary"])
			metadata = payload.get("metadata", {})
			row["checkpoint"] = metadata.get("checkpoint")
			row["forward_ms_mean"] = metadata.get("timing", {}).get("per_sample_forward_ms_mean")
			row["peak_allocated_bytes"] = metadata.get("timing", {}).get("peak_allocated_bytes_max")
			rows.append(row)
	if not rows:
		return
	summary_dir = output_root / "summary"
	summary_dir.mkdir(exist_ok=True)
	columns = sorted({key for row in rows for key in row})
	with (summary_dir / "pcb7_pcb8_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=columns)
		writer.writeheader()
		writer.writerows(rows)
	metric_keys = ("rmse_mean", "psnr_mean", "ssim_mean", "gradient_mae_mean", "copper_dice_mean", "copper_iou_mean")
	lines = [
		"| PCB | Method | RMSE↓ | PSNR↑ | SSIM↑ | Gradient MAE↓ | Copper Dice↑ | Copper IoU↑ |",
		"|---:|---|---:|---:|---:|---:|---:|---:|",
	]
	for row in sorted(rows, key=lambda item: (item["board_id"], item["method"])):
		values = [row.get(key, float("nan")) for key in metric_keys]
		lines.append(
			"| {} | {} | {:.6f} | {:.3f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
				row["board_id"], row["method"], *values
			)
		)
	(summary_dir / "pcb7_pcb8_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

	# Seed aggregates are kept separate from individual checkpoint results.
	aggregates = []
	for board_id in (7, 8):
		board_rows = [row for row in rows if row["board_id"] == board_id]
		for experiment_id in ("A", "D"):
			selected = [row for row in board_rows if row["method"].startswith(experiment_id + "_")]
			if not selected:
				continue
			aggregate = {"board_id": board_id, "experiment_id": experiment_id, "n_seeds": len(selected)}
			for key in metric_keys:
				values = np.asarray([row[key] for row in selected], dtype=np.float64)
				aggregate[key + "_seed_mean"] = float(values.mean())
				aggregate[key + "_seed_std"] = float(values.std(ddof=0))
			aggregates.append(aggregate)
	atomic_json(summary_dir / "multi_seed_summary.json", aggregates)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--boards", nargs="+", type=int, default=[7, 8], choices=[7, 8])
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--skip-fdk", action="store_true")
	parser.add_argument("--methods", nargs="*", help="Optional exact method names to run")
	return parser.parse_args()


def main():
	args = parse_args()
	config = load_config()
	output_root = args.output.resolve()
	output_root.mkdir(parents=True, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	specs = learned_specs(config)
	if args.methods:
		requested = set(args.methods)
		specs = [spec for spec in specs if spec["method"] in requested]
	manifest_path = output_root / "run_manifest.json"
	manifest = {
		"started_at": now_iso(),
		"updated_at": now_iso(),
		"state": "running",
		"boards": args.boards,
		"shape": list(SHAPE),
		"seed": SEED,
		"device": str(device),
		"gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
		"python": platform.python_version(),
		"torch": torch.__version__,
		"astra": astra.__version__,
		"methods": ["SIRT"] + ([] if args.skip_fdk else ["FDK"]) + [spec["method"] for spec in specs],
		"completed": [],
		"failures": [],
	}
	if manifest_path.is_file():
		previous = json.loads(manifest_path.read_text(encoding="utf-8"))
		manifest["completed"] = previous.get("completed", [])
		manifest["failures"] = previous.get("failures", [])
	atomic_json(manifest_path, manifest)
	try:
		for board_id in args.boards:
			board_dir = output_root / f"pcb{board_id}"
			board_dir.mkdir(exist_ok=True)
			dirty_dir, clean_dir, pairs = paired_files(board_id)
			validation_config = dict(config["data"])
			validation_config["expected_test_samples"] = 200
			validate_independent_test(
				dirty_dir,
				clean_dir,
				validation_config,
				board_dir / "dataset_validation.json",
			)
			common = {"board_id": board_id, "dirty_dir": str(dirty_dir), "clean_dir": str(clean_dir)}
			jobs = [("SIRT", lambda: sirt_iterator(pairs), {"source": "existing SIRT3D_CUDA reconstruction"})]
			if not args.skip_fdk:
				jobs.append(
					(
						"FDK",
						lambda: fdk_iterator(pairs),
						{
							"geometry": "180 projections; 45 deg; SO=500; OD=500; detector=300x300; spacing=1",
							"noise": "Poisson I0=5e4; RNG reset to 20260916 per board",
						},
					)
				)
			for method, iterator_factory, extra in jobs:
				key = f"pcb{board_id}/{method}"
				if args.methods and method not in set(args.methods):
					continue
				if result_complete(board_dir, method):
					print(f"[SKIP] {key} already complete", flush=True)
					continue
				try:
					metadata = {**common, **extra}
					evaluate_iterator(board_dir, method, iterator_factory(), len(pairs), metadata)
					if key not in manifest["completed"]:
						manifest["completed"].append(key)
				except Exception as exc:
					manifest["failures"].append(
						{
							"key": key,
							"time": now_iso(),
							"type": type(exc).__name__,
							"message": str(exc),
							"traceback": traceback.format_exc(),
						}
					)
					print(f"[FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)
				manifest["updated_at"] = now_iso()
				atomic_json(manifest_path, manifest)

			for spec in specs:
				method = spec["method"]
				key = f"pcb{board_id}/{method}"
				if result_complete(board_dir, method):
					print(f"[SKIP] {key} already complete", flush=True)
					continue
				try:
					timing = {}
					metadata = {
						**common,
						"checkpoint": str(spec["checkpoint"].resolve()),
						"kind": spec["kind"],
						"seed": spec.get("seed"),
						"timing": timing,
					}
					iterator = learned_iterator(spec, config, pairs, device, timing)
					evaluate_iterator(board_dir, method, iterator, len(pairs), metadata)
					if key not in manifest["completed"]:
						manifest["completed"].append(key)
				except Exception as exc:
					manifest["failures"].append(
						{
							"key": key,
							"time": now_iso(),
							"type": type(exc).__name__,
							"message": str(exc),
							"traceback": traceback.format_exc(),
						}
					)
					print(f"[FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)
				finally:
					manifest["updated_at"] = now_iso()
					atomic_json(manifest_path, manifest)
		collect_summary(output_root)
		manifest["state"] = "completed" if not manifest["failures"] else "completed_with_failures"
		manifest["finished_at"] = now_iso()
		manifest["updated_at"] = now_iso()
		atomic_json(manifest_path, manifest)
	except KeyboardInterrupt:
		manifest["state"] = "interrupted"
		manifest["updated_at"] = now_iso()
		atomic_json(manifest_path, manifest)
		raise
	print(output_root)


if __name__ == "__main__":
	main()
