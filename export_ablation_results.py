"""Export all completed ablation training/test results without modifying them."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"G:\CT_RE\CT_DRA\outputs\ablation")
OUTPUT = Path(r"G:\CT_RE\CT_DRA\outputs\experiment_results_export_20261009")
ORDER = [
	("A", "swin_unetr_baseline"),
	("B", "csnet_no_compression"),
	("C", "csnet_compressed_no_hr_skip"),
	("D", "csnet_l1"),
	("E", "csnet_mse"),
]
COLORS = {"A": "#4472C4", "B": "#70AD47", "C": "#A5A5A5", "D": "#ED7D31", "E": "#7030A0"}


def read_csv(path):
	with Path(path).open(encoding="utf-8-sig", newline="") as stream:
		return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
	columns = columns or list(rows[0])
	with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
		writer.writeheader(); writer.writerows(rows)


def numeric(row, key):
	return float(row[key])


def main():
	if OUTPUT.exists():
		raise FileExistsError("Refusing to overwrite existing export: {}".format(OUTPUT))
	OUTPUT.mkdir(parents=True)

	summary_source = {r["experiment"]: r for r in read_csv(ROOT / "summary" / "ablation_results.csv")}
	summary_rows, epoch_rows, sample_rows = [], [], []
	for experiment_id, experiment in ORDER:
		run = ROOT / experiment / "seed_20260916"
		status = json.loads((run / "status.json").read_text(encoding="utf-8"))
		if status["state"] != "completed":
			raise RuntimeError("{} is not completed: {}".format(experiment, status["state"]))
		source = summary_source[experiment]
		summary_rows.append({
			"id": experiment_id, "experiment": experiment, "seed": int(source["seed"]),
			"status": source["status"], "best_val_loss": numeric(source, "best_val_loss"),
			"rmse_mean": numeric(source, "rmse_mean"), "rmse_std": numeric(source, "rmse_std"),
			"psnr_mean_db": numeric(source, "psnr_mean"), "psnr_std_db": numeric(source, "psnr_std"),
			"ssim_mean": numeric(source, "ssim_mean"), "ssim_std": numeric(source, "ssim_std"),
			"gradient_mae_mean": numeric(source, "gradient_mae_mean"), "gradient_mae_std": numeric(source, "gradient_mae_std"),
			"copper_dice_mean": numeric(source, "copper_dice_mean"), "copper_dice_std": numeric(source, "copper_dice_std"),
			"copper_iou_mean": numeric(source, "copper_iou_mean"), "copper_iou_std": numeric(source, "copper_iou_std"),
			"params_m": numeric(source, "params") / 1e6, "flops_g": numeric(source, "flops") / 1e9,
			"peak_vram_gib": numeric(source, "peak_vram_bytes") / 1024 ** 3,
			"inference_ms_mean": numeric(source, "inference_ms_mean"), "inference_ms_std": numeric(source, "inference_ms_std"),
			"epochs_completed": int(status["last_completed_epoch"]),
		})
		for row in read_csv(run / "epoch_metrics.csv"):
			epoch_rows.append({"id": experiment_id, "experiment": experiment, "seed": int(source["seed"]), **row})
		for row in read_csv(run / "test_per_sample.csv"):
			sample_rows.append({"id": experiment_id, "experiment": experiment, "seed": int(source["seed"]), **row})

	comparisons = []
	by_id = {row["id"]: row for row in summary_rows}
	for label, left, right, purpose in (
		("B vs D", "B", "D", "spatial compression"),
		("C vs D", "C", "D", "HR structure skip"),
		("D vs E", "D", "E", "L1 versus MSE"),
		("A vs D", "A", "D", "accuracy-efficiency trade-off"),
	):
		for metric, direction in (("rmse_mean", "lower"), ("psnr_mean_db", "higher"), ("ssim_mean", "higher"), ("gradient_mae_mean", "lower"), ("copper_dice_mean", "higher"), ("flops_g", "lower"), ("peak_vram_gib", "lower"), ("inference_ms_mean", "lower")):
			lv, rv = left and by_id[left][metric], by_id[right][metric]
			comparisons.append({"comparison": label, "purpose": purpose, "metric": metric, "preferred_direction": direction, "left_id": left, "left_value": lv, "right_id": right, "right_value": rv, "right_minus_left": rv - lv, "relative_change_pct": (rv - lv) / lv * 100.0})

	write_csv(OUTPUT / "experiment_summary.csv", summary_rows)
	write_csv(OUTPUT / "all_epoch_metrics.csv", epoch_rows)
	write_csv(OUTPUT / "all_test_samples.csv", sample_rows)
	write_csv(OUTPUT / "causal_comparisons.csv", comparisons)

	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	for experiment_id, experiment in ORDER:
		rows = [r for r in epoch_rows if r["id"] == experiment_id]
		epochs = [int(r["epoch"]) for r in rows]
		axes[0, 0].plot(epochs, [float(r["train_loss"]) for r in rows], color=COLORS[experiment_id], label=experiment_id)
		axes[0, 1].plot(epochs, [float(r["val_loss"]) for r in rows], color=COLORS[experiment_id], label=experiment_id)
		axes[1, 0].plot(epochs, [float(r["val_psnr"]) for r in rows], color=COLORS[experiment_id], label=experiment_id)
		axes[1, 1].plot(epochs, [float(r["val_gradient_mae"]) for r in rows], color=COLORS[experiment_id], label=experiment_id)
	for ax, title, ylabel in zip(axes.ravel(), ("Training loss", "Validation loss", "Validation PSNR", "Validation Gradient MAE"), ("Loss", "Loss", "PSNR (dB)", "Gradient MAE")):
		ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
	axes[0, 0].legend(ncol=5, frameon=False, loc="upper right")
	fig.tight_layout(); fig.savefig(OUTPUT / "training_validation_curves.png", dpi=220); plt.close(fig)

	fig, axes = plt.subplots(2, 3, figsize=(14, 8))
	metrics = (("rmse_mean", "RMSE", False), ("psnr_mean_db", "PSNR (dB)", True), ("ssim_mean", "SSIM", True), ("gradient_mae_mean", "Gradient MAE", False), ("copper_dice_mean", "Copper Dice", True), ("copper_iou_mean", "Copper IoU", True))
	for ax, (key, title, higher) in zip(axes.ravel(), metrics):
		values = [r[key] for r in summary_rows]; ids = [r["id"] for r in summary_rows]
		ax.bar(ids, values, color=[COLORS[i] for i in ids]); ax.set_title(title + (" ↑" if higher else " ↓")); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
		for i, value in enumerate(values): ax.text(i, value, "{:.4f}".format(value), ha="center", va="bottom", fontsize=8)
	fig.tight_layout(); fig.savefig(OUTPUT / "test_quality_metrics.png", dpi=220); plt.close(fig)

	fig, axes = plt.subplots(1, 4, figsize=(15, 4))
	for ax, (key, title) in zip(axes, (("params_m", "Params (M)"), ("flops_g", "FLOPs (G)"), ("peak_vram_gib", "Peak VRAM (GiB)"), ("inference_ms_mean", "Inference time (ms)"))):
		values = [r[key] for r in summary_rows]; ids = [r["id"] for r in summary_rows]
		ax.bar(ids, values, color=[COLORS[i] for i in ids]); ax.set_title(title + " ↓"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
	fig.tight_layout(); fig.savefig(OUTPUT / "efficiency_metrics.png", dpi=220); plt.close(fig)

	fig, ax = plt.subplots(figsize=(7.5, 5.5))
	for row in summary_rows:
		ax.scatter(row["inference_ms_mean"], row["psnr_mean_db"], s=max(70, row["peak_vram_gib"] * 65), color=COLORS[row["id"]], label=row["id"], alpha=0.85, edgecolor="white")
		ax.annotate(row["id"], (row["inference_ms_mean"], row["psnr_mean_db"]), xytext=(5, 5), textcoords="offset points")
	ax.set_xlabel("Inference time (ms, lower is better)"); ax.set_ylabel("PSNR (dB, higher is better)"); ax.set_title("Accuracy-efficiency trade-off\nBubble area represents peak VRAM"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout(); fig.savefig(OUTPUT / "accuracy_efficiency_tradeoff.png", dpi=220); plt.close(fig)

	metadata = {"source_root": str(ROOT.resolve()), "source_files_modified": False, "experiments": [x[1] for x in ORDER], "seeds": [20260916], "epochs": len(epoch_rows), "test_sample_rows": len(sample_rows), "notes": "Legacy test split supplied validation curves; PCB 6 independent test supplied final metrics."}
	(OUTPUT / "export_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
	print(json.dumps({"output": str(OUTPUT), "experiments": len(summary_rows), "epoch_rows": len(epoch_rows), "test_rows": len(sample_rows)}, indent=2))


if __name__ == "__main__":
	main()
