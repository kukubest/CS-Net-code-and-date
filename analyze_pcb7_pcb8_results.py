"""Build paired causal contrasts and figures from PCB7/PCB8 evaluation CSVs."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "outputs" / "pcb7_pcb8_evaluation_20261019"
SUMMARY_DIR = RESULT_ROOT / "summary"
SEEDS = (20260916, 20260917, 20260918)
METRICS = (
	("rmse", "RMSE", "lower"),
	("psnr", "PSNR (dB)", "higher"),
	("ssim", "SSIM", "higher"),
	("gradient_mae", "Gradient MAE", "lower"),
	("copper_dice", "Copper Dice", "higher"),
	("copper_iou", "Copper IoU", "higher"),
)


def safe(value):
	return value.lower().replace(" ", "_").replace("-", "_")


def read_rows(board_id, method):
	path = RESULT_ROOT / f"pcb{board_id}" / f"{safe(method)}_per_sample.csv"
	with path.open(encoding="utf-8-sig") as stream:
		rows = list(csv.DictReader(stream))
	if len(rows) != 200:
		raise RuntimeError(f"{path}: expected 200 rows, found {len(rows)}")
	return {
		row["filename"]: {key: float(row[key]) for key, _, _ in METRICS}
		for row in rows
	}


def mean_seed_rows(board_id, experiment_id, experiment_name):
	seed_rows = [
		read_rows(board_id, f"{experiment_id}_{experiment_name}_seed_{seed}")
		for seed in SEEDS
	]
	filenames = sorted(seed_rows[0])
	return {
		filename: {
			key: float(np.mean([rows[filename][key] for rows in seed_rows]))
			for key, _, _ in METRICS
		}
		for filename in filenames
	}


def single_rows(board_id, experiment_id, experiment_name):
	return read_rows(board_id, f"{experiment_id}_{experiment_name}_seed_20260916")


def bootstrap_ci(values, seed, repetitions=5000):
	values = np.asarray(values, dtype=np.float64)
	rng = np.random.default_rng(seed)
	indices = rng.integers(0, len(values), size=(repetitions, len(values)))
	means = values[indices].mean(axis=1)
	return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values):
	p_values = np.asarray(p_values, dtype=np.float64)
	order = np.argsort(p_values)
	adjusted = np.empty_like(p_values)
	running = 0.0
	m = len(p_values)
	for rank, index in enumerate(order):
		value = min(1.0, (m - rank) * p_values[index])
		running = max(running, value)
		adjusted[index] = running
	return adjusted


def build_contrasts():
	rows = []
	for board_id in (7, 8):
		a = mean_seed_rows(board_id, "A", "swin_unetr_baseline")
		d_multi = mean_seed_rows(board_id, "D", "csnet_l1")
		b = single_rows(board_id, "B", "csnet_no_compression")
		c_multi = mean_seed_rows(board_id, "C", "csnet_compressed_no_hr_skip")
		d = single_rows(board_id, "D", "csnet_l1")
		e = single_rows(board_id, "E", "csnet_mse")
		comparisons = (
			("A vs D", a, d_multi, "three-seed prediction metric means"),
			("B vs D", b, d, "controlled seed 20260916"),
			("C vs D", c_multi, d_multi, "three-seed prediction metric means"),
			("D vs E", e, d, "controlled seed 20260916"),
		)
		for comparison, control, treatment, scope in comparisons:
			filenames = sorted(set(control) & set(treatment))
			for metric_index, (metric, _, direction) in enumerate(METRICS):
				control_values = np.asarray([control[name][metric] for name in filenames])
				treatment_values = np.asarray([treatment[name][metric] for name in filenames])
				raw_delta = treatment_values - control_values
				improvement = raw_delta if direction == "higher" else -raw_delta
				try:
					p_value = float(wilcoxon(treatment_values, control_values, alternative="two-sided").pvalue)
				except ValueError:
					p_value = 1.0
				ci_low, ci_high = bootstrap_ci(
					improvement,
					seed=20260916 + board_id * 100 + metric_index,
				)
				rows.append(
					{
						"board_id": board_id,
						"comparison": comparison,
						"scope": scope,
						"metric": metric,
						"direction": direction,
						"control_mean": float(control_values.mean()),
						"treatment_D_mean": float(treatment_values.mean()),
						"D_improvement_mean": float(improvement.mean()),
						"bootstrap_95ci_low": ci_low,
						"bootstrap_95ci_high": ci_high,
						"wilcoxon_p": p_value,
						"n_pairs": len(filenames),
					}
				)
	adjusted = holm_adjust([row["wilcoxon_p"] for row in rows])
	for row, value in zip(rows, adjusted):
		row["holm_adjusted_p"] = float(value)
		row["significant_0.05"] = bool(value < 0.05)
	return rows


def write_contrasts(rows):
	path = SUMMARY_DIR / "causal_contrasts.csv"
	with path.open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	lines = [
		"| PCB | Contrast | Metric | D-oriented effect | 95% CI | Holm p |",
		"|---:|---|---|---:|---:|---:|",
	]
	for row in rows:
		lines.append(
			"| {} | {} | {} | {:.6g} | [{:.6g}, {:.6g}] | {:.3g} |".format(
				row["board_id"],
				row["comparison"],
				row["metric"],
				row["D_improvement_mean"],
				row["bootstrap_95ci_low"],
				row["bootstrap_95ci_high"],
				row["holm_adjusted_p"],
			)
		)
	(SUMMARY_DIR / "causal_contrasts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summary_rows():
	path = SUMMARY_DIR / "pcb7_pcb8_results.csv"
	with path.open(encoding="utf-8-sig") as stream:
		return list(csv.DictReader(stream))


def formal_values(summary_rows):
	result = {}
	for board_id in (7, 8):
		board_rows = [row for row in summary_rows if int(row["board_id"]) == board_id]
		for experiment_id in ("A", "B", "C", "D", "E"):
			selected = [row for row in board_rows if row["method"].startswith(experiment_id + "_")]
			result[(board_id, experiment_id)] = {
				metric: float(np.mean([float(row[metric + "_mean"]) for row in selected]))
				for metric, _, _ in METRICS
			}
	return result


def write_paper_table(summary_rows):
	labels = {
		"A": "Swin-UNETR (A)",
		"B": "CS-Net no compression (B)",
		"C": "CS-Net no HR skip (C)",
		"D": "CS-Net L1 (D)",
		"E": "CS-Net MSE (E)",
	}
	rows = []
	for board_id in (7, 8):
		board_rows = [row for row in summary_rows if int(row["board_id"]) == board_id]
		for experiment_id in ("A", "B", "C", "D", "E"):
			selected = [row for row in board_rows if row["method"].startswith(experiment_id + "_")]
			row = {
				"board_id": board_id,
				"experiment_id": experiment_id,
				"method": labels[experiment_id],
				"n_seeds": len(selected),
			}
			for metric, _, _ in METRICS:
				values = np.asarray([float(item[metric + "_mean"]) for item in selected])
				row[metric + "_mean"] = float(values.mean())
				row[metric + "_seed_std"] = float(values.std(ddof=0))
			times = np.asarray([float(item["forward_ms_mean"]) for item in selected])
			vram = np.asarray([float(item["peak_allocated_bytes"]) for item in selected])
			row["forward_ms_mean"] = float(times.mean())
			row["peak_vram_gib"] = float(vram.mean() / 1024 ** 3)
			rows.append(row)
	columns = list(rows[0])
	with (SUMMARY_DIR / "paper_cross_board_table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=columns)
		writer.writeheader()
		writer.writerows(rows)
	lines = [
		"| PCB | Method | Seeds | RMSE↓ | PSNR↑ | SSIM↑ | Gradient MAE↓ | Dice↑ | IoU↑ | Time (ms) | VRAM (GiB) |",
		"|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for row in rows:
		lines.append(
			"| {board_id} | {method} | {n_seeds} | {rmse_mean:.6f} | {psnr_mean:.3f} | {ssim_mean:.6f} | "
			"{gradient_mae_mean:.6f} | {copper_dice_mean:.6f} | {copper_iou_mean:.6f} | "
			"{forward_ms_mean:.2f} | {peak_vram_gib:.3f} |".format(**row)
		)
	(SUMMARY_DIR / "paper_cross_board_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics(summary_rows):
	values = formal_values(summary_rows)
	colors = {"A": "#5B6F9D", "B": "#A66E3F", "C": "#7B8F50", "D": "#2E8B8B", "E": "#9B5B7D"}
	fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
	x = np.arange(2)
	width = 0.15
	for axis, (metric, title, direction) in zip(axes.flat, METRICS):
		for index, experiment_id in enumerate(("A", "B", "C", "D", "E")):
			axis.bar(
				x + (index - 2) * width,
				[values[(board, experiment_id)][metric] for board in (7, 8)],
				width,
				label=experiment_id,
				color=colors[experiment_id],
			)
		axis.set_xticks(x, ("PCB7", "PCB8"))
		axis.set_title(f"{title} ({'↓' if direction == 'lower' else '↑'})")
		axis.grid(axis="y", alpha=0.22)
	axes[0, 0].legend(ncol=5, loc="best", frameon=False)
	fig.suptitle("Cross-board formal ablation results", fontsize=15)
	fig.savefig(SUMMARY_DIR / "cross_board_ablation_metrics.png", dpi=220)
	plt.close(fig)


def plot_tradeoff(summary_rows):
	values = formal_values(summary_rows)
	colors = {7: "#3B82B4", 8: "#D97732"}
	fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
	for board_id in (7, 8):
		for experiment_id in ("A", "B", "C", "D", "E"):
			selected = [
				row for row in summary_rows
				if int(row["board_id"]) == board_id and row["method"].startswith(experiment_id + "_")
			]
			timing = float(np.mean([float(row["forward_ms_mean"]) for row in selected]))
			psnr = values[(board_id, experiment_id)]["psnr"]
			axis.scatter(timing, psnr, s=72, color=colors[board_id])
			axis.annotate(experiment_id, (timing, psnr), xytext=(5, 4), textcoords="offset points", fontsize=10)
	axis.set_xlabel("Measured forward time per volume (ms, AMP)")
	axis.set_ylabel("PSNR (dB)")
	axis.set_title("Accuracy–latency trade-off on unseen boards")
	axis.grid(alpha=0.25)
	axis.scatter([], [], color=colors[7], label="PCB7")
	axis.scatter([], [], color=colors[8], label="PCB8")
	axis.legend(frameon=False)
	fig.savefig(SUMMARY_DIR / "cross_board_accuracy_latency.png", dpi=220)
	plt.close(fig)


def audit_files():
	checks = []
	for board_id in (7, 8):
		for path in sorted((RESULT_ROOT / f"pcb{board_id}").glob("*_per_sample.csv")):
			with path.open(encoding="utf-8-sig") as stream:
				rows = list(csv.DictReader(stream))
			finite = all(
				math.isfinite(float(row[key]))
				for row in rows
				for key, _, _ in METRICS
			)
			checks.append({"file": str(path), "rows": len(rows), "finite": finite})
	payload = {
		"checked_files": len(checks),
		"all_valid": all(item["rows"] == 200 and item["finite"] for item in checks),
		"checks": checks,
	}
	(SUMMARY_DIR / "result_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
	if not payload["all_valid"]:
		raise RuntimeError("Result audit failed")


def main():
	SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
	audit_files()
	contrasts = build_contrasts()
	write_contrasts(contrasts)
	summary_rows = load_summary_rows()
	write_paper_table(summary_rows)
	plot_metrics(summary_rows)
	plot_tradeoff(summary_rows)
	print(SUMMARY_DIR)


if __name__ == "__main__":
	main()
