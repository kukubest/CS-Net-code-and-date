from __future__ import annotations

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
ABLATION_ROOT = ROOT / "outputs" / "ablation"
PCB78_ROOT = ROOT / "outputs" / "pcb7_pcb8_evaluation_20261019"
SUMMARY_DIR = ABLATION_ROOT / "summary" / "c_vs_d_multiseed"
SEEDS = (20260916, 20260917, 20260918)
METRICS = (
    ("rmse", "RMSE", "lower"),
    ("psnr", "PSNR (dB)", "higher"),
    ("ssim", "SSIM", "higher"),
    ("gradient_mae", "Gradient MAE", "lower"),
    ("copper_dice", "Copper Dice", "higher"),
    ("copper_iou", "Copper IoU", "higher"),
)
EXPERIMENTS = {
    "C": "csnet_compressed_no_hr_skip",
    "D": "csnet_l1",
}


def safe(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def sample_path(board: int, experiment_id: str, seed: int) -> Path:
    experiment = EXPERIMENTS[experiment_id]
    if board == 6:
        return ABLATION_ROOT / experiment / f"seed_{seed}" / "test_per_sample.csv"
    method = f"{experiment_id}_{experiment}_seed_{seed}"
    return PCB78_ROOT / f"pcb{board}" / f"{safe(method)}_per_sample.csv"


def read_samples(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 200:
        raise RuntimeError(f"{path}: expected 200 rows, found {len(rows)}")
    result = {}
    for row in rows:
        values = {metric: float(row[metric]) for metric, _, _ in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"{path}: non-finite metric for {row['filename']}")
        result[row["filename"]] = values
    return result


def load_board(board: int):
    loaded = {
        experiment_id: {
            seed: read_samples(sample_path(board, experiment_id, seed))
            for seed in SEEDS
        }
        for experiment_id in EXPERIMENTS
    }
    reference = set(loaded["C"][SEEDS[0]])
    for experiment_id in EXPERIMENTS:
        for seed in SEEDS:
            if set(loaded[experiment_id][seed]) != reference:
                raise RuntimeError(f"PCB{board} filename mismatch: {experiment_id}, seed {seed}")
    return loaded, sorted(reference)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 5000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyse():
    contrast_rows = []
    seed_rows = []
    board_payload = {}
    for board in (6, 7, 8):
        loaded, filenames = load_board(board)
        means = {}
        for experiment_id in EXPERIMENTS:
            means[experiment_id] = {
                filename: {
                    metric: float(np.mean([
                        loaded[experiment_id][seed][filename][metric] for seed in SEEDS
                    ]))
                    for metric, _, _ in METRICS
                }
                for filename in filenames
            }
            for seed in SEEDS:
                row = {"board_id": board, "experiment_id": experiment_id, "seed": seed}
                for metric, _, _ in METRICS:
                    row[metric] = float(np.mean([
                        loaded[experiment_id][seed][filename][metric] for filename in filenames
                    ]))
                seed_rows.append(row)

        board_payload[str(board)] = {}
        for metric_index, (metric, label, direction) in enumerate(METRICS):
            c_values = np.asarray([means["C"][name][metric] for name in filenames], dtype=np.float64)
            d_values = np.asarray([means["D"][name][metric] for name in filenames], dtype=np.float64)
            improvement = d_values - c_values if direction == "higher" else c_values - d_values
            try:
                p_value = float(wilcoxon(d_values, c_values, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            ci_low, ci_high = bootstrap_ci(
                improvement,
                seed=20260916 + board * 100 + metric_index,
            )
            c_mean = float(c_values.mean())
            d_mean = float(d_values.mean())
            relative = float(improvement.mean() / abs(c_mean) * 100.0) if c_mean else math.nan
            row = {
                "board_id": board,
                "comparison": "C vs D",
                "scope": "three-seed per-sample prediction metric means",
                "metric": metric,
                "direction": direction,
                "C_mean": c_mean,
                "D_mean": d_mean,
                "D_improvement_mean": float(improvement.mean()),
                "D_relative_improvement_percent": relative,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wilcoxon_p": p_value,
                "n_pairs": len(filenames),
            }
            contrast_rows.append(row)
            board_payload[str(board)][metric] = row

    adjusted = holm_adjust([row["wilcoxon_p"] for row in contrast_rows])
    for row, value in zip(contrast_rows, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["significant_0.05"] = bool(value < 0.05)
    return contrast_rows, seed_rows, board_payload


def write_outputs(contrast_rows, seed_rows, board_payload):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / "c_vs_d_paired_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    with (SUMMARY_DIR / "c_vs_d_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary = {"seeds": list(SEEDS), "boards": board_payload, "metrics": [m[0] for m in METRICS]}
    (SUMMARY_DIR / "c_vs_d_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# C vs D multi-seed comparison",
        "",
        "C removes the high-resolution structural skip; D is the complete CS-Net. Both use L1.",
        "Each board contains 200 paired samples. Predictions are averaged per sample over seeds 20260916/17/18 before paired testing.",
        "",
        "| PCB | Metric | C mean | D mean | D-oriented improvement | Relative improvement | 95% CI | Holm p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contrast_rows:
        lines.append(
            "| {board_id} | {metric} | {C_mean:.8f} | {D_mean:.8f} | {D_improvement_mean:.8f} | "
            "{D_relative_improvement_percent:.2f}% | [{bootstrap_95ci_low:.8f}, {bootstrap_95ci_high:.8f}] | "
            "{holm_adjusted_p:.3g} |".format(**row)
        )
    lines.extend(["", "## Structural conclusion", ""])
    for board in (6, 7, 8):
        grad = board_payload[str(board)]["gradient_mae"]
        ssim = board_payload[str(board)]["ssim"]
        dice = board_payload[str(board)]["copper_dice"]
        lines.append(
            f"- PCB{board}: HR skip changes Gradient MAE from {grad['C_mean']:.6f} to {grad['D_mean']:.6f} "
            f"({grad['D_relative_improvement_percent']:.2f}% D-oriented improvement), SSIM from "
            f"{ssim['C_mean']:.6f} to {ssim['D_mean']:.6f}, and Dice from "
            f"{dice['C_mean']:.6f} to {dice['D_mean']:.6f}."
        )
    lines.extend([
        "",
        "The primary claim should be based on Gradient MAE and local structural metrics. A small PSNR trade-off does not negate the skip's structural contribution.",
    ])
    (SUMMARY_DIR / "c_vs_d_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_seed_summary(seed_rows)


def plot_seed_summary(seed_rows):
    selected_metrics = (
        ("gradient_mae", "Gradient MAE", "lower"),
        ("ssim", "SSIM", "higher"),
        ("copper_dice", "Copper Dice", "higher"),
    )
    colors = {"C": "#7B8F50", "D": "#2E8B8B"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    for axis, (metric, label, direction) in zip(axes, selected_metrics):
        for index, experiment_id in enumerate(("C", "D")):
            means, stds = [], []
            for board in (6, 7, 8):
                values = [
                    row[metric] for row in seed_rows
                    if row["board_id"] == board and row["experiment_id"] == experiment_id
                ]
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=0)))
            axis.bar(
                x + (index - 0.5) * width,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=experiment_id,
                color=colors[experiment_id],
                alpha=0.92,
            )
        axis.set_xticks(x, ("PCB6", "PCB7", "PCB8"))
        axis.set_title(f"{label} ({'lower is better' if direction == 'lower' else 'higher is better'})")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("High-resolution skip contribution: C (no HR skip) vs D (complete CS-Net)")
    fig.savefig(SUMMARY_DIR / "c_vs_d_multiseed_structural_metrics.png", dpi=240)
    plt.close(fig)


def main():
    contrast_rows, seed_rows, board_payload = analyse()
    write_outputs(contrast_rows, seed_rows, board_payload)
    print(SUMMARY_DIR)


if __name__ == "__main__":
    main()
