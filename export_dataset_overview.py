"""Export a read-only audit and visual overview of every paired RAW sample.

This script never modifies source data. It creates a new output directory and
fails if that directory already exists.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHAPE = (40, 256, 256)
EXPECTED_BYTES = int(np.prod(SHAPE)) * 4
NAME_RE = re.compile(r"board(?P<board>\d+)_patch(?P<patch>\d+)_scale(?P<scale>\d+)x")


def parse_name(name):
	match = NAME_RE.search(name)
	if not match:
		return None, None, None
	return int(match.group("board")), int(match.group("patch")), int(match.group("scale"))


def array_stats(array, prefix):
	return {
		prefix + "_min": float(array.min()),
		prefix + "_max": float(array.max()),
		prefix + "_mean": float(array.mean()),
		prefix + "_std": float(array.std()),
		prefix + "_p01": float(np.percentile(array, 1)),
		prefix + "_p50": float(np.percentile(array, 50)),
		prefix + "_p99": float(np.percentile(array, 99)),
		prefix + "_finite": bool(np.isfinite(array).all()),
	}


def scan_split(split, role, dirty_dir, clean_dir):
	dirty_dir, clean_dir = Path(dirty_dir), Path(clean_dir)
	rows, errors = [], []
	for dirty_path in sorted(dirty_dir.glob("*_dirty.raw")):
		clean_name = dirty_path.name.replace("_dirty.raw", ".raw")
		clean_path = clean_dir / clean_name
		board, patch, scale = parse_name(clean_name)
		row = {
			"split": split, "role": role, "board_id": board,
			"patch_id": patch, "scale": scale, "dirty_file": dirty_path.name,
			"gt_file": clean_name, "dirty_path": str(dirty_path.resolve()),
			"gt_path": str(clean_path.resolve()),
			"dirty_bytes": dirty_path.stat().st_size,
			"gt_bytes": clean_path.stat().st_size if clean_path.is_file() else None,
			"shape_z": SHAPE[0], "shape_y": SHAPE[1], "shape_x": SHAPE[2],
			"dtype": "float32",
		}
		if not clean_path.is_file():
			errors.append("Missing GT: " + str(clean_path))
			row["pair_valid"] = False
			rows.append(row)
			continue
		if dirty_path.stat().st_size != EXPECTED_BYTES or clean_path.stat().st_size != EXPECTED_BYTES:
			errors.append("Invalid byte size: {} / {}".format(dirty_path.name, clean_path.name))
			row["pair_valid"] = False
			rows.append(row)
			continue
		dirty = np.fromfile(str(dirty_path), dtype=np.float32).reshape(SHAPE)
		gt = np.fromfile(str(clean_path), dtype=np.float32).reshape(SHAPE)
		row.update(array_stats(dirty, "dirty"))
		row.update(array_stats(gt, "gt"))
		row.update({
			"gt_air_fraction": float(np.mean(np.isclose(gt, 0.0, atol=1e-6))),
			"gt_substrate_fraction": float(np.mean(np.isclose(gt, 0.21, atol=1e-6))),
			"gt_solder_fraction": float(np.mean(np.isclose(gt, 0.82, atol=1e-6))),
			"gt_copper_fraction": float(np.mean(np.isclose(gt, 1.0, atol=1e-6))),
			"input_gt_mae": float(np.mean(np.abs(dirty - gt))),
			"input_gt_rmse": float(np.sqrt(np.mean((dirty - gt) ** 2))),
			"pair_valid": bool(np.isfinite(dirty).all() and np.isfinite(gt).all()),
		})
		rows.append(row)
	return rows, errors


def write_csv(path, rows):
	with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader(); writer.writerows(rows)


def aggregate(rows):
	result = []
	for split in ("train", "validation", "independent_test"):
		items = [row for row in rows if row["split"] == split]
		result.append({
			"split": split, "role": items[0]["role"], "paired_samples": len(items),
			"boards": ", ".join(map(str, sorted({r["board_id"] for r in items}))),
			"scale_1x": sum(r["scale"] == 1 for r in items),
			"scale_2x": sum(r["scale"] == 2 for r in items),
			"scale_4x": sum(r["scale"] == 4 for r in items),
			"dirty_mean": float(np.mean([r["dirty_mean"] for r in items])),
			"dirty_std_mean": float(np.mean([r["dirty_std"] for r in items])),
			"gt_mean": float(np.mean([r["gt_mean"] for r in items])),
			"gt_std_mean": float(np.mean([r["gt_std"] for r in items])),
			"input_gt_mae_mean": float(np.mean([r["input_gt_mae"] for r in items])),
			"input_gt_rmse_mean": float(np.mean([r["input_gt_rmse"] for r in items])),
			"copper_fraction_mean": float(np.mean([r["gt_copper_fraction"] for r in items])),
			"all_pairs_valid": all(r["pair_valid"] for r in items),
		})
	return result


def style_axes(axis):
	axis.spines[["top", "right"]].set_visible(False)
	axis.grid(axis="y", color="#D9E2F3", linewidth=0.7, alpha=0.65)


def make_charts(rows, summaries, output):
	colors = {"train": "#4472C4", "validation": "#70AD47", "independent_test": "#ED7D31"}
	labels = {"train": "Train", "validation": "Validation", "independent_test": "Independent test"}

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
	splits = [s["split"] for s in summaries]
	axes[0].bar([labels[s] for s in splits], [s["paired_samples"] for s in summaries], color=[colors[s] for s in splits])
	axes[0].set_title("Paired volumes by split"); axes[0].set_ylabel("Sample pairs")
	for i, s in enumerate(summaries): axes[0].text(i, s["paired_samples"] + 30, str(s["paired_samples"]), ha="center")
	style_axes(axes[0])
	bottom = np.zeros(3)
	for scale, shade in ((1, "#5B9BD5"), (2, "#A5A5A5"), (4, "#FFC000")):
		values = np.asarray([s["scale_{}x".format(scale)] for s in summaries])
		axes[1].bar([labels[s] for s in splits], values, bottom=bottom, label="{}x".format(scale), color=shade)
		bottom += values
	axes[1].set_title("Multi-scale composition"); axes[1].set_ylabel("Sample pairs"); axes[1].legend(frameon=False)
	style_axes(axes[1]); fig.tight_layout(); fig.savefig(output / "dataset_composition.png", dpi=220); plt.close(fig)

	fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
	for split in splits:
		items = [r for r in rows if r["split"] == split]
		axes[0].hist([r["dirty_mean"] for r in items], bins=30, alpha=0.55, label=labels[split], color=colors[split])
		axes[1].hist([r["input_gt_rmse"] for r in items], bins=30, alpha=0.55, label=labels[split], color=colors[split])
		axes[2].hist([100 * r["gt_copper_fraction"] for r in items], bins=30, alpha=0.55, label=labels[split], color=colors[split])
	for ax, title, xlabel in zip(axes, ("SIRT mean intensity", "SIRT-to-GT RMSE", "Copper content"), ("Mean intensity", "RMSE", "Copper voxels (%)")):
		ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Volumes"); style_axes(ax)
	axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(output / "dataset_distributions.png", dpi=220); plt.close(fig)

	fig, ax = plt.subplots(figsize=(9, 4.5))
	counts = Counter((r["split"], r["board_id"]) for r in rows)
	categories = sorted({r["board_id"] for r in rows})
	x = np.arange(len(categories)); width = 0.25
	for index, split in enumerate(splits):
		ax.bar(x + (index - 1) * width, [counts[(split, b)] for b in categories], width, label=labels[split], color=colors[split])
	ax.set_xticks(x, ["PCB {}".format(b) for b in categories]); ax.set_ylabel("Sample pairs"); ax.set_title("PCB-level data separation"); ax.legend(frameon=False); style_axes(ax)
	fig.tight_layout(); fig.savefig(output / "pcb_split_separation.png", dpi=220); plt.close(fig)


def make_montage(rows, output):
	chosen = []
	for split in ("train", "validation", "independent_test"):
		items = [r for r in rows if r["split"] == split]
		indices = np.linspace(0, len(items) - 1, 4, dtype=int)
		chosen.extend([items[i] for i in indices])
	fig, axes = plt.subplots(len(chosen), 4, figsize=(10, 2.1 * len(chosen)))
	for row_index, item in enumerate(chosen):
		dirty = np.fromfile(item["dirty_path"], dtype=np.float32).reshape(SHAPE)
		gt = np.fromfile(item["gt_path"], dtype=np.float32).reshape(SHAPE)
		views = ((dirty[30], "SIRT XY z=30"), (gt[30], "GT XY z=30"), (dirty[:, 128, :], "SIRT XZ y=128"), (gt[:, 128, :], "GT XZ y=128"))
		for col, (image, title) in enumerate(views):
			axes[row_index, col].imshow(image, cmap="gray", vmin=0, vmax=1, aspect="auto" if col > 1 else "equal")
			axes[row_index, col].axis("off")
			if row_index == 0: axes[row_index, col].set_title(title)
		axes[row_index, 0].text(-0.08, 0.5, "{}\nPCB {} {}x".format(item["split"], item["board_id"], item["scale"]), transform=axes[row_index, 0].transAxes, ha="right", va="center", fontsize=8)
	fig.suptitle("Representative paired volumes", y=0.995, fontsize=14)
	fig.tight_layout(); fig.savefig(output / "representative_volumes.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", default=r"G:\CT_RE\CT_DRA\outputs\dataset_export_20261009")
	args = parser.parse_args(); output = Path(args.output)
	if output.exists():
		raise FileExistsError("Refusing to overwrite existing output: {}".format(output))
	output.mkdir(parents=True)
	specs = [
		("train", "training/optimization", r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train", r"G:\CT_RE\dataset\Final_Adaptive_GT\train"),
		("validation", "checkpoint selection only", r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test", r"G:\CT_RE\dataset\Final_Adaptive_GT\test"),
		("independent_test", "final evaluation only", r"G:\CT_RE\dataset\Final_Adaptive_Dirty\independent_test", r"G:\CT_RE\dataset\Final_Adaptive_GT\independent_test"),
	]
	rows, errors = [], []
	for spec in specs:
		new_rows, new_errors = scan_split(*spec); rows.extend(new_rows); errors.extend(new_errors)
	if errors:
		(output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
		raise RuntimeError("Dataset scan failed; see errors.json")
	summaries = aggregate(rows)
	write_csv(output / "all_samples.csv", rows); write_csv(output / "split_summary.csv", summaries)
	make_charts(rows, summaries, output); make_montage(rows, output)
	metadata = {"created_by": "export_dataset_overview.py", "source_data_modified": False, "shape": SHAPE, "dtype": "float32", "expected_bytes": EXPECTED_BYTES, "sample_pairs": len(rows), "splits": summaries}
	(output / "export_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
	print(json.dumps({"output": str(output), "sample_pairs": len(rows), "summary": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
