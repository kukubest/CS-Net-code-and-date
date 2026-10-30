"""Fixed-sample, fixed-ROI qualitative comparison generation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_ORDER = [
	("GT", "gt"), ("SIRT", "sirt"), ("A", "swin_unetr_baseline"),
	("B", "csnet_no_compression"), ("C", "csnet_compressed_no_hr_skip"),
	("D", "csnet_l1"), ("E", "csnet_mse"),
]


def _best_window(score, height, width):
	best = (0, 0, float("-inf"))
	step_y, step_x = max(1, height // 4), max(1, width // 4)
	for y in range(0, max(1, score.shape[0] - height + 1), step_y):
		for x in range(0, max(1, score.shape[1] - width + 1), step_x):
			value = float(score[y:y + height, x:x + width].mean())
			if value > best[2]:
				best = (y, x, value)
	return [best[0], best[0] + height, best[1], best[1] + width]


def create_or_load_rois(gt, path, filename):
	path = Path(path)
	if path.is_file():
		return json.loads(path.read_text(encoding="utf-8"))
	trace_score = np.abs(np.diff(gt[30], axis=0, prepend=gt[30, :1])) + np.abs(
		np.diff(gt[30], axis=1, prepend=gt[30, :, :1]))
	via_score = (gt[20] == 0.0).astype(np.float32)
	trace = _best_window(trace_score, 80, 80)
	via = _best_window(via_score, 80, 80)
	# For XZ, choose the Y row with the most material transitions.
	xz_score = np.abs(np.diff(gt, axis=2)).sum(axis=(0, 2))
	y_index = int(np.argmax(xz_score))
	rois = {
		"filename": filename,
		"method": "deterministic GT-label/gradient selection; saved once and reused",
		"dense_trace_xy": {"z": 30, "yx": trace},
		"via_xy": {"z": 20, "yx": via},
		"layer_boundary_xz": {"y": y_index, "z": [0, 40], "x": [0, 256]},
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(rois, indent=2, ensure_ascii=False), encoding="utf-8")
	return rois


def generate_comparison(gt_path, dirty_path, prediction_paths, roi_path, output_path):
	shape = (40, 256, 256)
	gt = np.fromfile(str(gt_path), dtype=np.float32).reshape(shape)
	sirt = np.fromfile(str(dirty_path), dtype=np.float32).reshape(shape)
	rois = create_or_load_rois(gt, roi_path, Path(gt_path).name)
	volumes = {"gt": gt, "sirt": sirt}
	for name, path in prediction_paths.items():
		if Path(path).is_file():
			volumes[name] = np.fromfile(str(path), dtype=np.float32).reshape(shape)
	if any(key not in volumes for _, key in DISPLAY_ORDER):
		return False
	fig, axes = plt.subplots(3, len(DISPLAY_ORDER), figsize=(17.5, 7.5))
	for col, (label, key) in enumerate(DISPLAY_ORDER):
		volume = volumes[key]
		t = rois["dense_trace_xy"]; y0, y1, x0, x1 = t["yx"]
		axes[0, col].imshow(volume[t["z"], y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
		v = rois["via_xy"]; y0, y1, x0, x1 = v["yx"]
		axes[1, col].imshow(volume[v["z"], y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
		b = rois["layer_boundary_xz"]
		axes[2, col].imshow(volume[b["z"][0]:b["z"][1], b["y"], b["x"][0]:b["x"][1]],
			cmap="gray", vmin=0, vmax=1, aspect="auto")
		axes[0, col].set_title(label)
		for row in range(3):
			axes[row, col].axis("off")
	for row, label in enumerate(("Dense trace (XY)", "Via (XY)", "Layer boundary (XZ)")):
		axes[row, 0].set_ylabel(label)
	fig.tight_layout()
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
	plt.close(fig)
	return True
