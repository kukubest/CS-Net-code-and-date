"""Reproducible full-volume reconstruction metrics."""

import math

import numpy as np
from skimage.metrics import structural_similarity


def gradient_mae_3d(prediction, target):
	"""Mean absolute error of forward differences over Z, Y and X.

	Each axis is averaged independently and the three axis means are averaged,
	so the much shorter Z axis is not underweighted by its voxel count.
	"""
	axis_errors = []
	for axis in range(3):
		pred_grad = np.diff(prediction, axis=axis)
		target_grad = np.diff(target, axis=axis)
		axis_errors.append(float(np.mean(np.abs(pred_grad - target_grad))))
	return float(np.mean(axis_errors))


def copper_overlap_metrics(prediction, target, prediction_threshold=0.91):
	"""Copper overlap using the generator's exact GT material label (1.00)."""
	gt_mask = np.isclose(target, 1.0, rtol=0.0, atol=1e-6)
	pred_mask = prediction >= float(prediction_threshold)
	intersection = int(np.logical_and(gt_mask, pred_mask).sum())
	pred_count = int(pred_mask.sum())
	gt_count = int(gt_mask.sum())
	union = pred_count + gt_count - intersection
	dice = 1.0 if pred_count + gt_count == 0 else 2.0 * intersection / (pred_count + gt_count)
	iou = 1.0 if union == 0 else intersection / union
	return float(dice), float(iou)


def volume_metrics(prediction, target, copper_config=None):
	prediction = np.asarray(prediction, dtype=np.float32)
	target = np.asarray(target, dtype=np.float32)
	mse = float(np.mean((prediction - target) ** 2))
	rmse = math.sqrt(mse)
	psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
	metrics = {
		"rmse": rmse,
		"psnr": psnr,
		"ssim": float(structural_similarity(target, prediction, data_range=1.0)),
		"gradient_mae": gradient_mae_3d(prediction, target),
	}
	if copper_config and copper_config.get("enabled", False):
		dice, iou = copper_overlap_metrics(
			prediction, target, copper_config["prediction_threshold"])
		metrics["copper_dice"] = dice
		metrics["copper_iou"] = iou
	return metrics


def summarize_metric_rows(rows):
	if not rows:
		return {}
	keys = sorted(k for k in rows[0] if k != "filename")
	result = {}
	for key in keys:
		values = np.asarray([row[key] for row in rows], dtype=np.float64)
		result[key + "_mean"] = float(np.mean(values))
		result[key + "_std"] = float(np.std(values, ddof=0))
	return result
