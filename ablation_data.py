"""Strict paired RAW loading and independent-test integrity checks."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def clean_name_from_dirty(name):
	if not name.endswith("_dirty.raw"):
		raise ValueError("Dirty RAW name must end with _dirty.raw: {}".format(name))
	return name[:-10] + ".raw"


class PairedRawDataset(Dataset):
	def __init__(self, dirty_dir, clean_dir, shape=(40, 256, 256), return_name=False):
		self.dirty_dir = Path(dirty_dir)
		self.clean_dir = Path(clean_dir)
		self.shape = tuple(shape)
		self.return_name = return_name
		self.pairs = []
		for dirty in sorted(self.dirty_dir.glob("*_dirty.raw")):
			clean = self.clean_dir / clean_name_from_dirty(dirty.name)
			if not clean.is_file():
				raise FileNotFoundError("Missing GT for {}: {}".format(dirty, clean))
			self.pairs.append((dirty, clean))
		if not self.pairs:
			raise ValueError("No paired RAW samples found in {}".format(self.dirty_dir))

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		dirty_path, clean_path = self.pairs[index]
		dirty = np.fromfile(str(dirty_path), dtype=np.float32).reshape(self.shape)
		clean = np.fromfile(str(clean_path), dtype=np.float32).reshape(self.shape)
		items = (torch.from_numpy(dirty).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0))
		if self.return_name:
			return items + (dirty_path.name,)
		return items


def validate_independent_test(dirty_dir, clean_dir, data_config, report_path=None):
	"""Read every test RAW and reject any invalid final-evaluation input."""
	dirty_dir, clean_dir = Path(dirty_dir), Path(clean_dir)
	shape = tuple(data_config["volume_shape"])
	expected_count = int(data_config["expected_test_samples"])
	expected_bytes = int(data_config["expected_raw_bytes"])
	dirty_files = sorted(dirty_dir.glob("*_dirty.raw"))
	clean_files = sorted(clean_dir.glob("*.raw"))
	errors = []
	warnings = []
	if len(dirty_files) != expected_count:
		errors.append("dirty count {} != {}".format(len(dirty_files), expected_count))
	if len(clean_files) != expected_count:
		errors.append("clean count {} != {}".format(len(clean_files), expected_count))
	clean_by_name = {p.name: p for p in clean_files}
	global_min, global_max = float("inf"), float("-inf")
	checked = 0
	for dirty in dirty_files:
		clean_name = clean_name_from_dirty(dirty.name)
		clean = clean_by_name.get(clean_name)
		if clean is None:
			errors.append("missing clean pair for {}".format(dirty.name))
			continue
		for role, path in (("dirty", dirty), ("clean", clean)):
			if path.stat().st_size != expected_bytes:
				errors.append("{} {} has {} bytes, expected {}".format(
					role, path.name, path.stat().st_size, expected_bytes))
				continue
			array = np.fromfile(str(path), dtype=np.float32)
			if array.size != int(np.prod(shape)):
				errors.append("{} {} has invalid shape".format(role, path.name))
				continue
			if not np.isfinite(array).all():
				errors.append("{} {} contains NaN/Inf".format(role, path.name))
			global_min = min(global_min, float(array.min()))
			global_max = max(global_max, float(array.max()))
		checked += 1
	if global_min < -1e-6 or global_max > 1.0 + 1e-6:
		errors.append("value range [{:.8g}, {:.8g}] is outside [0, 1]".format(global_min, global_max))
	manifest = clean_dir / "manifest.csv"
	info = clean_dir / "dataset_info.txt"
	if not manifest.is_file():
		warnings.append("manifest.csv is missing")
	if not info.is_file():
		warnings.append("dataset_info.txt is missing")
	report = {
		"valid": not errors,
		"dirty_dir": str(dirty_dir.resolve()),
		"clean_dir": str(clean_dir.resolve()),
		"paired_samples_checked": checked,
		"shape": list(shape),
		"dtype": "float32",
		"value_min": None if global_min == float("inf") else global_min,
		"value_max": None if global_max == float("-inf") else global_max,
		"errors": errors,
		"warnings": warnings,
	}
	if report_path:
		Path(report_path).parent.mkdir(parents=True, exist_ok=True)
		Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
	if errors:
		raise RuntimeError("Independent test validation failed: " + "; ".join(errors[:10]))
	return report
