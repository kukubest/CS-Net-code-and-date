"""Build a PCB-level independent test set from one previously unseen board.

The default configuration uses PCB/6 and writes clean/dirty pairs to an
``independent_test`` split.  Crop positions are deterministic and are saved in
a CSV manifest so that every model is evaluated on exactly the same volumes.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from build_full_dataset import build_3d_patch_4layers, process_and_resize_patch


LAYER_PREFIXES = ("t", "b", "d", "s")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_layer(board_dir, prefix, board_id):
	"""Find a layer image without assuming that it is stored as JPEG."""
	for suffix in IMAGE_SUFFIXES:
		candidate = board_dir / f"{prefix}{board_id}{suffix}"
		if candidate.is_file():
			return candidate
	raise FileNotFoundError(
		f"Missing layer {prefix}{board_id} in {board_dir}; "
		f"supported extensions: {', '.join(IMAGE_SUFFIXES)}"
	)


def load_gray(path):
	with Image.open(path) as image:
		return np.asarray(image.convert("L"))


def generate_clean_testset(
	pcb_root,
	clean_dir,
	board_id,
	num_patches,
	target_size,
	seed,
	overwrite,
):
	board_dir = pcb_root / str(board_id)
	layer_paths = {
		prefix: find_layer(board_dir, prefix, board_id) for prefix in LAYER_PREFIXES
	}
	layers = {prefix: load_gray(path) for prefix, path in layer_paths.items()}

	shapes = {array.shape for array in layers.values()}
	if len(shapes) != 1:
		raise ValueError(f"Layer sizes are not aligned: {shapes}")
	canvas_h, canvas_w = next(iter(shapes))
	valid_scales = [
		scale for scale in (1, 2, 4)
		if scale * target_size <= min(canvas_h, canvas_w)
	]
	if not valid_scales:
		raise ValueError(
			f"Board {board_id} ({canvas_w}x{canvas_h}) is smaller than one "
			f"{target_size}x{target_size} crop"
		)

	clean_dir.mkdir(parents=True, exist_ok=True)
	existing = list(clean_dir.glob("*.raw"))
	if existing and not overwrite:
		raise FileExistsError(
			f"{clean_dir} already contains {len(existing)} RAW files. "
			"Use --overwrite only when intentionally rebuilding this split."
		)
	if overwrite:
		for path in existing:
			path.unlink()

	rng = random.Random(seed)
	records = []
	attempts = 0
	max_attempts = num_patches * 20

	while len(records) < num_patches and attempts < max_attempts:
		attempts += 1
		scale = rng.choice(valid_scales)
		crop_size = target_size * scale
		y = rng.randint(0, canvas_h - crop_size)
		x = rng.randint(0, canvas_w - crop_size)

		masks = {}
		for prefix, layer in layers.items():
			crop = layer[y : y + crop_size, x : x + crop_size]
			masks[prefix] = process_and_resize_patch(crop, target_size)

		top_ratio = float(np.mean(masks["t"]))
		bottom_ratio = float(np.mean(masks["b"]))
		top_var = float(np.var(masks["t"]))
		bottom_var = float(np.var(masks["b"]))
		if top_ratio < 0.05 and bottom_ratio < 0.05:
			continue
		if top_ratio > 0.95 and bottom_ratio > 0.95:
			continue
		if top_var < 0.05 and bottom_var < 0.05:
			continue

		index = len(records)
		filename = f"board{board_id}_patch{index:04d}_scale{scale}x.raw"
		volume = build_3d_patch_4layers(
			masks["t"], masks["b"], masks["d"], masks["s"],
			shape=(40, target_size, target_size),
		)
		volume.astype(np.float32).tofile(clean_dir / filename)
		records.append(
			{
				"filename": filename,
				"board_id": board_id,
				"scale": scale,
				"crop_x": x,
				"crop_y": y,
				"crop_size": crop_size,
				"top_ratio": f"{top_ratio:.8f}",
				"bottom_ratio": f"{bottom_ratio:.8f}",
			}
		)
		if index == 0 or len(records) % 25 == 0:
			print(f"  [{len(records):3d}/{num_patches}] {filename}")

	if len(records) != num_patches:
		raise RuntimeError(
			f"Only generated {len(records)}/{num_patches} valid patches after "
			f"{attempts} attempts"
		)

	manifest_path = clean_dir / "manifest.csv"
	with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(records[0]))
		writer.writeheader()
		writer.writerows(records)

	metadata_path = clean_dir / "dataset_info.txt"
	metadata_path.write_text(
		"split=independent_test\n"
		f"board_id={board_id}\nseed={seed}\n"
		f"volume_shape=40,{target_size},{target_size}\n"
		f"patches={len(records)}\n"
		f"source_dir={board_dir.resolve()}\n"
		+ "\n".join(f"source_{key}={value.resolve()}" for key, value in layer_paths.items())
		+ "\n",
		encoding="utf-8",
	)
	return manifest_path


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--pcb-root", type=Path, default=Path(r"G:\CT_RE\PCB"))
	parser.add_argument("--board-id", type=int, default=6)
	parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\CT_RE\dataset"))
	parser.add_argument("--split", default="independent_test")
	parser.add_argument("--num-patches", type=int, default=200)
	parser.add_argument("--target-size", type=int, default=256)
	parser.add_argument("--seed", type=int, default=20260916)
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument(
		"--dirty-only",
		action="store_true",
		help="Keep the existing clean split and only run the ASTRA simulation.",
	)
	parser.add_argument(
		"--simulate-dirty",
		action="store_true",
		help="Run the existing ASTRA CL simulation after clean volumes are built.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	clean_dir = args.dataset_root / "Final_Adaptive_GT" / args.split
	dirty_dir = args.dataset_root / "Final_Adaptive_Dirty" / args.split

	print(f"Building PCB {args.board_id} independent test set")
	if args.dirty_only:
		if not clean_dir.is_dir() or not list(clean_dir.glob("*.raw")):
			raise FileNotFoundError(f"No existing clean split found at {clean_dir}")
		print(f"Keeping existing clean split: {clean_dir}")
	else:
		print(f"Clean output: {clean_dir}")
		manifest = generate_clean_testset(
			args.pcb_root,
			clean_dir,
			args.board_id,
			args.num_patches,
			args.target_size,
			args.seed,
			args.overwrite,
		)
		print(f"Clean test set complete; manifest: {manifest}")

	if args.simulate_dirty:
		from simulate_cl_dirty_data import batch_generate_dirty_dataset

		existing_dirty = list(dirty_dir.glob("*.raw")) if dirty_dir.is_dir() else []
		if existing_dirty and not args.overwrite:
			raise FileExistsError(
				f"{dirty_dir} already contains {len(existing_dirty)} RAW files. "
				"Use --overwrite only when intentionally rebuilding them."
			)
		if args.overwrite:
			for path in existing_dirty:
				path.unlink()
		# The simulator injects Poisson noise through NumPy's global RNG.
		# Reset it here so the paired test input is exactly reproducible.
		np.random.seed(args.seed)
		print(f"Dirty output: {dirty_dir}")
		batch_generate_dirty_dataset(
			str(clean_dir), str(dirty_dir), shape=(40, args.target_size, args.target_size)
		)
		for sidecar in ("manifest.csv", "dataset_info.txt"):
			source = clean_dir / sidecar
			if source.is_file():
				shutil.copy2(str(source), str(dirty_dir / sidecar))
	else:
		print("Dirty simulation skipped. Re-run with --simulate-dirty on an ASTRA CUDA host.")


if __name__ == "__main__":
	main()
