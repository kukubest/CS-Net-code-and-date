# CS-Net: 3D PCB CT reconstruction

This public package contains the training/evaluation code and lightweight
experimental evidence for the CS-Net ablation study. It intentionally contains
**no 3D volumes, reconstructed prediction volumes, or model checkpoints**.

## Included

- Unified resumable five-group ablation runner (`run_ablation.py`).
- CS-Net and baseline model definitions, data loading, metrics, visualization,
  dataset-generation, evaluation, and analysis scripts.
- Per-epoch/per-sample numeric results, aggregate tables, selected figures, and
  the final experiment report.
- SHA-256 file inventory and a machine-readable exclusion audit.

## Not included

- Training/validation/test RAW volumes and generated prediction volumes.
- `.pt`, `.pth`, `.ckpt`, NumPy arrays, NIfTI/MHA/HDF5 volumes, or other weights.
- Private/local datasets. The independent test-set metadata are documented, but
  the 3D data themselves must be obtained separately.

## Environment used for the recorded experiments

- Python 3.11.15
- PyTorch 2.5.1 + CUDA 12.1
- cuDNN 9.1 (reported as 90100)
- GPU: NVIDIA GeForce RTX 3090 Ti (24 GiB)

See `docs/REPRODUCIBILITY.md` and `docs/DATA_AVAILABILITY.md` before running.

## Quick start

1. Create a Python 3.11 environment and install `requirements.txt`.
2. Place paired float32 RAW files under your own data directory.
3. Edit the six paths in `ablation_config.json`.
4. Validate the framework:

```bash
python run_ablation.py --mode smoke-test --device cuda
```

5. Run all five groups with automatic resume:

```bash
python run_ablation.py --experiments all --seeds 20260916 --device cuda
```

Additional examples:

```bash
python run_ablation.py --experiments csnet_l1 --seeds 20260916 --device cuda
python run_ablation.py --experiments csnet_compressed_no_hr_skip csnet_l1 --seeds 20260917 20260918 --device cuda
python run_ablation.py --experiments all --mode eval-only --device cuda
python run_ablation.py --mode summary-only
```

Running the same command resumes recoverable experiments and skips completed
runs. Use `--no-resume` only when you deliberately want to disable restoration.

## Repository layout

- Root Python files: runnable training and evaluation code.
- `results/`: lightweight results only, never volumetric data.
- `docs/`: data policy, path setup, reproducibility, and upload checklist.
- `manifests/`: complete file list, checksums, and exclusion audit.

## License and citation

No license was chosen automatically. Before publishing, select a license and
add author/paper metadata as described in `LICENSE_SELECTION_REQUIRED.md`.
