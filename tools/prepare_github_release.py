#!/usr/bin/env python3
"""Create a non-destructive, volume-free GitHub release package for CS-Net.

This script only copies selected files into a new directory. It never deletes,
moves, or edits any existing project file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".raw", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".nii", ".mha",
    ".mhd", ".h5", ".hdf5", ".onnx", ".safetensors", ".bin",
}
FORBIDDEN_NAMES = {"checkpoint", "checkpoints", "predictions", "volumes"}
RESULT_SUFFIXES = {".csv", ".json", ".md", ".png", ".pdf", ".txt", ".xlsx"}


CODE_FILES = [
    "ablation_data.py",
    "ablation_metrics.py",
    "ablation_models.py",
    "ablation_visualization.py",
    "run_ablation.py",
    "models.py",
    "train_3d_swin_unetr.py",
    "train_3d_unet_baseline.py",
    "train_unetr_baseline.py",
    "train_fast_robust_clnet.py",
    "train_ablation_noskip.py",
    "build_full_dataset.py",
    "build_independent_testset.py",
    "simulate_cl_dirty_data.py",
    "generate_fdk_baseline.py",
    "unified_evaluate_200.py",
    "evaluate_pcb7_pcb8.py",
    "evaluate_noise_robustness_200.py",
    "evaluate_3d_unet.py",
    "evaluate_unetr_baseline.py",
    "evaluate_fast_robust_clnet.py",
    "evaluate_fdk_baseline.py",
    "evaluate_robustness.py",
    "benchmark_complexity.py",
    "export_ablation_results.py",
    "analyze_pcb7_pcb8_results.py",
    "compare_c_d_multiseed.py",
    "make_pcb_dataset_figure.py",
    "plot_qualitative_comparison_paper.py",
    "export_dataset_overview.py",
    "run_c_multiseed_pipeline.ps1",
    "ABLATION_README.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path, records: list, category: str) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(destination))
    shutil.copy2(str(source), str(destination))
    records.append(
        {
            "category": category,
            "public_path": destination.as_posix(),
            "source_path": source.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def copy_tree_filtered(source: Path, destination: Path, records: list, category: str) -> None:
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in RESULT_SUFFIXES:
            continue
        if any(part.lower() in FORBIDDEN_NAMES for part in item.parts):
            continue
        copy_exact(item, destination / item.relative_to(source), records, category)


def write_new(path: Path, text: str, records: list, category: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(path))
    path.write_text(text, encoding="utf-8", newline="\n")
    records.append(
        {
            "category": category,
            "public_path": path.as_posix(),
            "source_path": "generated for public release",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def make_sanitized_config(project: Path) -> str:
    config = json.loads((project / "ablation_config.json").read_text(encoding="utf-8"))
    config["paths"] = {
        "dirty_train": "data/dirty/train",
        "clean_train": "data/clean/train",
        "dirty_val": "data/dirty/validation",
        "clean_val": "data/clean/validation",
        "dirty_independent_test": "data/dirty/independent_test",
        "clean_independent_test": "data/clean/independent_test",
        "output_root": "outputs/ablation",
    }
    config["evaluation"]["save_predictions"] = False
    config["public_release_notes"] = {
        "data_not_included": True,
        "volume_files_not_included": True,
        "weights_not_included": True,
        "path_policy": "Replace paths above with local dataset locations before running.",
    }
    return json.dumps(config, ensure_ascii=False, indent=2) + "\n"


def project_readme() -> str:
    return """# CS-Net: 3D PCB CT reconstruction

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
"""


def reproducibility_doc() -> str:
    return """# Reproducibility

## Experimental roles

- Existing training partition: training only.
- Historical `test` partition: validation/checkpoint selection only.
- PCB 6 independent set: final testing only; it is never used for training,
  validation, early stopping, hyperparameter selection, or epoch selection.
- PCB 7 and PCB 8: additional cross-board generalization evaluation.

## Volume convention

Each sample is float32 with shape `40 x 256 x 256`. The common strategy pads
the Z axis reflectively to 64 slices, performs the network operation, and crops
back to the real 40-slice region. Metrics are computed on the valid 40 slices.

The recorded PCB 6 independent test set contains 200 paired GT/SIRT samples,
generated with seed 20260916. Expected RAW size is 10,485,760 bytes per file.

## Five ablation groups

- A `swin_unetr_baseline`: original Swin-UNETR, L1.
- B `csnet_no_compression`: CS-Net modules retained, compression bypassed, L1.
- C `csnet_compressed_no_hr_skip`: compressed backbone without HR skip, L1.
- D `csnet_l1`: full CS-Net, L1.
- E `csnet_mse`: full CS-Net, MSE.

The causal comparisons are B vs D (compression), C vs D (HR skip), D vs E
(loss), and A vs D (accuracy/efficiency trade-off).

## Timing and efficiency

Inference timing uses 50 warm-up forwards and 100 timed forwards with CUDA
synchronization. The result JSON files document the FLOPs and peak-VRAM scope.

## Determinism

The runner records Python, PyTorch, CUDA, cuDNN, GPU, seed, and deterministic
settings. The recorded configuration uses deterministic algorithms with
`warn_only=True`; warnings should be retained because some GPU operators may
not have fully deterministic implementations.
"""


def data_doc() -> str:
    return """# Data availability and public-data policy

This GitHub package does not distribute 3D data volumes because of their size.
The following are excluded even when they are experimental outputs:

- paired GT/SIRT RAW files;
- reconstructed prediction RAW files;
- NumPy, NIfTI, MHA/MHD, HDF5, or similar multidimensional arrays;
- checkpoints and model weights.

The public `results/` folder contains only lightweight derived artifacts such
as CSV/JSON tables, PNG/PDF figures, and the final experiment report. These are
sufficient to inspect reported metrics without redistributing the volumes.

If sample data are later published through an external archive, use a separate
versioned release and provide checksums, data-generation settings, license, and
a stable DOI/URL. Do not commit the volumes directly to the Git repository.
"""


def path_doc() -> str:
    return """# Path configuration

`ablation_config.json` is the public, sanitized configuration. Replace only the
six dataset paths with local directories. The unified runner is the recommended
entry point.

Some legacy standalone scripts predate the unified runner and retain Windows
absolute paths from the original laboratory workstation. They are included for
methodological traceability. Search for `G:\\CT_RE` before using those scripts
and replace the paths locally. Do not treat those legacy paths as downloadable
data locations.
"""


def upload_checklist() -> str:
    return """# GitHub upload checklist

Before publication:

- [ ] Choose and add a software license.
- [ ] Fill in authors, paper title, DOI/preprint URL, and citation metadata.
- [ ] Review all code and result tables for third-party redistribution rights.
- [ ] Confirm the repository contains no patient/private/proprietary data.
- [ ] Confirm `manifests/no_volume_audit.json` reports zero forbidden files.
- [ ] Confirm no file exceeds GitHub's 100 MB hard limit.
- [ ] Keep checkpoints and all 3D volumes in an external archive, not Git.
- [ ] Create a tagged release and attach the ZIP if desired.
- [ ] Add representative figures to the GitHub repository description/README.

Recommended upload scope:

1. Source code and sanitized configuration.
2. Numeric result tables and selected publication figures.
3. Reproducibility/data-availability documentation.
4. File/checksum manifests.

Do not upload the original project directory wholesale.
"""


def gitignore_text() -> str:
    return """# 3D volumes and predictions
*.raw
*.npy
*.npz
*.nii
*.nii.gz
*.mha
*.mhd
*.h5
*.hdf5

# Model weights/checkpoints
*.pt
*.pth
*.ckpt
*.onnx
*.safetensors
checkpoints/
**/predictions/

# Generated runs and caches
outputs/
__pycache__/
*.py[cod]
.pytest_cache/
.idea/
.vscode/
tmp/
"""


def requirements_text() -> str:
    return """# Recorded experiments used Python 3.11.15 and torch 2.5.1+cu121.
# Install the correct CUDA build of PyTorch for your platform first.
torch==2.5.1
monai
numpy
scipy
scikit-image
matplotlib
Pillow
thop
# Dataset simulation / SIRT3D_CUDA generation; conda installation is often preferred:
astra-toolbox
"""


def audit_exclusions(project: Path) -> dict:
    counts = defaultdict(lambda: {"count": 0, "bytes": 0})
    scan_roots = [project / "checkpoints", project / "outputs", project / "output"]
    for root in scan_roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if item.name.lower().endswith(".nii.gz"):
                suffix = ".nii.gz"
            if suffix in FORBIDDEN_SUFFIXES:
                reason = "3D volume/prediction" if suffix not in {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"} else "model weight/checkpoint"
                key = "{} ({})".format(reason, suffix)
                counts[key]["count"] += 1
                counts[key]["bytes"] += item.stat().st_size
    return {
        "policy": "Forbidden files are excluded from the public package; no source file was deleted.",
        "source_exclusions": dict(sorted(counts.items())),
    }


def validate_package(destination: Path) -> dict:
    forbidden = []
    oversized = []
    file_count = 0
    total_bytes = 0
    for item in destination.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        total_bytes += item.stat().st_size
        lower = item.name.lower()
        suffix = ".nii.gz" if lower.endswith(".nii.gz") else item.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            forbidden.append(item.relative_to(destination).as_posix())
        if item.stat().st_size >= 100 * 1024 * 1024:
            oversized.append(item.relative_to(destination).as_posix())
    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "forbidden_file_count": len(forbidden),
        "forbidden_files": forbidden,
        "files_at_or_above_100_mib_count": len(oversized),
        "files_at_or_above_100_mib": oversized,
        "passed": not forbidden and not oversized,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    destination = args.destination.resolve()
    zip_path = args.zip_path.resolve()
    if destination.exists() or zip_path.exists():
        raise FileExistsError("Destination or ZIP already exists; refusing to overwrite.")
    destination.mkdir(parents=True)
    records = []

    for name in CODE_FILES:
        copy_exact(project / name, destination / name, records, "source_code")
    copy_exact(Path(__file__).resolve(), destination / "tools" / "prepare_github_release.py", records, "release_tool")

    # Lightweight per-run ablation evidence.
    ablation = project / "outputs" / "ablation"
    copy_tree_filtered(ablation / "summary", destination / "results" / "ablation" / "summary", records, "ablation_summary")
    if ablation.is_dir():
        for experiment in sorted(ablation.iterdir()):
            if not experiment.is_dir() or experiment.name.startswith("_"):
                continue
            for seed_dir in sorted(experiment.glob("seed_*")):
                copy_tree_filtered(seed_dir, destination / "results" / "ablation" / experiment.name / seed_dir.name, records, "ablation_run")

    # Other compact experiment outputs.
    selected_trees = [
        ("outputs/pcb7_pcb8_evaluation_20261019", "results/pcb7_pcb8/evaluation", "multi_pcb"),
        ("outputs/pcb7_pcb8_cross_ratio_20261022", "results/pcb7_pcb8/cross_ratio", "cross_ratio"),
        ("outputs/noise_robustness_200", "results/noise_robustness", "robustness"),
        ("outputs/unified_200_test", "results/unified_200_test", "unified_test"),
        ("outputs/phase1_audit_20261010", "results/phase1_audit", "audit"),
        ("output/figures", "results/figures", "figure"),
    ]
    for source_rel, destination_rel, category in selected_trees:
        copy_tree_filtered(project / source_rel, destination / destination_rel, records, category)

    report = project / "output" / "pdf" / "CS-Net_完整消融与多PCB多随机种子实验报告_20261022.pdf"
    copy_exact(report, destination / "results" / "reports" / report.name, records, "report")

    # Public configuration and documentation are new files; originals remain untouched.
    write_new(destination / "ablation_config.json", make_sanitized_config(project), records, "configuration")
    write_new(destination / "README.md", project_readme(), records, "documentation")
    write_new(destination / "requirements.txt", requirements_text(), records, "environment")
    write_new(destination / ".gitignore", gitignore_text(), records, "configuration")
    write_new(destination / "LICENSE_SELECTION_REQUIRED.md", "# License required\n\nChoose a software/data license before publishing. No license was inferred or added automatically.\n", records, "documentation")
    write_new(destination / "docs" / "REPRODUCIBILITY.md", reproducibility_doc(), records, "documentation")
    write_new(destination / "docs" / "DATA_AVAILABILITY.md", data_doc(), records, "documentation")
    write_new(destination / "docs" / "PATH_CONFIGURATION.md", path_doc(), records, "documentation")
    write_new(destination / "docs" / "GITHUB_UPLOAD_CHECKLIST.md", upload_checklist(), records, "documentation")

    env = json.loads((ablation / "run_manifest.json").read_text(encoding="utf-8"))
    public_env = {
        "environment": env.get("environment", {}),
        "experiments": env.get("experiments", []),
        "recorded_seeds": [20260916, 20260917, 20260918],
        "note": "Local absolute paths and invocation history intentionally omitted.",
    }
    write_new(destination / "results" / "ablation" / "recorded_environment.json", json.dumps(public_env, ensure_ascii=False, indent=2) + "\n", records, "environment")

    # Manifest paths are relative to the release root.
    for record in records:
        public = Path(record["public_path"])
        try:
            record["public_path"] = public.relative_to(destination).as_posix()
        except ValueError:
            pass
        source = Path(record["source_path"])
        if record["source_path"] != "generated for public release":
            try:
                record["source_path"] = source.relative_to(project).as_posix()
            except ValueError:
                record["source_path"] = source.name

    manifest_dir = destination / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_dir / "upload_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "public_path", "source_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["public_path"]))
    exclusion = audit_exclusions(project)
    (manifest_dir / "excluded_source_summary.json").write_text(json.dumps(exclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validation = validate_package(destination)
    (manifest_dir / "no_volume_audit.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError("Package validation failed: {}".format(validation))

    with zipfile.ZipFile(str(zip_path), "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in sorted(destination.rglob("*")):
            if item.is_file():
                archive.write(str(item), arcname=(destination.name + "/" + item.relative_to(destination).as_posix()))

    print(json.dumps({
        "destination": str(destination),
        "zip": str(zip_path),
        "files": validation["file_count"],
        "uncompressed_bytes": validation["total_bytes"],
        "zip_bytes": zip_path.stat().st_size,
        "forbidden_files": validation["forbidden_file_count"],
        "oversized_files": validation["files_at_or_above_100_mib_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
