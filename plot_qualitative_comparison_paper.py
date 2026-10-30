from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SAMPLE_NAME = "board5_patch0000_scale2x"
SHAPE = (40, 256, 256)
Z_SLICE, Y_SLICE, X_SLICE = 8, 128, 128

PRED_DIR = Path(r"G:\CT_RE\dataset\Final_Predictions")
DIRTY_DIR = Path(r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test")
GT_DIR = Path(r"G:\CT_RE\dataset\Final_Adaptive_GT\test")
OUT_DIR = Path(r"G:\CT_RE\CT_DRA\output\figures")

METHODS = [
    ("SIRT", DIRTY_DIR / f"{SAMPLE_NAME}_dirty.raw"),
    ("FDK", PRED_DIR / f"{SAMPLE_NAME}_fdk_pred.raw"),
    ("3D U-Net", PRED_DIR / f"{SAMPLE_NAME}_unet_pred.raw"),
    ("UNETR", PRED_DIR / f"{SAMPLE_NAME}_unetr_pred.raw"),
    ("Swin-UNETR", PRED_DIR / f"{SAMPLE_NAME}_clean_pred.raw"),
    ("CS-Net (Ours)", PRED_DIR / f"{SAMPLE_NAME}_fast_robust_pred.raw"),
    ("Ground truth", GT_DIR / f"{SAMPLE_NAME}.raw"),
]


def load_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing volume: {path}")
    n_voxels = int(np.prod(SHAPE))
    if path.stat().st_size == n_voxels * np.dtype(np.float16).itemsize:
        dtype = np.float16
    elif path.stat().st_size == n_voxels * np.dtype(np.float32).itemsize:
        dtype = np.float32
    else:
        raise ValueError(f"Unexpected file size: {path}")
    return np.fromfile(path, dtype=dtype).astype(np.float32).reshape(SHAPE)


def add_panel_border(ax, color="#5a5a5a", linewidth=0.45):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    volumes = [(name, load_volume(path)) for name, path in METHODS]
    n_cols = len(volumes)

    # The orthogonal views retain the true 40:256 height ratio instead of
    # being stretched into squares. This makes geometry directly comparable.
    fig = plt.figure(figsize=(7.25, 1.72), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[1.0, SHAPE[0] / SHAPE[1], SHAPE[0] / SHAPE[2]],
        left=0.058,
        right=0.997,
        bottom=0.055,
        top=0.825,
        wspace=0.035,
        hspace=0.085,
    )

    row_labels = ["XY", "XZ", "YZ"]
    row_axes = []
    for col, (name, volume) in enumerate(volumes):
        views = (
            volume[Z_SLICE, :, :],
            volume[:, Y_SLICE, :],
            volume[:, :, X_SLICE],
        )
        for row, view in enumerate(views):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                row_axes.append(ax)
            ax.imshow(
                view,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
                origin="upper",
            )
            add_panel_border(ax)
            if row == 0:
                title_weight = "bold" if "Ours" in name else "normal"
                title_color = "#1f4e79" if "Ours" in name else "black"
                ax.set_title(name, pad=3.2, fontweight=title_weight, color=title_color)

    # Plane abbreviations remain legible after reduction to a two-column width.
    for ax, label in zip(row_axes, row_labels):
        pos = ax.get_position()
        fig.text(
            0.048,
            (pos.y0 + pos.y1) / 2,
            label,
            ha="right",
            va="center",
            fontsize=7.4,
        )

    # Compact panel letters provide an unambiguous reference in the manuscript.
    for col in range(n_cols):
        ax = fig.axes[col * 3]
        ax.text(
            0.02,
            0.98,
            f"({chr(97 + col)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"qualitative_orthogonal_comparison_{SAMPLE_NAME}_paper"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
