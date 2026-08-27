"""Standard-orientation MRI panels with high-resolution segmentation outlines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .labels import FETA_LABELS, LABEL_COLORS, LABEL_TITLES
from .volumetry import integer_labels


VIEW_SPECS = {
    "axial": {"axis": 2, "left": "R", "right": "L", "top": "A", "bottom": "P"},
    "coronal": {"axis": 1, "left": "R", "right": "L", "top": "S", "bottom": "I"},
    "sagittal": {"axis": 0, "left": "P", "right": "A", "top": "S", "bottom": "I"},
}


def _display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    raw = np.take(volume, index, axis=axis)
    shown = raw.T
    if axis in {1, 2}:  # radiological convention: patient right appears on image left
        shown = np.fliplr(shown)
    return shown


def _best_slice(labels: np.ndarray, axis: int) -> int:
    other_axes = tuple(value for value in range(3) if value != axis)
    areas = (labels > 0).sum(axis=other_axes)
    return int(np.argmax(areas))


def _crop_bounds(mask: np.ndarray, margin: int = 8) -> tuple[slice, slice]:
    coordinates = np.where(mask)
    if len(coordinates[0]) == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(int(coordinates[0].min()) - margin, 0)
    y1 = min(int(coordinates[0].max()) + margin + 1, mask.shape[0])
    x0 = max(int(coordinates[1].min()) - margin, 0)
    x1 = min(int(coordinates[1].max()) + margin + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def load_aligned_canonical(
    image_path: str | Path,
    segmentation_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
    image = nib.as_closest_canonical(nib.load(str(image_path)))
    segmentation = nib.as_closest_canonical(nib.load(str(segmentation_path)))
    aligned = resample_from_to(segmentation, (image.shape, image.affine), order=0, mode="constant", cval=0)
    intensity = np.nan_to_num(image.get_fdata(dtype=np.float32), copy=False)
    labels = integer_labels(aligned.get_fdata(dtype=np.float32))
    return intensity, labels, nib.aff2axcodes(image.affine)


def save_radiology_figure(
    image_path: str | Path,
    segmentation_path: str | Path,
    output_path: str | Path,
    *,
    subject_id: str | None = None,
    gestational_age_weeks: float | None = None,
    fill_alpha: float = 0.0,
    dpi: int = 300,
) -> Path:
    """Save axial/coronal/sagittal RAS panels in radiological convention."""

    intensity, labels, orientation = load_aligned_canonical(image_path, segmentation_path)
    foreground_intensity = intensity[labels > 0]
    if foreground_intensity.size == 0:
        foreground_intensity = intensity[np.isfinite(intensity)]
    vmin, vmax = np.percentile(foreground_intensity, [1, 99])
    if vmax <= vmin:
        vmin, vmax = float(intensity.min()), float(intensity.max() + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="#080B10")
    present_labels = [value for value in FETA_LABELS if value and np.any(labels == value)]
    for ax, (view, spec) in zip(axes, VIEW_SPECS.items()):
        axis = int(spec["axis"])
        index = _best_slice(labels, axis)
        image_slice = _display_slice(intensity, axis, index)
        label_slice = _display_slice(labels, axis, index)
        crop = _crop_bounds(label_slice > 0)
        image_slice = image_slice[crop]
        label_slice = label_slice[crop]
        ax.imshow(image_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        for value in present_labels:
            mask = label_slice == value
            if mask.any():
                if fill_alpha:
                    overlay = np.ma.masked_where(~mask, mask)
                    ax.imshow(
                        overlay,
                        origin="lower",
                        cmap=plt.matplotlib.colors.ListedColormap([LABEL_COLORS[value]]),
                        alpha=fill_alpha,
                        interpolation="nearest",
                    )
                ax.contour(mask.astype(float), levels=[0.5], colors=[LABEL_COLORS[value]], linewidths=1.15, origin="lower")
        ax.set_title(view.capitalize(), color="white", fontsize=13, weight="bold", pad=8)
        ax.text(0.015, 0.50, spec["left"], transform=ax.transAxes, color="white", va="center", fontsize=11, weight="bold")
        ax.text(0.985, 0.50, spec["right"], transform=ax.transAxes, color="white", ha="right", va="center", fontsize=11, weight="bold")
        ax.text(0.50, 0.985, spec["top"], transform=ax.transAxes, color="white", ha="center", va="top", fontsize=11, weight="bold")
        ax.text(0.50, 0.015, spec["bottom"], transform=ax.transAxes, color="white", ha="center", va="bottom", fontsize=11, weight="bold")
        ax.set_axis_off()
    title = subject_id or Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")
    if gestational_age_weeks is not None:
        title += f"  •  {gestational_age_weeks:.1f} weeks"
    fig.suptitle(title, color="white", fontsize=17, weight="bold", x=0.025, ha="left", y=0.99)
    handles = [
        Line2D([0], [0], color=LABEL_COLORS[value], lw=2.5, label=LABEL_TITLES[value])
        for value in present_labels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        frameon=False,
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.text(
        0.985,
        0.015,
        f"Canonical {''.join(orientation)} • radiological convention",
        ha="right",
        color="#B7C0CD",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.94), w_pad=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
