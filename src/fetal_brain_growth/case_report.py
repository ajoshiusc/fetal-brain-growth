"""Single-case image, segmentation, validation, and growth-chart report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .charts import INK, POINT_COLORS, _draw_quantiles
from .labels import FETA_LABELS, LABEL_COLORS, LABEL_TITLES, REGION_TITLES
from .radiology import VIEW_SPECS, _best_slice, _crop_bounds, _display_slice, load_aligned_canonical
from .references import DEFAULT_QUANTILES


def _draw_image_panel(
    ax: plt.Axes,
    intensity: np.ndarray,
    labels: np.ndarray,
    view: str,
    vmin: float,
    vmax: float,
) -> None:
    spec = VIEW_SPECS[view]
    axis = int(spec["axis"])
    index = _best_slice(labels, axis)
    background = _display_slice(intensity, axis, index)
    overlay = _display_slice(labels, axis, index)
    crop = _crop_bounds(overlay > 0)
    background, overlay = background[crop], overlay[crop]
    ax.imshow(background, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
    for label in sorted(FETA_LABELS):
        mask = overlay == label
        if label and mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[LABEL_COLORS[label]], linewidths=1.2, origin="lower")
    ax.set_title(view.capitalize(), color="white", fontsize=12, weight="bold")
    ax.text(0.015, 0.50, spec["left"], transform=ax.transAxes, color="white", va="center", weight="bold")
    ax.text(0.985, 0.50, spec["right"], transform=ax.transAxes, color="white", ha="right", va="center", weight="bold")
    ax.text(0.50, 0.985, spec["top"], transform=ax.transAxes, color="white", ha="center", va="top", weight="bold")
    ax.text(0.50, 0.015, spec["bottom"], transform=ax.transAxes, color="white", ha="center", va="bottom", weight="bold")
    ax.set_axis_off()


def save_case_report(
    image_path: str | Path,
    segmentation_path: str | Path,
    curves: pd.DataFrame,
    scores: pd.DataFrame,
    output_path: str | Path,
    *,
    subject_id: str,
    gestational_age_weeks: float,
    dice: pd.DataFrame | None = None,
    segmentation_source: str = "FetalSynthSeg prediction",
    regions: tuple[str, ...] = ("total_brain", "intracranial_volume", "external_csf", "cerebellum"),
    dpi: int = 300,
) -> Path:
    """Save a radiology-ready real-case card with four reference panels."""

    intensity, labels, orientation = load_aligned_canonical(image_path, segmentation_path)
    foreground = intensity[labels > 0]
    vmin, vmax = np.percentile(foreground, [1, 99]) if foreground.size else (float(intensity.min()), float(intensity.max()))
    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=(1.05, 1.0), hspace=0.30, wspace=0.27)
    for index, view in enumerate(VIEW_SPECS):
        ax = fig.add_subplot(grid[0, index], facecolor="#080B10")
        _draw_image_panel(ax, intensity, labels, view, float(vmin), float(vmax))

    text_axis = fig.add_subplot(grid[0, 3])
    text_axis.axis("off")
    text_axis.text(0, 0.98, segmentation_source, va="top", fontsize=13, weight="bold", color=INK)
    if dice is not None:
        text_axis.text(0, 0.88, f"Mean tissue Dice: {dice.dice.mean():.3f}", va="top", fontsize=11, color=INK)
        dice_lines = [f"{row.tissue}: {row.dice:.3f}" for row in dice.itertuples(index=False)]
        text_axis.text(0, 0.79, "\n".join(dice_lines), va="top", fontsize=9.2, linespacing=1.25, color="#435166")
    legend = [
        Line2D([0], [0], color=LABEL_COLORS[label], lw=2.5, label=LABEL_TITLES[label])
        for label in sorted(FETA_LABELS) if label
    ]
    text_axis.legend(handles=legend, loc="lower left", frameon=False, fontsize=8.5, ncol=1, borderaxespad=0)

    for index, region in enumerate(regions):
        ax = fig.add_subplot(grid[1, index])
        curve = curves.loc[curves.region == region].sort_values("gestational_age_weeks")
        _draw_quantiles(ax, curve, list(DEFAULT_QUANTILES))
        point = scores.loc[scores.region == region].iloc[0]
        ax.scatter(
            point.gestational_age_weeks,
            point.volume_ml,
            s=100,
            marker="X" if point.status in {"low_reference_flag", "high_reference_flag"} else "o",
            color=POINT_COLORS.get(point.status, "#111111"),
            edgecolor="white",
            linewidth=1.1,
            zorder=7,
        )
        ax.set_title(
            f"{REGION_TITLES[region]}\n{point.volume_ml:.1f} mL • P{point.estimated_percentile_bounded:.0f} (bounded)",
            loc="left",
            fontsize=11,
            weight="bold",
            color=INK,
        )
        ax.set_xlabel("Gestational age (weeks)")
        ax.set_ylabel("Volume (mL)")
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{subject_id} • {gestational_age_weeks:.1f} weeks • real fetal T2 SVR",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.012,
        f"Canonical {''.join(orientation)} • radiological convention • research reference only, not a diagnosis",
        fontsize=8.5,
        color="#5D6877",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
