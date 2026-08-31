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


def _percentile_display(point: pd.Series) -> str:
    """Return a reader-facing percentile label, including explicit tail limits."""

    display = getattr(point, "percentile_display", None)
    if isinstance(display, str) and display:
        return display
    percentile = float(point.estimated_percentile_bounded)
    if point.status == "low_reference_flag" or percentile <= 3:
        return "P3 or lower"
    if point.status == "high_reference_flag" or percentile >= 97:
        return "P97 or higher"
    return f"P{percentile:.0f} (estimated)"


def _chart_column_count(region_count: int) -> int:
    """Choose two to four columns while minimizing unused report panels."""

    if region_count <= 1:
        return 1
    candidates = range(2, min(4, region_count) + 1)
    return min(
        candidates,
        key=lambda columns: (
            int(np.ceil(region_count / columns)) * columns - region_count,
            int(np.ceil(region_count / columns)),
        ),
    )


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
            ax.contour(mask.astype(float), levels=[0.5], colors=[LABEL_COLORS[label]], linewidths=1.45, origin="lower")
    ax.set_title(view.capitalize(), color="white", fontsize=17, weight="bold", pad=5)
    ax.text(0.015, 0.50, spec["left"], transform=ax.transAxes, color="white", va="center", fontsize=13, weight="bold")
    ax.text(0.985, 0.50, spec["right"], transform=ax.transAxes, color="white", ha="right", va="center", fontsize=13, weight="bold")
    ax.text(0.50, 0.985, spec["top"], transform=ax.transAxes, color="white", ha="center", va="top", fontsize=13, weight="bold")
    ax.text(0.50, 0.015, spec["bottom"], transform=ax.transAxes, color="white", ha="center", va="bottom", fontsize=13, weight="bold")
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
    dpi: int = 320,
) -> Path:
    """Save a radiology-ready real-case card with the requested reference panels."""

    intensity, labels, orientation = load_aligned_canonical(image_path, segmentation_path)
    foreground = intensity[labels > 0]
    vmin, vmax = np.percentile(foreground, [1, 99]) if foreground.size else (float(intensity.min()), float(intensity.max()))
    chart_columns = _chart_column_count(len(regions))
    chart_rows = int(np.ceil(len(regions) / chart_columns))
    fig = plt.figure(figsize=(20, 4.6 + 4.25 * chart_rows), facecolor="white")
    grid = fig.add_gridspec(
        1 + chart_rows,
        12,
        height_ratios=(0.90, *([1.0] * chart_rows)),
        hspace=0.30,
        wspace=0.28,
        left=0.045,
        right=0.985,
        bottom=0.095,
        top=0.925,
    )
    for index, view in enumerate(VIEW_SPECS):
        ax = fig.add_subplot(grid[0, index * 3 : (index + 1) * 3], facecolor="#080B10")
        _draw_image_panel(ax, intensity, labels, view, float(vmin), float(vmax))

    text_axis = fig.add_subplot(grid[0, 9:12])
    text_axis.axis("off")
    text_axis.text(0, 0.98, segmentation_source, va="top", fontsize=17, weight="bold", color=INK)
    if dice is not None:
        text_axis.text(0, 0.86, f"Mean tissue Dice: {dice.dice.mean():.3f}", va="top", fontsize=13, color=INK)
        dice_lines = [f"{row.tissue}: {row.dice:.3f}" for row in dice.itertuples(index=False)]
        midpoint = int(np.ceil(len(dice_lines) / 2))
        text_axis.text(0, 0.74, "\n".join(dice_lines[:midpoint]), va="top", fontsize=10.5, linespacing=1.25, color="#435166")
        text_axis.text(0.54, 0.74, "\n".join(dice_lines[midpoint:]), va="top", fontsize=10.5, linespacing=1.25, color="#435166")
    else:
        text_axis.text(
            0,
            0.82,
            "This automatic segmentation supplies every plotted volume.",
            va="top",
            fontsize=12,
            color="#435166",
            wrap=True,
        )
    legend = [
        Line2D([0], [0], color=LABEL_COLORS[label], lw=2.5, label=LABEL_TITLES[label])
        for label in sorted(FETA_LABELS) if label
    ]
    text_axis.legend(
        handles=legend,
        loc="lower left" if dice is not None else "upper left",
        bbox_to_anchor=None if dice is not None else (0, 0.66),
        frameon=False,
        fontsize=10.5,
        ncol=2,
        borderaxespad=0,
        columnspacing=0.8,
        handlelength=1.6,
    )

    for index, region in enumerate(regions):
        row, column = divmod(index, chart_columns)
        column_span = 12 // chart_columns
        ax = fig.add_subplot(
            grid[1 + row, column * column_span : (column + 1) * column_span]
        )
        curve = curves.loc[curves.region == region].sort_values("gestational_age_weeks")
        _draw_quantiles(ax, curve, list(DEFAULT_QUANTILES))
        point = scores.loc[scores.region == region].iloc[0]
        ax.scatter(
            point.gestational_age_weeks,
            point.volume_ml,
            s=120,
            marker="X" if point.status in {"low_reference_flag", "high_reference_flag"} else "o",
            color=POINT_COLORS.get(point.status, "#111111"),
            edgecolor="white",
            linewidth=1.1,
            zorder=7,
        )
        ax.set_title(
            f"{REGION_TITLES[region]}\n{point.volume_ml:.1f} mL • {_percentile_display(point)}",
            loc="left",
            fontsize=15,
            weight="bold",
            color=INK,
        )
        if row == chart_rows - 1:
            ax.set_xlabel("Gestational age (weeks)", fontsize=12.5)
        ax.set_ylabel("Volume (mL)", fontsize=12.5)
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11.5)

    fig.suptitle(
        f"{subject_id} • {gestational_age_weeks:.1f} weeks • real fetal T2 SVR",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=26,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.032,
        "Percentiles inside P3–P97 are interpolated estimates • Tail values are shown as P3 or lower / P97 or higher",
        fontsize=10.5,
        weight="bold",
        color="#435166",
    )
    fig.text(
        0.04,
        0.012,
        f"Canonical {''.join(orientation)} • radiological convention • research reference only, not a diagnosis",
        fontsize=10.5,
        color="#5D6877",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
