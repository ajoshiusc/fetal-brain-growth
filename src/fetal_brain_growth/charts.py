"""High-resolution, radiologist-oriented fetal brain growth charts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labels import REGION_TITLES
from .references import DEFAULT_QUANTILES, quantile_column


INK = "#17233c"
BLUE = "#2463A6"
POINT_COLORS = {
    "within_reference_interval": "#14866D",
    "low_reference_flag": "#C43D3D",
    "high_reference_flag": "#C43D3D",
    "comparison_only_definition_mismatch": "#C47A15",
}


def _load(frame: pd.DataFrame | str | Path | None) -> pd.DataFrame | None:
    if frame is None or isinstance(frame, pd.DataFrame):
        return frame
    return pd.read_csv(frame)


def _present_quantiles(curves: pd.DataFrame, requested: Iterable[float]) -> list[float]:
    return [float(q) for q in requested if quantile_column(float(q)) in curves.columns]


def _draw_quantiles(ax, curve: pd.DataFrame, quantiles: list[float]) -> None:
    age = curve["gestational_age_weeks"].to_numpy(dtype=float)
    lower = [q for q in quantiles if q < 0.5]
    pairs = [(q, 1 - q) for q in lower if any(abs(candidate - (1 - q)) < 1e-8 for candidate in quantiles)]
    pairs.sort(key=lambda pair: pair[0])
    alphas = np.linspace(0.10, 0.28, max(len(pairs), 1))
    for alpha, (low, high) in zip(alphas, pairs):
        ax.fill_between(
            age,
            curve[quantile_column(low)].to_numpy(dtype=float),
            curve[quantile_column(high)].to_numpy(dtype=float),
            color=BLUE,
            alpha=float(alpha),
            linewidth=0,
            label=f"P{100*low:g}–P{100*high:g}",
        )
    for quantile in quantiles:
        values = curve[quantile_column(quantile)].to_numpy(dtype=float)
        if quantile == 0.5:
            ax.plot(age, values, color=INK, linewidth=2.2, label="P50", zorder=4)
        elif quantile in {0.03, 0.10, 0.90, 0.97}:
            ax.plot(age, values, color=BLUE, linewidth=0.8, alpha=0.75, zorder=3)


def save_growth_chart(
    curves: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    observations: pd.DataFrame | str | Path | None = None,
    regions: Iterable[str] | None = None,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    title: str = "Fetal brain volume reference",
    subtitle: str | None = None,
    dpi: int = 300,
) -> Path:
    curves = _load(curves)
    observations = _load(observations)
    assert curves is not None
    quantiles = _present_quantiles(curves, quantiles)
    if 0.5 not in quantiles:
        raise ValueError("Curves must contain a median column.")
    if regions is None:
        regions = list(dict.fromkeys(curves["region"].astype(str)))
    else:
        regions = list(regions)
    columns = 3 if len(regions) >= 7 else (2 if len(regions) > 1 else 1)
    rows = int(np.ceil(len(regions) / columns))
    panel_width = 6.2 if columns == 3 else 7.0
    fig, axes = plt.subplots(rows, columns, figsize=(panel_width * columns, 4.6 * rows), squeeze=False)
    for ax, region in zip(axes.flat, regions):
        curve = curves.loc[curves["region"] == region].sort_values("gestational_age_weeks")
        if curve.empty:
            ax.set_visible(False)
            continue
        _draw_quantiles(ax, curve, quantiles)
        if observations is not None:
            points = observations.loc[observations["region"] == region].sort_values(
                ["gestational_age_weeks", "volume_ml"]
            )
            multiple_points = len(points) > 1
            for point_index, point in enumerate(points.itertuples(index=False)):
                status = getattr(point, "status", "within_reference_interval")
                ax.scatter(
                    float(point.gestational_age_weeks),
                    float(point.volume_ml),
                    s=48,
                    facecolor=POINT_COLORS.get(status, "#111111"),
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=6,
                )
                if hasattr(point, "subject_id"):
                    if multiple_points:
                        amplitude = 6 + 5 * (point_index // 2)
                        y_offset = amplitude if point_index % 2 == 0 else -amplitude
                        x_offset = -4 if point_index >= len(points) - 2 else 4
                        horizontal_alignment = "right" if x_offset < 0 else "left"
                    else:
                        x_offset, y_offset, horizontal_alignment = 4, 4, "left"
                    ax.annotate(
                        str(point.subject_id),
                        (float(point.gestational_age_weeks), float(point.volume_ml)),
                        xytext=(x_offset, y_offset),
                        textcoords="offset points",
                        fontsize=7,
                        color=INK,
                        ha=horizontal_alignment,
                        va="bottom" if y_offset >= 0 else "top",
                        bbox={
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.70,
                        },
                        arrowprops=(
                            {"arrowstyle": "-", "color": "#8793A6", "linewidth": 0.5}
                            if multiple_points
                            else None
                        ),
                    )
        ax.set_title(REGION_TITLES.get(region, region.replace("_", " ").title()), loc="left", weight="bold")
        ax.set_xlabel("Gestational age (weeks)")
        ax.set_ylabel("Volume (mL)")
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)
    for ax in axes.flat[len(regions):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        fig.legend(unique.values(), unique.keys(), loc="upper right", frameon=False, ncol=min(4, len(unique)))
    fig.suptitle(title, x=0.06, y=0.995, ha="left", color=INK, fontsize=18, weight="bold")
    if subtitle:
        fig.text(0.06, 0.965, subtitle, ha="left", va="top", color="#4C5A6D", fontsize=9)
    fig.text(
        0.06,
        0.012,
        "Research use only • A centile flag is not a diagnosis • Confirm segmentation QC and reference compatibility",
        color="#5D6877",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.94 if subtitle else 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_published_polynomial_comparison(
    curves: Iterable[tuple[pd.DataFrame, dict[str, object]]],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for curve, metadata in curves:
        ax.plot(
            curve["gestational_age_weeks"],
            curve["mean_ml"],
            linewidth=2.3,
            label=str(metadata.get("title", metadata.get("model_id", "Published model"))),
        )
    ax.set_title("Published total-brain polynomial models", loc="left", fontsize=16, weight="bold", color=INK)
    ax.set_xlabel("Gestational age (weeks)")
    ax.set_ylabel("Predicted mean volume (mL)")
    ax.grid(axis="y", color="#DDE3EA")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.11, 0.01, "Mean curves only; published coefficients do not define centiles.", fontsize=9, color="#5D6877")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
