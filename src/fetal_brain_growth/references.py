"""Tabulated, fitted, and published-coefficient fetal volume references."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .labels import (
    DEFINITION_MISMATCH_NOTES,
    REFERENCE_MEASURES,
    SCOREABLE_REGIONS,
)


DEFAULT_QUANTILES = (0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97)


def packaged_data_path(name: str) -> Path:
    return Path(str(files("fetal_brain_growth").joinpath("data", name)))


DEFAULT_REN_TABLE = packaged_data_path("ren2022_weekly_mean_sd.csv")
DEFAULT_PUBLISHED_MODELS = packaged_data_path("published_models.json")


def quantile_column(quantile: float) -> str:
    value = f"{100 * float(quantile):g}".replace(".", "_")
    return f"p{value}_ml"


def parse_quantiles(values: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, str):
        quantiles = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    else:
        quantiles = tuple(float(value) for value in values)
    if len(quantiles) < 3 or any(not 0 < value < 1 for value in quantiles):
        raise ValueError("Provide at least three quantiles strictly between 0 and 1.")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("Quantiles must be unique and strictly increasing.")
    if 0.5 not in quantiles:
        raise ValueError("Quantiles must include 0.5.")
    return quantiles


def load_weekly_reference(path: str | Path = DEFAULT_REN_TABLE) -> pd.DataFrame:
    reference = pd.read_csv(path)
    required = {"measure", "gestational_age_weeks", "mean_ml", "sd_ml"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")
    reference = reference.copy()
    for column in ("gestational_age_weeks", "mean_ml", "sd_ml"):
        reference[column] = pd.to_numeric(reference[column], errors="raise")
    if (reference["sd_ml"] <= 0).any():
        raise ValueError("Reference SD values must be positive.")
    if reference.duplicated(["measure", "gestational_age_weeks"]).any():
        raise ValueError("Reference table contains duplicate measure/week rows.")
    return reference.sort_values(["measure", "gestational_age_weeks"]).reset_index(drop=True)


def _polynomial_cv_rmse(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    residuals = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        coefficients = np.polyfit(x[keep], y[keep], degree)
        residuals.append(float(np.polyval(coefficients, x[index]) - y[index]))
    return float(np.sqrt(np.mean(np.square(residuals))))


def _select_degree(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> tuple[int, dict[str, float]]:
    scores = {}
    mean_scale = max(float(np.ptp(mean)), float(np.mean(mean)), 1e-6)
    log_sd = np.log(sd)
    sd_scale = max(float(np.ptp(log_sd)), 1.0)
    for degree in (2, 3):
        mean_score = _polynomial_cv_rmse(x, mean, degree) / mean_scale
        sd_score = _polynomial_cv_rmse(x, log_sd, degree) / sd_scale
        scores[f"degree_{degree}_normalized_cv"] = mean_score + 0.25 * sd_score
    # Prefer the simpler model unless cubic improves the combined CV score by >5%.
    degree = 3 if scores["degree_3_normalized_cv"] < 0.95 * scores["degree_2_normalized_cv"] else 2
    return degree, scores


def build_table_curves(
    reference: pd.DataFrame | str | Path = DEFAULT_REN_TABLE,
    *,
    method: str = "interpolate",
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    grid_step_weeks: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create smooth multi-quantile curves from weekly published mean/SD values.

    ``interpolate`` preserves the table values. ``quadratic`` and ``cubic`` fit
    the mean and log(SD) across week. ``auto`` compares degree 2 and 3 using
    leave-one-week-out prediction error and favors quadratic unless cubic is
    materially better. Quantiles use an explicit Normal approximation.
    """

    if isinstance(reference, (str, Path)):
        reference = load_weekly_reference(reference)
    method = method.lower()
    if method not in {"interpolate", "quadratic", "cubic", "auto"}:
        raise ValueError("method must be interpolate, quadratic, cubic, or auto.")
    quantiles = parse_quantiles(quantiles)
    measure_to_region = {measure: region for region, measure in REFERENCE_MEASURES.items()}
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    floored_values = 0
    for measure, group in reference.groupby("measure", sort=False):
        group = group.sort_values("gestational_age_weeks")
        x = group["gestational_age_weeks"].to_numpy(dtype=float)
        observed_mean = group["mean_ml"].to_numpy(dtype=float)
        observed_sd = group["sd_ml"].to_numpy(dtype=float)
        grid = np.arange(x.min(), x.max() + grid_step_weeks / 2, grid_step_weeks)
        degree = None
        selection: dict[str, float] = {}
        if method == "interpolate":
            mean = np.interp(grid, x, observed_mean)
            sd = np.interp(grid, x, observed_sd)
        else:
            if method == "auto":
                degree, selection = _select_degree(x, observed_mean, observed_sd)
            else:
                degree = 2 if method == "quadratic" else 3
            mean_coefficients = np.polyfit(x, observed_mean, degree)
            log_sd_coefficients = np.polyfit(x, np.log(observed_sd), degree)
            mean = np.polyval(mean_coefficients, grid)
            sd = np.exp(np.polyval(log_sd_coefficients, grid))
            selection.update(
                {
                    "degree": degree,
                    "mean_rmse_ml": float(np.sqrt(np.mean(np.square(np.polyval(mean_coefficients, x) - observed_mean)))),
                    "mean_cv_rmse_ml": _polynomial_cv_rmse(x, observed_mean, degree),
                    "mean_coefficients_descending": mean_coefficients.tolist(),
                    "log_sd_coefficients_descending": log_sd_coefficients.tolist(),
                }
            )
        diagnostics[str(measure)] = selection
        for index, age in enumerate(grid):
            row: dict[str, object] = {
                "region": measure_to_region[str(measure)],
                "measure": str(measure),
                "gestational_age_weeks": float(age),
                "mean_ml": float(mean[index]),
                "sd_ml": float(sd[index]),
            }
            for quantile in quantiles:
                value = float(mean[index] + norm.ppf(quantile) * sd[index])
                if value < 0:
                    floored_values += 1
                    value = 0.0
                row[quantile_column(quantile)] = value
            rows.append(row)
    metadata: dict[str, object] = {
        "source": "Ren et al. 2022 Table 1 weekly mean and SD",
        "doi": "10.3389/fnins.2022.886083",
        "curve_method": method,
        "quantiles": list(quantiles),
        "quantile_model": "Normal approximation: mean(age) + Phi^-1(q) * SD(age)",
        "grid_step_weeks": grid_step_weeks,
        "nonnegative_floor_count": floored_values,
        "diagnostics": diagnostics,
        "warning": "Summary-data reconstruction, not a refit of individual participant data.",
    }
    return pd.DataFrame(rows), metadata


def score_against_curves(
    volumes: pd.DataFrame | str | Path,
    curves: pd.DataFrame | str | Path,
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    definition_guard: bool = True,
) -> pd.DataFrame:
    if isinstance(volumes, (str, Path)):
        volumes = pd.read_csv(volumes)
    if isinstance(curves, (str, Path)):
        curves = pd.read_csv(curves)
    quantiles = parse_quantiles(quantiles)
    q_columns = [quantile_column(value) for value in quantiles]
    missing = {"subject_id", "gestational_age_weeks", "region", "volume_ml"} - set(volumes.columns)
    if missing:
        raise ValueError(f"Volumes are missing columns: {sorted(missing)}")
    rows = []
    for record in volumes.itertuples(index=False):
        curve = curves.loc[curves["region"] == record.region].sort_values("gestational_age_weeks")
        if curve.empty:
            continue
        age = float(record.gestational_age_weeks)
        low_age, high_age = curve["gestational_age_weeks"].iloc[[0, -1]]
        if not float(low_age) <= age <= float(high_age):
            raise ValueError(f"{record.subject_id}: GA {age:g} is outside {low_age:g}-{high_age:g} weeks.")
        expected = {
            column: float(np.interp(age, curve["gestational_age_weeks"], curve[column]))
            for column in q_columns
        }
        observed = float(record.volume_ml)
        if definition_guard and record.region not in SCOREABLE_REGIONS:
            status = "comparison_only_definition_mismatch"
        elif observed < expected[q_columns[0]]:
            status = "low_reference_flag"
        elif observed > expected[q_columns[-1]]:
            status = "high_reference_flag"
        else:
            status = "within_reference_interval"
        quantile_values = np.array([expected[column] for column in q_columns])
        estimated_percentile = float(
            np.interp(observed, quantile_values, np.asarray(quantiles), left=quantiles[0], right=quantiles[-1])
        )
        if observed < quantile_values[0]:
            percentile_bound = "lower"
            percentile_display = f"P{100 * quantiles[0]:g} or lower"
        elif observed > quantile_values[-1]:
            percentile_bound = "upper"
            percentile_display = f"P{100 * quantiles[-1]:g} or higher"
        else:
            percentile_bound = None
            percentile_display = f"P{100 * estimated_percentile:.0f} (estimated)"
        rows.append(
            {
                "subject_id": record.subject_id,
                "gestational_age_weeks": age,
                "region": record.region,
                "volume_ml": observed,
                **expected,
                "estimated_percentile_bounded": 100 * estimated_percentile,
                "percentile_bound": percentile_bound,
                "percentile_display": percentile_display,
                "status": status,
                "interpretation_note": DEFINITION_MISMATCH_NOTES.get(
                    record.region,
                    "Definition-aligned research comparison; local validation is still required.",
                ),
            }
        )
    return pd.DataFrame(rows)


def load_published_models(path: str | Path = DEFAULT_PUBLISHED_MODELS) -> dict[str, dict[str, object]]:
    return json.loads(Path(path).read_text())


def published_polynomial_curve(
    model_id: str,
    *,
    models_path: str | Path = DEFAULT_PUBLISHED_MODELS,
    grid_step_weeks: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    models = load_published_models(models_path)
    if model_id not in models:
        raise ValueError(f"Unknown published model {model_id!r}; choose from {sorted(models)}.")
    model = models[model_id]
    low, high = map(float, model["age_range_weeks"])
    age = np.arange(low, high + grid_step_weeks / 2, grid_step_weeks)
    centered = age - float(model.get("age_center", 0.0))
    coefficients = np.asarray(model["coefficients_ascending"], dtype=float)
    linear_output = sum(coefficient * centered**power for power, coefficient in enumerate(coefficients))
    transform = model.get("output_transform", "identity")
    if transform == "identity":
        volume = linear_output
    elif transform == "square":
        volume = linear_output**2
    elif transform == "exp":
        volume = np.exp(linear_output)
    else:
        raise ValueError(f"Unsupported output transform {transform!r}.")
    curve = pd.DataFrame(
        {
            "region": model["region"],
            "measure": model["measure"],
            "gestational_age_weeks": age,
            "mean_ml": volume,
        }
    )
    return curve, {"model_id": model_id, **model}
