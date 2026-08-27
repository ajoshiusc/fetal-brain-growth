"""Fit a local FetalSynthSeg-matched cross-sectional quantile reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from .references import DEFAULT_QUANTILES, parse_quantiles, quantile_column


def _validate_cohort(data: pd.DataFrame, minimum_subjects: int) -> pd.DataFrame:
    required = {"subject_id", "gestational_age_weeks", "region", "volume_ml"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Local reference data are missing columns: {sorted(missing)}")
    data = data.copy()
    data["gestational_age_weeks"] = pd.to_numeric(data["gestational_age_weeks"], errors="raise")
    data["volume_ml"] = pd.to_numeric(data["volume_ml"], errors="raise")
    if data.duplicated(["subject_id", "region"]).any():
        raise ValueError("Use one cross-sectional observation per subject and region.")
    age_counts = data.groupby("subject_id")["gestational_age_weeks"].nunique()
    if (age_counts != 1).any():
        raise ValueError("Each subject must have one consistent gestational age across regions.")
    if not np.isfinite(data[["gestational_age_weeks", "volume_ml"]]).all().all():
        raise ValueError("Age and volume must be finite.")
    if (data["volume_ml"] <= 0).any():
        raise ValueError("Volumes must be positive because fitting is performed in log-volume space.")
    for region, group in data.groupby("region"):
        if group["subject_id"].nunique() < minimum_subjects:
            raise ValueError(f"{region}: fewer than {minimum_subjects} independent subjects.")
        if group["gestational_age_weeks"].max() - group["gestational_age_weeks"].min() < 8:
            raise ValueError(f"{region}: gestational-age span is under 8 weeks.")
        age_bins = pd.cut(group["gestational_age_weeks"], bins=4, include_lowest=True).value_counts()
        if (age_bins < 5).any():
            raise ValueError(f"{region}: each of four equal-width age bins needs at least five controls.")
    return data


def _pipeline(method: str, quantile: float, alpha: float, spline_knots: int):
    regressor = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    if method in {"quadratic", "cubic"}:
        degree = 2 if method == "quadratic" else 3
        return make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            regressor,
        )
    return make_pipeline(
        SplineTransformer(n_knots=spline_knots, degree=3, include_bias=False),
        StandardScaler(),
        regressor,
    )


def fit_local_quantile_reference(
    volumes: pd.DataFrame | str | Path,
    *,
    method: str = "spline",
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    alpha: float = 0.001,
    spline_knots: int = 5,
    grid_step_weeks: float = 0.05,
    minimum_subjects: int = 120,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit direct quantile curves to a local, protocol-matched cohort.

    The input is long-form output from this package's volumetry functions. A
    cubic spline is the default; quadratic and cubic polynomials are available
    for sites that pre-specify a simpler model. Each quantile is fit directly
    in log-volume space, avoiding a Gaussian residual assumption.
    """

    if isinstance(volumes, (str, Path)):
        volumes = pd.read_csv(volumes)
    method = method.lower()
    if method not in {"spline", "quadratic", "cubic"}:
        raise ValueError("method must be spline, quadratic, or cubic.")
    quantiles = parse_quantiles(quantiles)
    data = _validate_cohort(volumes, minimum_subjects)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    crossing_corrections = 0
    for region, group in data.groupby("region", sort=True):
        x = group[["gestational_age_weeks"]].to_numpy(dtype=float)
        y = np.log(group["volume_ml"].to_numpy(dtype=float))
        grid = np.arange(x.min(), x.max() + grid_step_weeks / 2, grid_step_weeks)
        predictions = []
        for quantile in quantiles:
            model = _pipeline(method, quantile, alpha, spline_knots)
            model.fit(x, y)
            predictions.append(np.exp(model.predict(grid[:, None])))
        raw = np.stack(predictions, axis=1)
        ordered = np.sort(raw, axis=1)
        crossing_corrections += int(np.count_nonzero(np.abs(raw - ordered) > 1e-8))
        for index, age in enumerate(grid):
            row: dict[str, object] = {
                "region": str(region),
                "gestational_age_weeks": float(age),
            }
            for q_index, quantile in enumerate(quantiles):
                row[quantile_column(quantile)] = float(ordered[index, q_index])
            rows.append(row)
        diagnostics[str(region)] = {
            "subjects": int(group["subject_id"].nunique()),
            "age_min_weeks": float(x.min()),
            "age_max_weeks": float(x.max()),
        }
    metadata: dict[str, object] = {
        "source": "Local cross-sectional FetalSynthSeg-compatible cohort",
        "method": method,
        "response": "log(volume_ml)",
        "quantiles": list(quantiles),
        "regularization_alpha": alpha,
        "spline_knots": spline_knots if method == "spline" else None,
        "minimum_subjects_per_region": minimum_subjects,
        "crossing_values_reordered": crossing_corrections,
        "diagnostics": diagnostics,
        "warning": (
            "Research reference. Validate acquisition, reconstruction, segmentation, "
            "sampling, age estimation, and external performance before clinical use."
        ),
    }
    return pd.DataFrame(rows), metadata


def save_fitted_reference(
    curves: pd.DataFrame,
    metadata: dict[str, object],
    output_csv: str | Path,
) -> tuple[Path, Path]:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_csv, index=False)
    metadata_path = output_csv.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output_csv, metadata_path
