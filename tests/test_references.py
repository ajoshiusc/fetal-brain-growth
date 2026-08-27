from __future__ import annotations

import numpy as np
import pandas as pd

from fetal_brain_growth.fitted_reference import fit_local_quantile_reference
from fetal_brain_growth.references import (
    DEFAULT_QUANTILES,
    build_table_curves,
    published_polynomial_curve,
    quantile_column,
    score_against_curves,
)


def test_interpolated_reference_has_ordered_quantiles_and_exact_weekly_median():
    curves, metadata = build_table_curves(method="interpolate")
    columns = [quantile_column(value) for value in DEFAULT_QUANTILES]
    assert np.all(np.diff(curves[columns].to_numpy(), axis=1) >= 0)
    row = curves.loc[(curves.region == "total_brain") & np.isclose(curves.gestational_age_weeks, 30)].iloc[0]
    assert np.isclose(row.p50_ml, row.mean_ml)
    assert metadata["doi"] == "10.3389/fnins.2022.886083"


def test_polynomial_auto_reports_selected_degree():
    _, metadata = build_table_curves(method="auto", grid_step_weeks=0.5)
    for diagnostic in metadata["diagnostics"].values():
        assert diagnostic["degree"] in {2, 3}
        assert diagnostic["mean_cv_rmse_ml"] >= 0


def test_published_jarvis_equation():
    curve, metadata = published_polynomial_curve("jarvis2016_total_brain", grid_step_weeks=1)
    row = curve.loc[np.isclose(curve.gestational_age_weeks, 30)].iloc[0]
    assert np.isclose(row.mean_ml, 89.69 - 13.33 * 30 + 0.53 * 30**2)
    assert metadata["doi"] == "10.1002/pd.4961"


def test_definition_guard_prevents_overclassification():
    curves, _ = build_table_curves(grid_step_weeks=1)
    values = pd.DataFrame(
        [{"subject_id": "x", "gestational_age_weeks": 30, "region": "ventricles", "volume_ml": 999}]
    )
    score = score_against_curves(values, curves).iloc[0]
    assert score.status == "comparison_only_definition_mismatch"


def test_local_quadratic_quantile_reference():
    rng = np.random.default_rng(4)
    subjects = []
    for index, age in enumerate(np.linspace(20, 36, 40)):
        subjects.append(
            {
                "subject_id": f"s{index:03d}",
                "gestational_age_weeks": age,
                "region": "white_matter",
                "volume_ml": np.exp(1.3 + 0.09 * age + rng.normal(0, 0.08)),
            }
        )
    curves, metadata = fit_local_quantile_reference(
        pd.DataFrame(subjects), method="quadratic", minimum_subjects=30, grid_step_weeks=0.5
    )
    columns = [quantile_column(value) for value in DEFAULT_QUANTILES]
    assert np.all(np.diff(curves[columns].to_numpy(), axis=1) >= 0)
    assert metadata["method"] == "quadratic"
