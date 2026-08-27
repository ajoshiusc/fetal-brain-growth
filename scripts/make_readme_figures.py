#!/usr/bin/env python3
"""Generate public-safe documentation figures from a synthetic phantom."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fetal_brain_growth.charts import save_growth_chart, save_published_polynomial_comparison
from fetal_brain_growth.radiology import save_radiology_figure
from fetal_brain_growth.references import build_table_curves, published_polynomial_curve, score_against_curves
from fetal_brain_growth.synthetic import make_synthetic_phantom


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output = project / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        image = Path(temporary) / "synthetic_svr.nii.gz"
        segmentation = Path(temporary) / "synthetic_seg.nii.gz"
        make_synthetic_phantom(image, segmentation)
        save_radiology_figure(
            image,
            segmentation,
            output / "synthetic_segmentation_qc.png",
            subject_id="Synthetic demonstration",
            gestational_age_weeks=30.0,
            fill_alpha=0.08,
            dpi=300,
        )

    curves, _ = build_table_curves(method="interpolate")
    observations = pd.DataFrame(
        [
            {"subject_id": "demo", "gestational_age_weeks": 30.0, "region": "total_brain", "volume_ml": 188.0},
            {"subject_id": "demo", "gestational_age_weeks": 30.0, "region": "intracranial_volume", "volume_ml": 232.0},
            {"subject_id": "demo", "gestational_age_weeks": 30.0, "region": "external_csf", "volume_ml": 33.0},
            {"subject_id": "demo", "gestational_age_weeks": 30.0, "region": "cerebellum", "volume_ml": 11.0},
        ]
    )
    scores = score_against_curves(observations, curves)
    save_growth_chart(
        curves,
        output / "multi_quantile_growth_chart.png",
        observations=scores,
        regions=["total_brain", "intracranial_volume", "external_csf", "cerebellum"],
        title="Fetal brain volume overview",
        subtitle="Ren 2022 weekly mean/SD • reconstructed Normal-approximation centiles",
        dpi=300,
    )
    models = [published_polynomial_curve("jarvis2016_total_brain"), published_polynomial_curve("ren2022_total_brain")]
    save_published_polynomial_comparison(models, output / "published_polynomial_models.png", dpi=300)


if __name__ == "__main__":
    main()
