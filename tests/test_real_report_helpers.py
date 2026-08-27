from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fetal_brain_growth.case_report import save_case_report
from fetal_brain_growth.references import build_table_curves, score_against_curves
from fetal_brain_growth.synthetic import make_synthetic_phantom
from fetal_brain_growth.validation import tissue_dice
from fetal_brain_growth.volumetry import measure_segmentation


def test_tissue_dice_resamples_to_reference_geometry(tmp_path: Path):
    labels = np.zeros((12, 14, 16), dtype=np.int16)
    labels[2:10, 3:11, 4:12] = 5
    reference = nib.Nifti1Image(labels, np.diag([1.0, 1.0, 1.0, 1.0]))
    predicted = nib.Nifti1Image(labels[::2, ::2, ::2], np.diag([2.0, 2.0, 2.0, 1.0]))
    reference_path = tmp_path / "reference.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    nib.save(reference, reference_path)
    nib.save(predicted, predicted_path)
    result = tissue_dice(predicted_path, reference_path)
    assert list(result.label) == list(range(1, 8))
    assert np.isfinite(result.dice).all()
    assert result.loc[result.label == 5, "dice"].iloc[0] > 0.65


def test_case_report_writes_radiology_card(tmp_path: Path):
    image = tmp_path / "image.nii.gz"
    segmentation = tmp_path / "segmentation.nii.gz"
    make_synthetic_phantom(image, segmentation, shape=(80, 88, 76))
    _, aggregates, _ = measure_segmentation(
        segmentation, subject_id="test-case", gestational_age_weeks=30.0
    )
    curves, _ = build_table_curves(grid_step_weeks=0.5)
    scores = score_against_curves(aggregates, curves)
    output = save_case_report(
        image,
        segmentation,
        curves,
        scores,
        tmp_path / "case_report.png",
        subject_id="test-case",
        gestational_age_weeks=30.0,
        dpi=100,
    )
    assert output.stat().st_size > 50_000

    extra_curve = curves.loc[curves.region == "total_brain"].copy()
    extra_curve["region"] = "white_matter"
    nine_curves = pd.concat([curves, extra_curve], ignore_index=True)
    nine_regions = tuple(nine_curves.region.drop_duplicates())
    nine_scores = []
    for region in nine_regions:
        curve = nine_curves.loc[nine_curves.region == region]
        point = curve.iloc[np.abs(curve.gestational_age_weeks - 30.0).argmin()]
        nine_scores.append(
            {
                "region": region,
                "gestational_age_weeks": 30.0,
                "volume_ml": point.p50_ml,
                "estimated_percentile_bounded": 50.0,
                "status": "within_reference_interval",
            }
        )
    nine_panel_output = save_case_report(
        image,
        segmentation,
        nine_curves,
        pd.DataFrame(nine_scores),
        tmp_path / "nine_panel_case_report.png",
        subject_id="test-case",
        gestational_age_weeks=30.0,
        regions=nine_regions,
        dpi=60,
    )
    assert nine_panel_output.stat().st_size > 50_000
