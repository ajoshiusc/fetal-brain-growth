from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fetal_brain_growth.cli import build_parser
from fetal_brain_growth.feta_reference import fit_feta_matched_reference, generate_feta_predictions
from fetal_brain_growth.labels import FETA_MATCHED_REFERENCE_REGIONS
from fetal_brain_growth.references import DEFAULT_QUANTILES, quantile_column


def _matched_control_volumes() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    rows = []
    ages = np.linspace(22.7, 34.8, 30)
    for region_index, region in enumerate(FETA_MATCHED_REFERENCE_REGIONS):
        for subject_index, age in enumerate(ages):
            log_volume = (
                0.8
                + 0.22 * region_index
                + 0.105 * age
                - 0.0012 * (age - 28.5) ** 2
                + rng.normal(0, 0.06)
            )
            rows.append(
                {
                    "subject_id": f"sub-{subject_index:03d}",
                    "gestational_age_weeks": age,
                    "region": region,
                    "volume_ml": np.exp(log_volume),
                }
            )
    return pd.DataFrame(rows)


def test_feta_matched_reference_covers_all_labels_with_ordered_quantiles():
    curves, metadata = fit_feta_matched_reference(
        _matched_control_volumes(),
        degree=2,
        grid_step_weeks=0.25,
    )

    columns = [quantile_column(value) for value in DEFAULT_QUANTILES]
    assert list(metadata["regions"]) == list(FETA_MATCHED_REFERENCE_REGIONS)
    assert metadata["subjects"] == 30
    assert metadata["degree"] == 2
    assert set(curves.region) == set(FETA_MATCHED_REFERENCE_REGIONS)
    assert np.all(np.diff(curves[columns].to_numpy(), axis=1) >= 0)


def test_feta_matched_reference_rejects_unsupported_degree():
    with pytest.raises(ValueError, match="degree must be 2 or 3"):
        fit_feta_matched_reference(_matched_control_volumes(), degree=4)


def test_feta_gallery_defaults_to_protocol_matched_quadratic_reference():
    args = build_parser().parse_args(["feta-gallery", "--output-dir", "out"])
    assert args.reference == "feta-neurotypical"
    assert args.feta_degree == 2
    assert args.device == "auto"


def test_feta_reference_defaults_to_automatic_segmentation():
    args = build_parser().parse_args(["feta-reference", "--output-dir", "out"])
    assert args.checkpoint is None
    assert args.device == "auto"


def test_feta_prediction_cache_reuses_provenance_matched_output_without_torch(tmp_path):
    feta_root = tmp_path / "feta"
    image = feta_root / "sub-001" / "anat" / "sub-001_T2w.nii.gz"
    image.parent.mkdir(parents=True)
    image.touch()
    (feta_root / "participants.tsv").write_text(
        "participant_id\tPathology\tGestational age\nsub-001\tNeurotypical\t30\n"
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction = prediction_dir / "sub-001_fetalsynthseg.nii.gz"
    prediction.touch()
    (prediction_dir / "sub-001_fetalsynthseg.json").write_text(
        json.dumps(
            {
                "input": str(image.resolve()),
                "output": str(prediction.resolve()),
                "checkpoint_sha256": "recorded-checksum",
            }
        )
    )

    paths = generate_feta_predictions(
        feta_root,
        ["sub-001"],
        prediction_dir,
        checkpoint=tmp_path / "missing-checkpoint.ckpt",
    )

    assert paths == {"sub-001": prediction.resolve()}
