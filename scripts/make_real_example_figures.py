#!/usr/bin/env python3
"""Generate documentation from the official real 30-week fetal atlas example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fetal_brain_growth.case_report import save_case_report
from fetal_brain_growth.charts import save_growth_chart
from fetal_brain_growth.labels import REFERENCE_GROUPS
from fetal_brain_growth.radiology import save_radiology_figure
from fetal_brain_growth.references import build_table_curves, score_against_curves
from fetal_brain_growth.validation import tissue_dice
from fetal_brain_growth.volumetry import measure_segmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Official sub-sta30 T2w NIfTI")
    parser.add_argument("--predicted-segmentation", required=True, help="FetalSynthSeg output NIfTI")
    parser.add_argument("--manual-segmentation", required=True, help="Official manual label map")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--gestational-age", type=float, default=30.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    subject_id = "IMAGINE atlas example sub-sta30"
    dice = tissue_dice(args.predicted_segmentation, args.manual_segmentation)
    tissues, aggregates, qc = measure_segmentation(
        args.predicted_segmentation,
        subject_id="sub-sta30",
        gestational_age_weeks=args.gestational_age,
    )
    curves, curve_metadata = build_table_curves(method="interpolate")
    scores = score_against_curves(aggregates, curves)

    save_radiology_figure(
        args.image,
        args.predicted_segmentation,
        images_dir / "real_fetal_segmentation_qc.png",
        subject_id=subject_id,
        gestational_age_weeks=args.gestational_age,
        fill_alpha=0.05,
        dpi=300,
    )
    save_growth_chart(
        curves,
        images_dir / "real_fetal_growth_chart.png",
        observations=scores,
        regions=tuple(REFERENCE_GROUPS),
        title="Public fetal MRI example across all Ren 2022 volume measures",
        subtitle=(
            "30-week CC0 IMAGINE atlas • FetalSynthSeg prediction • "
            "green = definition-aligned screen; orange = comparison only"
        ),
        dpi=300,
    )
    save_case_report(
        args.image,
        args.predicted_segmentation,
        curves,
        scores,
        images_dir / "real_fetal_case_report.png",
        subject_id=subject_id,
        gestational_age_weeks=args.gestational_age,
        dice=dice,
        regions=tuple(REFERENCE_GROUPS),
        dpi=360,
    )

    dice.to_csv(output_dir / "real_example_dice.csv", index=False)
    tissues.to_csv(output_dir / "real_example_tissue_volumes.csv", index=False)
    scores.to_csv(output_dir / "real_example_reference_scores.csv", index=False)
    summary = {
        "example": "sub-sta30, 30-week neurotypical fetal T2 atlas image",
        "image_source": "IMAGINE Fetal T2-weighted MRI Atlas",
        "source_doi": "10.7910/DVN/WE9JVR",
        "source_license": "CC0 1.0",
        "segmentation": "FetalSynthSeg v1 KISPI-all_fss.ckpt prediction",
        "fetalsynthseg_commit": "03c439edef02fc830e31a38169c5aa09ca98eeb4",
        "mean_tissue_dice": float(dice.dice.mean()),
        "minimum_tissue_dice": float(dice.dice.min()),
        "maximum_tissue_dice": float(dice.dice.max()),
        "segmentation_qc": qc,
        "growth_reference": curve_metadata,
        "warning": "One-case execution check and research comparison; not diagnostic validation.",
    }
    (output_dir / "real_example_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
