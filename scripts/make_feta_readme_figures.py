#!/usr/bin/env python3
"""Generate README figures from an automatically segmented real FeTA SVR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fetal_brain_growth.case_report import save_case_report
from fetal_brain_growth.charts import save_growth_chart
from fetal_brain_growth.feta_reference import (
    build_feta_matched_reference,
    find_feta_image,
    generate_feta_predictions,
    load_feta_participants,
    resolve_feta_root,
    save_feta_matched_reference,
)
from fetal_brain_growth.labels import FETA_MATCHED_REFERENCE_REGIONS
from fetal_brain_growth.radiology import save_radiology_figure
from fetal_brain_growth.references import score_against_curves
from fetal_brain_growth.volumetry import measure_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically segment a real FeTA MRI and create its README analysis figures."
    )
    parser.add_argument("--feta-root", help="FeTA 2.2 root; otherwise use FETA_ROOT or the known local path")
    parser.add_argument("--subject-id", default="sub-050")
    parser.add_argument("--reference-dir", default="meeting_outputs/feta_10_cases_matched")
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument("--degree", type=int, choices=(2, 3), default=2)
    parser.add_argument("--checkpoint", default="models/KISPI-all_fss.ckpt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    feta_root = resolve_feta_root(args.feta_root)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = reference_dir / "feta_neurotypical_reference_curves.csv"
    metadata_path = reference_dir / "feta_neurotypical_reference_metadata.json"
    prediction_dir = reference_dir / "fetalsynthseg_predictions"
    cached_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if (
        curves_path.exists()
        and cached_metadata.get("segmentation_method") == "FetalSynthSeg v1 automatic prediction"
        and cached_metadata.get("checkpoint_sha256")
    ):
        curves = pd.read_csv(curves_path)
        metadata = cached_metadata
    else:
        curves, metadata, control_tissues, control_volumes, control_qc = build_feta_matched_reference(
            feta_root,
            prediction_dir=prediction_dir,
            checkpoint=args.checkpoint,
            device_name=args.device,
            degree=args.degree,
        )
        save_feta_matched_reference(
            reference_dir,
            curves,
            metadata,
            control_tissues,
            control_volumes,
            control_qc,
        )

    participants = load_feta_participants(feta_root)
    participant = participants.loc[participants.participant_id == args.subject_id]
    if len(participant) != 1:
        raise ValueError(f"Could not uniquely identify {args.subject_id!r} in participants.tsv.")
    gestational_age = float(participant["Gestational age"].iloc[0])
    phenotype = str(participant.Pathology.iloc[0])
    image_path = find_feta_image(feta_root, args.subject_id)
    segmentation_path = generate_feta_predictions(
        feta_root,
        [args.subject_id],
        prediction_dir,
        checkpoint=args.checkpoint,
        device_name=args.device,
    )[args.subject_id]
    tissues, aggregates, qc = measure_segmentation(
        segmentation_path,
        subject_id=args.subject_id,
        gestational_age_weeks=gestational_age,
    )
    case_volumes = pd.concat(
        [
            aggregates.loc[aggregates.region.isin(("total_brain", "intracranial_volume"))],
            tissues,
        ],
        ignore_index=True,
        sort=False,
    )
    scores = score_against_curves(case_volumes, curves, definition_guard=False)

    save_radiology_figure(
        image_path,
        segmentation_path,
        output_dir / "real_fetal_svr_segmentation_qc.png",
        subject_id=f"FeTA {args.subject_id} • automatic FetalSynthSeg prediction",
        gestational_age_weeks=gestational_age,
        fill_alpha=0.05,
        dpi=300,
    )
    save_growth_chart(
        curves,
        output_dir / "real_fetal_svr_growth_chart.png",
        observations=scores,
        regions=FETA_MATCHED_REFERENCE_REGIONS,
        title="Real fetal SVR case on FeTA-matched quantiles",
        subtitle=(
            f"{args.subject_id} • {gestational_age:.1f} weeks • automatic segmentation • "
            f"FeTA phenotype: {phenotype} • {metadata['subjects']}-control teaching reference"
        ),
        dpi=300,
    )
    save_case_report(
        image_path,
        segmentation_path,
        curves,
        scores,
        output_dir / "real_fetal_svr_case_report.png",
        subject_id=f"FeTA {args.subject_id}",
        gestational_age_weeks=gestational_age,
        segmentation_source=f"FetalSynthSeg automatic prediction • {phenotype}",
        regions=FETA_MATCHED_REFERENCE_REGIONS,
        dpi=360,
    )
    print(
        json.dumps(
            {
                "subject_id": args.subject_id,
                "gestational_age_weeks": gestational_age,
                "phenotype": phenotype,
                "segmentation_qc_warnings": qc.get("warnings", []),
                "reference_subjects": metadata["subjects"],
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
