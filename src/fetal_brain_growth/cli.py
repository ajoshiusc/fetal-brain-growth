"""Command-line interface for segmentation, volumetry, references, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .charts import save_growth_chart, save_published_polynomial_comparison
from .fitted_reference import fit_local_quantile_reference, save_fitted_reference
from .radiology import save_radiology_figure
from .references import (
    build_table_curves,
    parse_quantiles,
    published_polynomial_curve,
    score_against_curves,
)
from .volumetry import analyze_manifest


def _write_metadata(metadata: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def command_measure(args: argparse.Namespace) -> None:
    outputs = analyze_manifest(args.manifest, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def command_reference(args: argparse.Namespace) -> None:
    quantiles = parse_quantiles(args.quantiles)
    source = args.reference_table if args.reference_table else None
    if source is None:
        curves, metadata = build_table_curves(method=args.method, quantiles=quantiles)
    else:
        curves, metadata = build_table_curves(source, method=args.method, quantiles=quantiles)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output, index=False)
    _write_metadata(metadata, output.with_suffix(".json"))
    print(output)


def command_chart(args: argparse.Namespace) -> None:
    quantiles = parse_quantiles(args.quantiles)
    curves = pd.read_csv(args.curves)
    observations = pd.read_csv(args.observations) if args.observations else None
    if observations is not None and "status" not in observations:
        observations = score_against_curves(
            observations,
            curves,
            quantiles=quantiles,
            definition_guard=not args.matched_reference,
        )
        if args.scores:
            Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
            observations.to_csv(args.scores, index=False)
    regions = args.regions.split(",") if args.regions else None
    save_growth_chart(
        curves,
        args.output,
        observations=observations,
        regions=regions,
        quantiles=quantiles,
        title=args.title,
        subtitle=args.subtitle,
        dpi=args.dpi,
    )
    print(args.output)


def command_fit_reference(args: argparse.Namespace) -> None:
    curves, metadata = fit_local_quantile_reference(
        args.volumes,
        method=args.method,
        quantiles=parse_quantiles(args.quantiles),
        alpha=args.alpha,
        spline_knots=args.spline_knots,
        minimum_subjects=args.minimum_subjects,
    )
    outputs = save_fitted_reference(curves, metadata, args.output)
    print("\n".join(map(str, outputs)))


def command_radiology(args: argparse.Namespace) -> None:
    save_radiology_figure(
        args.image,
        args.segmentation,
        args.output,
        subject_id=args.subject_id,
        gestational_age_weeks=args.gestational_age,
        fill_alpha=args.fill_alpha,
        dpi=args.dpi,
    )
    print(args.output)


def command_published(args: argparse.Namespace) -> None:
    model_ids = args.models.split(",")
    curves = [
        published_polynomial_curve(model_id.strip(), models_path=args.models_json)
        if args.models_json
        else published_polynomial_curve(model_id.strip())
        for model_id in model_ids
    ]
    save_published_polynomial_comparison(curves, args.output, dpi=args.dpi)
    print(args.output)


def command_segment(args: argparse.Namespace) -> None:
    from .segmentation import segment_image

    metadata = segment_image(
        args.image,
        args.output,
        args.checkpoint,
        device_name=args.device,
        qc_path=args.qc,
        metadata_path=args.metadata,
        verify_checksum=not args.skip_checksum,
    )
    print(json.dumps(metadata, indent=2))


def command_feta_gallery(args: argparse.Namespace) -> None:
    from .feta_gallery import build_feta_gallery

    case_ids = args.case_ids.split(",") if args.case_ids else None
    kwargs = {"case_ids": case_ids} if case_ids else {}
    paths = build_feta_gallery(
        args.feta_root,
        args.output_dir,
        reference=args.reference,
        feta_degree=args.feta_degree,
        checkpoint=args.checkpoint,
        device_name=args.device,
        **kwargs,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def command_feta_reference(args: argparse.Namespace) -> None:
    from .feta_reference import build_feta_matched_reference, save_feta_matched_reference

    curves, metadata, tissues, matched, qc_records = build_feta_matched_reference(
        args.feta_root,
        prediction_dir=Path(args.output_dir) / "fetalsynthseg_predictions",
        checkpoint=args.checkpoint,
        device_name=args.device,
        degree=args.degree,
        quantiles=parse_quantiles(args.quantiles),
    )
    paths = save_feta_matched_reference(
        args.output_dir,
        curves,
        metadata,
        tissues,
        matched,
        qc_records,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fbg", description="Fetal MRI segmentation and growth-chart research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser("measure", help="Measure a manifest of FeTA-label segmentations")
    measure.add_argument("--manifest", required=True)
    measure.add_argument("--output-dir", required=True)
    measure.set_defaults(func=command_measure)

    reference = subparsers.add_parser("build-reference", help="Reconstruct Ren 2022 multi-quantile curves")
    reference.add_argument("--method", choices=("interpolate", "quadratic", "cubic", "auto"), default="interpolate")
    reference.add_argument("--reference-table", help="Optional mean/SD CSV in the documented schema")
    reference.add_argument("--quantiles", default="0.03,0.10,0.25,0.50,0.75,0.90,0.97")
    reference.add_argument("--output", required=True)
    reference.set_defaults(func=command_reference)

    chart = subparsers.add_parser("chart", help="Draw a multi-quantile growth chart")
    chart.add_argument("--curves", required=True)
    chart.add_argument("--observations")
    chart.add_argument("--scores")
    chart.add_argument("--regions", help="Comma-separated region identifiers")
    chart.add_argument("--quantiles", default="0.03,0.10,0.25,0.50,0.75,0.90,0.97")
    chart.add_argument("--title", default="Fetal brain volume reference")
    chart.add_argument("--subtitle")
    chart.add_argument(
        "--matched-reference",
        action="store_true",
        help="Disable literature label-definition guards for a protocol-matched local reference.",
    )
    chart.add_argument("--dpi", type=int, default=300)
    chart.add_argument("--output", required=True)
    chart.set_defaults(func=command_chart)

    fit_reference = subparsers.add_parser("fit-reference", help="Fit protocol-matched quantile curves")
    fit_reference.add_argument("--volumes", required=True)
    fit_reference.add_argument("--method", choices=("spline", "quadratic", "cubic"), default="spline")
    fit_reference.add_argument("--quantiles", default="0.03,0.10,0.25,0.50,0.75,0.90,0.97")
    fit_reference.add_argument("--alpha", type=float, default=0.001)
    fit_reference.add_argument("--spline-knots", type=int, default=5)
    fit_reference.add_argument("--minimum-subjects", type=int, default=120)
    fit_reference.add_argument("--output", required=True)
    fit_reference.set_defaults(func=command_fit_reference)

    radiology = subparsers.add_parser("radiology", help="Save standard-orientation outline panels")
    radiology.add_argument("--image", required=True)
    radiology.add_argument("--segmentation", required=True)
    radiology.add_argument("--subject-id")
    radiology.add_argument("--gestational-age", type=float)
    radiology.add_argument("--fill-alpha", type=float, default=0.0)
    radiology.add_argument("--dpi", type=int, default=300)
    radiology.add_argument("--output", required=True)
    radiology.set_defaults(func=command_radiology)

    published = subparsers.add_parser("published", help="Plot published coefficient-only mean models")
    published.add_argument("--models", default="jarvis2016_total_brain,ren2022_total_brain")
    published.add_argument("--models-json", help="Optional published-model JSON using references/published_models.json schema")
    published.add_argument("--dpi", type=int, default=300)
    published.add_argument("--output", required=True)
    published.set_defaults(func=command_published)

    segment = subparsers.add_parser("segment", help="Run the official FetalSynthSeg checkpoint")
    segment.add_argument("--image", required=True)
    segment.add_argument("--output", required=True)
    segment.add_argument("--checkpoint", required=True)
    segment.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    segment.add_argument("--qc")
    segment.add_argument("--metadata")
    segment.add_argument("--skip-checksum", action="store_true")
    segment.set_defaults(func=command_segment)

    feta_reference = subparsers.add_parser(
        "feta-reference",
        help="Fit all-region curves to automatic segmentations of neurotypical FeTA cases",
    )
    feta_reference.add_argument(
        "--feta-root",
        help="FeTA 2.2 BIDS root; defaults to FETA_ROOT or the detected local dataset",
    )
    feta_reference.add_argument("--degree", choices=(2, 3), type=int, default=2)
    feta_reference.add_argument("--checkpoint", help="FetalSynthSeg checkpoint; defaults to models/KISPI-all_fss.ckpt")
    feta_reference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    feta_reference.add_argument("--quantiles", default="0.03,0.10,0.25,0.50,0.75,0.90,0.97")
    feta_reference.add_argument("--output-dir", required=True)
    feta_reference.set_defaults(func=command_feta_reference)

    gallery = subparsers.add_parser(
        "feta-gallery",
        help="Build a ten-case FeTA gallery from automatic FetalSynthSeg predictions",
    )
    gallery.add_argument(
        "--feta-root",
        help="FeTA 2.2 BIDS root; defaults to FETA_ROOT or the detected local dataset",
    )
    gallery.add_argument("--output-dir", required=True)
    gallery.add_argument("--case-ids", help="Comma-separated list of exactly ten FeTA subject IDs")
    gallery.add_argument(
        "--reference",
        choices=("feta-neurotypical", "ren2022"),
        default="feta-neurotypical",
        help="Default uses all FeTA cases labeled Neurotypical; Ren is the literature alternative",
    )
    gallery.add_argument("--feta-degree", choices=(2, 3), type=int, default=2)
    gallery.add_argument("--checkpoint", help="FetalSynthSeg checkpoint; defaults to models/KISPI-all_fss.ckpt")
    gallery.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    gallery.set_defaults(func=command_feta_gallery)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
