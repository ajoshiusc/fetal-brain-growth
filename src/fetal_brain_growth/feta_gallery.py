"""Build a ten-case FeTA teaching set from automatic segmentations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .case_report import save_case_report
from .charts import save_growth_chart
from .feta_reference import (
    build_feta_matched_reference,
    find_feta_image,
    generate_feta_predictions,
    load_feta_participants,
    resolve_feta_root,
    save_feta_matched_reference,
)
from .labels import FETA_LABELS, FETA_MATCHED_REFERENCE_REGIONS, LABEL_COLORS
from .radiology import _best_slice, _crop_bounds, _display_slice, load_aligned_canonical
from .references import build_table_curves, score_against_curves
from .volumetry import measure_segmentation


DEFAULT_CASE_IDS = (
    "sub-036", "sub-027", "sub-034", "sub-051", "sub-061",
    "sub-007", "sub-014", "sub-001", "sub-019", "sub-050",
)
REN_SCOREABLE_REGIONS = ("total_brain", "intracranial_volume", "external_csf", "cerebellum")
PHENOTYPE_COLORS = {"Neurotypical": "#1976A3", "Pathological": "#7B3F91"}


def _case_status(
    scores: pd.DataFrame,
    score_regions: tuple[str, ...],
    *,
    interval_description: str,
) -> tuple[str, str]:
    scores = scores.loc[scores.region.isin(score_regions)]
    flagged = scores.loc[scores.status.isin({"low_reference_flag", "high_reference_flag"})]
    if flagged.empty:
        return (
            f"within_all_{len(score_regions)}_reference_intervals",
            f"Within all {len(score_regions)} {interval_description} reference intervals",
        )
    details = []
    for row in flagged.itertuples(index=False):
        direction = "low" if row.status == "low_reference_flag" else "high"
        details.append(f"{row.region}: {direction}, {row.percentile_display}")
    return "one_or_more_reference_flags", "; ".join(details)


def _save_overview(records: list[dict[str, object]], output_path: Path, *, dpi: int = 240) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 9.0), facecolor="white")
    for ax, record in zip(axes.flat, records):
        intensity, labels, _ = load_aligned_canonical(record["image_path"], record["segmentation_path"])
        index = _best_slice(labels, 2)
        background = _display_slice(intensity, 2, index)
        overlay = _display_slice(labels, 2, index)
        crop = _crop_bounds(overlay > 0)
        background, overlay = background[crop], overlay[crop]
        foreground = background[overlay > 0]
        low, high = np.percentile(foreground, [1, 99])
        ax.imshow(background, cmap="gray", origin="lower", vmin=low, vmax=high)
        for label in sorted(FETA_LABELS):
            mask = overlay == label
            if label and mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=[LABEL_COLORS[label]], linewidths=0.85, origin="lower")
        flagged = record["volume_screen"] == "one_or_more_reference_flags"
        screen_text = (
            "reference flag(s)"
            if flagged
            else f"within {record['reference_interval_count']} intervals"
        )
        ax.set_title(
            f"{record['subject_id']} • {record['gestational_age_weeks']:.1f} w\n"
            f"{record['feta_phenotype']} • {screen_text}",
            fontsize=11.5,
            color=PHENOTYPE_COLORS.get(str(record["feta_phenotype"]), "#17233C"),
            weight="bold",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(3 if flagged else 1.2)
            spine.set_edgecolor("#C43D3D" if flagged else "#8190A5")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Ten real FeTA fetal MRIs with automatic FetalSynthSeg outlines",
        fontsize=23,
        weight="bold",
        color="#17233C",
    )
    fig.text(
        0.5,
        0.015,
        "Title color is the FeTA dataset phenotype; red frame means ≥1 research reference flag. Neither is a diagnosis generated here.",
        ha="center",
        fontsize=11,
        color="#5D6877",
    )
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_feta_gallery(
    feta_root: str | Path | None,
    output_dir: str | Path,
    *,
    case_ids: Iterable[str] = DEFAULT_CASE_IDS,
    reference: str = "feta-neurotypical",
    feta_degree: int = 2,
    checkpoint: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Path]:
    """Create local teaching figures and measurements from automatic predictions.

    Raw FeTA data and generated case figures remain subject to the dataset's
    research/education terms and should not be committed automatically.
    """

    feta_root = resolve_feta_root(feta_root)
    output_dir = Path(output_dir).resolve()
    case_ids = tuple(case_ids)
    if len(case_ids) != 10 or len(set(case_ids)) != 10:
        raise ValueError("Provide exactly ten unique case IDs.")
    participants = load_feta_participants(feta_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "fetalsynthseg_predictions"
    if reference == "feta-neurotypical":
        curves, metadata, control_tissues, control_matched, control_qc = build_feta_matched_reference(
            feta_root,
            prediction_dir=prediction_dir,
            checkpoint=checkpoint,
            device_name=device_name,
            degree=feta_degree,
        )
        save_feta_matched_reference(
            output_dir,
            curves,
            metadata,
            control_tissues,
            control_matched,
            control_qc,
        )
        score_regions = tuple(FETA_MATCHED_REFERENCE_REGIONS)
        definition_guard = False
        interval_description = "protocol-matched"
    elif reference == "ren2022":
        curves, metadata = build_table_curves(method="interpolate")
        score_regions = REN_SCOREABLE_REGIONS
        definition_guard = True
        interval_description = "definition-aligned"
    else:
        raise ValueError("reference must be feta-neurotypical or ren2022.")
    cards = output_dir / "case_cards"
    cards.mkdir(exist_ok=True)
    tissues, aggregates, qc_records, records = [], [], [], []
    prediction_paths = generate_feta_predictions(
        feta_root,
        case_ids,
        prediction_dir,
        checkpoint=checkpoint,
        device_name=device_name,
    )

    for subject_id in case_ids:
        row = participants.loc[participants.participant_id == subject_id]
        if len(row) != 1:
            raise ValueError(f"Could not uniquely identify {subject_id}.")
        age = float(row["Gestational age"].iloc[0])
        phenotype = str(row.Pathology.iloc[0])
        image_path = find_feta_image(feta_root, subject_id)
        segmentation_path = prediction_paths[subject_id]
        tissue, aggregate, qc = measure_segmentation(
            segmentation_path,
            subject_id=subject_id,
            gestational_age_weeks=age,
        )
        tissue["segmentation_source"] = "FetalSynthSeg automatic prediction"
        tissues.append(tissue)
        aggregates.append(aggregate)
        qc_records.append({"subject_id": subject_id, **qc})
        records.append(
            {
                "subject_id": subject_id,
                "gestational_age_weeks": age,
                "feta_phenotype": phenotype,
                "image_path": str(image_path),
                "segmentation_path": str(segmentation_path),
            }
        )

    tissue_frame = pd.concat(tissues, ignore_index=True)
    aggregate_frame = pd.concat(aggregates, ignore_index=True)
    if reference == "feta-neurotypical":
        case_volumes = pd.concat(
            [
                aggregate_frame.loc[
                    aggregate_frame.region.isin(("total_brain", "intracranial_volume"))
                ],
                tissue_frame,
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        case_volumes = aggregate_frame
    scores = score_against_curves(case_volumes, curves, definition_guard=definition_guard)
    summaries = []
    for record in records:
        case_scores = scores.loc[scores.subject_id == record["subject_id"]]
        screen, detail = _case_status(
            case_scores,
            score_regions,
            interval_description=interval_description,
        )
        record.update(
            {
                "volume_screen": screen,
                "reference_result_detail": detail,
                "reference_interval_count": len(score_regions),
            }
        )
        summaries.append(record)
        save_case_report(
            record["image_path"],
            record["segmentation_path"],
            curves,
            case_scores,
            cards / f"{record['subject_id']}_case_report.png",
            subject_id=str(record["subject_id"]),
            gestational_age_weeks=float(record["gestational_age_weeks"]),
            segmentation_source=f"FetalSynthSeg automatic prediction • {record['feta_phenotype']}",
            regions=score_regions,
            dpi=300,
        )

    summary = pd.DataFrame(summaries)
    paths = {
        "summary": output_dir / "case_summary.csv",
        "tissues": output_dir / "feta_tissue_volumes.csv",
        "scores": output_dir / "reference_scores.csv",
        "qc": output_dir / "segmentation_qc.json",
        "overview": output_dir / "ten_case_overview.png",
        "growth_chart": output_dir / "ten_case_growth_chart.png",
        "curves": output_dir / "reference_curves.csv",
        "metadata": output_dir / "reference_metadata.json",
    }
    summary.to_csv(paths["summary"], index=False)
    tissue_frame.to_csv(paths["tissues"], index=False)
    scores.to_csv(paths["scores"], index=False)
    curves.to_csv(paths["curves"], index=False)
    paths["qc"].write_text(json.dumps(qc_records, indent=2) + "\n")
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    _save_overview(records, paths["overview"])
    save_growth_chart(
        curves,
        paths["growth_chart"],
        observations=scores.loc[scores.region.isin(score_regions)],
        regions=score_regions,
        title="Ten real FeTA cases on protocol-matched volume references",
        subtitle=(
            f"{metadata['subjects']} QC-passing neurotypical FeTA controls • "
            f"degree-{feta_degree} log-volume model • "
            "small in-sample teaching reference, not a clinical norm"
            if reference == "feta-neurotypical"
            else "Ren 2022 literature reference • four definition-aligned screens"
        ),
        dpi=240,
    )
    return paths
