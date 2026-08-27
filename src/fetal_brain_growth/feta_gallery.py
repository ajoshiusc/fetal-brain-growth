"""Build a local ten-case FeTA image/segmentation/growth-chart teaching set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .case_report import save_case_report
from .charts import save_growth_chart
from .labels import FETA_LABELS, LABEL_COLORS
from .radiology import _best_slice, _crop_bounds, _display_slice, load_aligned_canonical
from .references import build_table_curves, score_against_curves
from .volumetry import measure_segmentation


DEFAULT_CASE_IDS = (
    "sub-036", "sub-027", "sub-034", "sub-051", "sub-061",
    "sub-005", "sub-014", "sub-001", "sub-019", "sub-050",
)
SCOREABLE_REGIONS = ("total_brain", "intracranial_volume", "external_csf", "cerebellum")
PHENOTYPE_COLORS = {"Neurotypical": "#1976A3", "Pathological": "#7B3F91"}


def _find_case_files(root: Path, subject_id: str) -> tuple[Path, Path]:
    directory = root / subject_id / "anat"
    images = sorted(directory.glob("*_T2w.nii.gz"))
    labels = sorted(directory.glob("*_dseg.nii.gz"))
    if len(images) != 1 or len(labels) != 1:
        raise FileNotFoundError(f"Expected one T2w and one dseg for {subject_id} in {directory}.")
    return images[0], labels[0]


def _case_status(scores: pd.DataFrame) -> tuple[str, str]:
    scores = scores.loc[scores.region.isin(SCOREABLE_REGIONS)]
    flagged = scores.loc[scores.status.isin({"low_reference_flag", "high_reference_flag"})]
    if flagged.empty:
        return "within_all_4_reference_intervals", "Within all four definition-aligned intervals"
    details = []
    for row in flagged.itertuples(index=False):
        direction = "low" if row.status == "low_reference_flag" else "high"
        details.append(f"{row.region}: {direction}, bounded P{row.estimated_percentile_bounded:.0f}")
    return "one_or_more_reference_flags", "; ".join(details)


def _save_overview(records: list[dict[str, object]], output_path: Path, *, dpi: int = 240) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.0), facecolor="white")
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
        ax.set_title(
            f"{record['subject_id']} • {record['gestational_age_weeks']:.1f} w\n"
            f"{record['feta_phenotype']} • {'reference flag(s)' if flagged else 'within 4 intervals'}",
            fontsize=9.5,
            color=PHENOTYPE_COLORS.get(str(record["feta_phenotype"]), "#17233C"),
            weight="bold",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(3 if flagged else 1.2)
            spine.set_edgecolor("#C43D3D" if flagged else "#8190A5")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ten real FeTA fetal MRIs with expert segmentation outlines", fontsize=17, weight="bold", color="#17233C")
    fig.text(
        0.5,
        0.015,
        "Title color is the FeTA dataset phenotype; red frame means ≥1 research reference flag. Neither is a diagnosis generated here.",
        ha="center",
        fontsize=9,
        color="#5D6877",
    )
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_feta_gallery(
    feta_root: str | Path,
    output_dir: str | Path,
    *,
    case_ids: Iterable[str] = DEFAULT_CASE_IDS,
) -> dict[str, Path]:
    """Create local teaching figures from FeTA expert annotations.

    Raw FeTA data and generated case figures remain subject to the dataset's
    research/education terms and should not be committed automatically.
    """

    feta_root = Path(feta_root).resolve()
    output_dir = Path(output_dir).resolve()
    case_ids = tuple(case_ids)
    if len(case_ids) != 10 or len(set(case_ids)) != 10:
        raise ValueError("Provide exactly ten unique case IDs.")
    participants = pd.read_csv(feta_root / "participants.tsv", sep="\t")
    required = {"participant_id", "Pathology", "Gestational age"}
    if missing := required - set(participants.columns):
        raise ValueError(f"participants.tsv is missing {sorted(missing)}.")
    curves, metadata = build_table_curves(method="interpolate")
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = output_dir / "case_cards"
    cards.mkdir(exist_ok=True)
    tissues, aggregates, qc_records, records = [], [], [], []

    for subject_id in case_ids:
        row = participants.loc[participants.participant_id == subject_id]
        if len(row) != 1:
            raise ValueError(f"Could not uniquely identify {subject_id}.")
        age = float(row["Gestational age"].iloc[0])
        phenotype = str(row.Pathology.iloc[0])
        image_path, segmentation_path = _find_case_files(feta_root, subject_id)
        tissue, aggregate, qc = measure_segmentation(
            segmentation_path,
            subject_id=subject_id,
            gestational_age_weeks=age,
        )
        tissue["segmentation_source"] = "FeTA expert annotation"
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
    scores = score_against_curves(aggregate_frame, curves)
    summaries = []
    for record in records:
        case_scores = scores.loc[scores.subject_id == record["subject_id"]]
        screen, detail = _case_status(case_scores)
        record.update({"volume_screen": screen, "reference_result_detail": detail})
        summaries.append(record)
        save_case_report(
            record["image_path"],
            record["segmentation_path"],
            curves,
            case_scores,
            cards / f"{record['subject_id']}_case_report.png",
            subject_id=str(record["subject_id"]),
            gestational_age_weeks=float(record["gestational_age_weeks"]),
            segmentation_source=f"FeTA expert annotation • {record['feta_phenotype']}",
            dpi=220,
        )

    summary = pd.DataFrame(summaries)
    paths = {
        "summary": output_dir / "case_summary.csv",
        "tissues": output_dir / "feta_tissue_volumes.csv",
        "scores": output_dir / "reference_scores.csv",
        "qc": output_dir / "segmentation_qc.json",
        "overview": output_dir / "ten_case_overview.png",
        "growth_chart": output_dir / "ten_case_growth_chart.png",
        "metadata": output_dir / "reference_metadata.json",
    }
    summary.to_csv(paths["summary"], index=False)
    tissue_frame.to_csv(paths["tissues"], index=False)
    scores.to_csv(paths["scores"], index=False)
    paths["qc"].write_text(json.dumps(qc_records, indent=2) + "\n")
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    _save_overview(records, paths["overview"])
    save_growth_chart(
        curves,
        paths["growth_chart"],
        observations=scores.loc[scores.region.isin(SCOREABLE_REGIONS)],
        regions=SCOREABLE_REGIONS,
        title="Ten real FeTA cases on fetal brain volume references",
        subtitle="Expert FeTA segmentations • selected teaching set • reference flags are not diagnoses",
        dpi=240,
    )
    return paths
