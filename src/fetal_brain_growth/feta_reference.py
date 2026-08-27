"""Protocol-matched teaching references from neurotypical FeTA segmentations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .labels import FETA_MATCHED_REFERENCE_REGIONS
from .references import DEFAULT_QUANTILES, parse_quantiles, quantile_column
from .volumetry import measure_segmentation


DEFAULT_LOCAL_FETA_ROOT = Path("/deneb_disk/feta_2022/feta_2.2")
FETA_DATA_DOI = "10.1038/s41597-021-00946-3"


def resolve_feta_root(path: str | Path | None = None) -> Path:
    """Resolve an explicit path, ``FETA_ROOT``, or the known local FeTA path."""

    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if environment_path := os.environ.get("FETA_ROOT"):
        candidates.append(Path(environment_path))
    candidates.append(DEFAULT_LOCAL_FETA_ROOT)
    for candidate in candidates:
        if (candidate / "participants.tsv").is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not locate FeTA participants.tsv. Pass --feta-root or set FETA_ROOT. Searched: {searched}"
    )


def load_feta_participants(feta_root: str | Path) -> pd.DataFrame:
    root = Path(feta_root)
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    required = {"participant_id", "Pathology", "Gestational age"}
    if missing := required - set(participants.columns):
        raise ValueError(f"participants.tsv is missing {sorted(missing)}.")
    participants = participants.copy()
    participants["Gestational age"] = pd.to_numeric(participants["Gestational age"], errors="raise")
    if participants.participant_id.duplicated().any():
        raise ValueError("participants.tsv contains duplicate participant IDs.")
    return participants


def find_feta_case_files(root: str | Path, subject_id: str) -> tuple[Path, Path]:
    directory = Path(root) / subject_id / "anat"
    images = sorted(directory.glob("*_T2w.nii.gz"))
    labels = sorted(directory.glob("*_dseg.nii.gz"))
    if len(images) != 1 or len(labels) != 1:
        raise FileNotFoundError(f"Expected one T2w and one dseg for {subject_id} in {directory}.")
    return images[0], labels[0]


def collect_neurotypical_feta_volumes(
    feta_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    """Measure all FeTA cases explicitly labeled ``Neurotypical``."""

    root = resolve_feta_root(feta_root)
    participants = load_feta_participants(root)
    normal = participants.loc[
        participants.Pathology.astype(str).str.casefold() == "neurotypical"
    ].sort_values(["Gestational age", "participant_id"])
    if normal.empty:
        raise ValueError("No participants are labeled Neurotypical.")
    tissues, aggregates, qc_records = [], [], []
    for _, row in normal.iterrows():
        subject_id = str(row["participant_id"])
        age = float(row["Gestational age"])
        _, segmentation_path = find_feta_case_files(root, subject_id)
        tissue, aggregate, qc = measure_segmentation(
            segmentation_path,
            subject_id=subject_id,
            gestational_age_weeks=age,
        )
        included = not bool(qc.get("warnings"))
        qc_records.append({"subject_id": subject_id, "included_in_reference": included, **qc})
        if included:
            tissues.append(tissue)
            aggregates.append(
                aggregate.loc[aggregate.region.isin(("total_brain", "intracranial_volume"))]
            )
    tissue_frame = pd.concat(tissues, ignore_index=True)
    aggregate_frame = pd.concat(aggregates, ignore_index=True)
    matched = pd.concat([aggregate_frame, tissue_frame], ignore_index=True, sort=False)
    return tissue_frame, matched, qc_records, normal.reset_index(drop=True)


def _validate_matched_cohort(volumes: pd.DataFrame, minimum_subjects: int) -> pd.DataFrame:
    required = {"subject_id", "gestational_age_weeks", "region", "volume_ml"}
    if missing := required - set(volumes.columns):
        raise ValueError(f"FeTA reference data are missing columns: {sorted(missing)}")
    data = volumes.loc[volumes.region.isin(FETA_MATCHED_REFERENCE_REGIONS)].copy()
    if data.duplicated(["subject_id", "region"]).any():
        raise ValueError("Use one observation per subject and region.")
    if not np.isfinite(data[["gestational_age_weeks", "volume_ml"]]).all().all():
        raise ValueError("Age and volume must be finite.")
    if (data.volume_ml <= 0).any():
        raise ValueError("All fitted volumes must be positive.")
    expected_regions = set(FETA_MATCHED_REFERENCE_REGIONS)
    if set(data.region) != expected_regions:
        raise ValueError(f"Expected matched regions {sorted(expected_regions)}.")
    for region, group in data.groupby("region"):
        if group.subject_id.nunique() < minimum_subjects:
            raise ValueError(f"{region}: fewer than {minimum_subjects} neurotypical subjects.")
        if group.gestational_age_weeks.max() - group.gestational_age_weeks.min() < 8:
            raise ValueError(f"{region}: gestational-age span is under eight weeks.")
    return data


def _leave_one_out_rmse(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    residuals = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        coefficients = np.polyfit(x[keep], y[keep], degree)
        residuals.append(float(np.polyval(coefficients, x[index]) - y[index]))
    return float(np.sqrt(np.mean(np.square(residuals))))


def fit_feta_matched_reference(
    volumes: pd.DataFrame,
    *,
    degree: int = 2,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    grid_step_weeks: float = 0.05,
    minimum_subjects: int = 25,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit small-cohort log-volume curves with a constant Gaussian residual SD.

    FeTA 2.2 has only 31 cases labeled neurotypical. A quadratic location model and one
    residual scale per region are intentionally used instead of unstable direct
    P3/P97 regression. Cases failing technical segmentation QC are excluded,
    but no volume-based outlier exclusion is performed.
    """

    if degree not in {2, 3}:
        raise ValueError("degree must be 2 or 3.")
    quantiles = parse_quantiles(quantiles)
    data = _validate_matched_cohort(volumes, minimum_subjects)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    for region in FETA_MATCHED_REFERENCE_REGIONS:
        group = data.loc[data.region == region].sort_values("gestational_age_weeks")
        age = group.gestational_age_weeks.to_numpy(dtype=float)
        log_volume = np.log(group.volume_ml.to_numpy(dtype=float))
        age_center = float(age.mean())
        centered_age = age - age_center
        coefficients = np.polyfit(centered_age, log_volume, degree)
        fitted = np.polyval(coefficients, centered_age)
        residual = log_volume - fitted
        degrees_of_freedom = len(age) - (degree + 1)
        residual_sd = float(np.sqrt(np.sum(np.square(residual)) / degrees_of_freedom))
        grid = np.arange(age.min(), age.max() + grid_step_weeks / 2, grid_step_weeks)
        location = np.polyval(coefficients, grid - age_center)
        for index, gestational_age in enumerate(grid):
            row: dict[str, object] = {
                "region": region,
                "gestational_age_weeks": float(gestational_age),
                "median_ml": float(np.exp(location[index])),
                "log_residual_sd": residual_sd,
            }
            for quantile in quantiles:
                row[quantile_column(quantile)] = float(
                    np.exp(location[index] + norm.ppf(quantile) * residual_sd)
                )
            rows.append(row)
        total_variation = float(np.sum(np.square(log_volume - log_volume.mean())))
        diagnostics[region] = {
            "subjects": int(group.subject_id.nunique()),
            "age_min_weeks": float(age.min()),
            "age_max_weeks": float(age.max()),
            "age_center_weeks": age_center,
            "log_volume_coefficients_descending_centered_age": coefficients.tolist(),
            "log_residual_sd": residual_sd,
            "log_volume_r_squared": 1.0 - float(np.sum(np.square(residual))) / total_variation,
            "leave_one_out_rmse_log_volume": _leave_one_out_rmse(centered_age, log_volume, degree),
        }
    metadata: dict[str, object] = {
        "source": "QC-passing FeTA 2.2 expert segmentations labeled Neurotypical",
        "source_doi": FETA_DATA_DOI,
        "phenotype_filter": "Pathology == Neurotypical (case-insensitive exact match)",
        "subjects": int(data.subject_id.nunique()),
        "subject_ids": sorted(data.subject_id.unique().tolist()),
        "age_min_weeks": float(data.gestational_age_weeks.min()),
        "age_max_weeks": float(data.gestational_age_weeks.max()),
        "regions": list(FETA_MATCHED_REFERENCE_REGIONS),
        "degree": degree,
        "response": "log(volume_ml)",
        "scale_model": "constant residual SD in log-volume space",
        "quantiles": list(quantiles),
        "quantile_model": "exp(fitted log-volume + Normal quantile * residual SD)",
        "outlier_policy": "No volume-based exclusions; technical segmentation-QC failures are excluded.",
        "diagnostics": diagnostics,
        "license_note": "FeTA data and derived local tables are for research and education under FeTA terms.",
        "warning": (
            f"Small, in-sample, cross-sectional teaching reference ({data.subject_id.nunique()} controls), "
            "not a validated clinical norm. Do not extrapolate beyond its observed age range."
        ),
    }
    return pd.DataFrame(rows), metadata


def build_feta_matched_reference(
    feta_root: str | Path | None = None,
    *,
    degree: int = 2,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    root = resolve_feta_root(feta_root)
    tissues, matched, qc_records, participants = collect_neurotypical_feta_volumes(root)
    curves, metadata = fit_feta_matched_reference(
        matched,
        degree=degree,
        quantiles=quantiles,
        minimum_subjects=25,
    )
    metadata["feta_root_name"] = root.name
    metadata["normal_participant_rows"] = int(len(participants))
    metadata["segmentation_qc_excluded_cases"] = [
        record["subject_id"] for record in qc_records if record.get("warnings")
    ]
    return curves, metadata, tissues, matched, qc_records


def save_feta_matched_reference(
    output_dir: str | Path,
    curves: pd.DataFrame,
    metadata: dict[str, object],
    tissues: pd.DataFrame,
    matched: pd.DataFrame,
    qc_records: list[dict[str, object]],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "curves": output_dir / "feta_neurotypical_reference_curves.csv",
        "metadata": output_dir / "feta_neurotypical_reference_metadata.json",
        "tissues": output_dir / "feta_neurotypical_tissue_volumes.csv",
        "matched_volumes": output_dir / "feta_neurotypical_matched_volumes.csv",
        "qc": output_dir / "feta_neurotypical_segmentation_qc.json",
    }
    curves.to_csv(paths["curves"], index=False)
    tissues.to_csv(paths["tissues"], index=False)
    matched.to_csv(paths["matched_volumes"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    paths["qc"].write_text(json.dumps(qc_records, indent=2) + "\n")
    return paths
