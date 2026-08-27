"""Affine-aware FeTA/FetalSynthSeg volumetry and lightweight QC."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .labels import FETA_LABELS, REFERENCE_GROUPS, REFERENCE_MEASURES


def integer_labels(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Segmentation must be 3-D; got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("Segmentation contains NaN or infinite values.")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-3):
        raise ValueError("Segmentation contains non-integer values.")
    labels = rounded.astype(np.int16, copy=False)
    unexpected = sorted(set(np.unique(labels).tolist()) - set(FETA_LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels {unexpected}; expected FeTA labels 0-7.")
    return labels


def segmentation_qc(labels: np.ndarray) -> dict[str, object]:
    brain = labels > 0
    warnings: list[str] = []
    if not brain.any():
        return {
            "foreground_voxels": 0,
            "largest_component_fraction": 0.0,
            "touches_image_boundary": False,
            "missing_labels": list(FETA_LABELS.values())[1:],
            "warnings": ["Segmentation is empty."],
        }
    components, count = ndimage.label(brain)
    sizes = np.bincount(components.ravel())[1:]
    largest_fraction = float(sizes.max() / sizes.sum())
    touches_boundary = bool(
        brain[0].any()
        or brain[-1].any()
        or brain[:, 0].any()
        or brain[:, -1].any()
        or brain[:, :, 0].any()
        or brain[:, :, -1].any()
    )
    missing = [name for value, name in FETA_LABELS.items() if value and not (labels == value).any()]
    if largest_fraction < 0.98:
        warnings.append("Less than 98% of labeled voxels are in the largest component.")
    if touches_boundary:
        warnings.append("Segmentation touches an image boundary; inspect for truncation.")
    if missing:
        warnings.append("Missing expected labels: " + ", ".join(missing) + ".")
    return {
        "foreground_voxels": int(brain.sum()),
        "connected_components": int(count),
        "largest_component_fraction": largest_fraction,
        "touches_image_boundary": touches_boundary,
        "missing_labels": missing,
        "warnings": warnings,
    }


def measure_segmentation(
    segmentation: str | Path | nib.spatialimages.SpatialImage,
    *,
    subject_id: str = "subject",
    gestational_age_weeks: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return native tissue volumes, compatible aggregates, and QC metadata."""

    image = nib.load(str(segmentation)) if isinstance(segmentation, (str, Path)) else segmentation
    labels = integer_labels(np.asanyarray(image.dataobj))
    voxel_volume_ml = float(abs(np.linalg.det(image.affine[:3, :3])) / 1000.0)
    if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0:
        raise ValueError("Segmentation affine does not define a positive voxel volume.")
    counts = np.bincount(labels.ravel(), minlength=8)
    base = {
        "subject_id": subject_id,
        "gestational_age_weeks": gestational_age_weeks,
        "voxel_volume_ml": voxel_volume_ml,
    }
    tissues = pd.DataFrame(
        [
            {
                **base,
                "label": value,
                "region": name,
                "voxel_count": int(counts[value]),
                "volume_ml": float(counts[value] * voxel_volume_ml),
            }
            for value, name in FETA_LABELS.items()
            if value
        ]
    )
    aggregates = pd.DataFrame(
        [
            {
                **base,
                "region": region,
                "source_measure": REFERENCE_MEASURES[region],
                "labels": "+".join(map(str, group)),
                "voxel_count": int(sum(counts[value] for value in group)),
                "volume_ml": float(sum(counts[value] for value in group) * voxel_volume_ml),
            }
            for region, group in REFERENCE_GROUPS.items()
        ]
    )
    qc = segmentation_qc(labels)
    qc.update(
        {
            "subject_id": subject_id,
            "shape": list(labels.shape),
            "voxel_volume_ml": voxel_volume_ml,
            "orientation": "".join(nib.aff2axcodes(image.affine)),
        }
    )
    return tissues, aggregates, qc


def analyze_manifest(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Measure a cross-sectional manifest without imposing a reference model."""

    manifest_path = Path(manifest_path).resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "gestational_age_weeks", "segmentation_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["subject_id"].duplicated().any():
        raise ValueError("subject_id values must be unique.")
    tissues, aggregates, qc_records = [], [], []
    for row in manifest.itertuples(index=False):
        segmentation = Path(str(row.segmentation_path))
        if not segmentation.is_absolute():
            segmentation = (manifest_path.parent / segmentation).resolve()
        tissue, aggregate, qc = measure_segmentation(
            segmentation,
            subject_id=str(row.subject_id),
            gestational_age_weeks=float(row.gestational_age_weeks),
        )
        qc["segmentation_path"] = str(segmentation)
        tissues.append(tissue)
        aggregates.append(aggregate)
        qc_records.append(qc)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tissues": output_dir / "tissue_volumes.csv",
        "aggregates": output_dir / "reference_compatible_volumes.csv",
        "qc": output_dir / "segmentation_qc.json",
    }
    pd.concat(tissues, ignore_index=True).to_csv(paths["tissues"], index=False)
    pd.concat(aggregates, ignore_index=True).to_csv(paths["aggregates"], index=False)
    paths["qc"].write_text(json.dumps(qc_records, indent=2) + "\n")
    return paths
