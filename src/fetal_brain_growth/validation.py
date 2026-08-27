"""Validation helpers for labeled fetal MRI examples."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd

from .labels import FETA_LABELS, LABEL_TITLES
from .volumetry import integer_labels


def tissue_dice(
    segmentation: str | Path | nib.spatialimages.SpatialImage,
    reference: str | Path | nib.spatialimages.SpatialImage,
) -> pd.DataFrame:
    """Calculate per-label Dice after nearest-neighbor geometry alignment."""

    predicted = nib.load(str(segmentation)) if isinstance(segmentation, (str, Path)) else segmentation
    manual = nib.load(str(reference)) if isinstance(reference, (str, Path)) else reference
    if predicted.shape != manual.shape or not np.allclose(predicted.affine, manual.affine):
        predicted = resample_from_to(predicted, (manual.shape, manual.affine), order=0, mode="constant", cval=0)
    predicted_labels = integer_labels(predicted.get_fdata(dtype=np.float32))
    manual_labels = integer_labels(manual.get_fdata(dtype=np.float32))
    rows = []
    for label in sorted(FETA_LABELS):
        if label == 0:
            continue
        predicted_mask = predicted_labels == label
        manual_mask = manual_labels == label
        denominator = int(predicted_mask.sum() + manual_mask.sum())
        dice = 1.0 if denominator == 0 else 2.0 * np.count_nonzero(predicted_mask & manual_mask) / denominator
        rows.append(
            {
                "label": label,
                "region": FETA_LABELS[label],
                "tissue": LABEL_TITLES[label],
                "dice": float(dice),
                "predicted_voxels": int(predicted_mask.sum()),
                "manual_voxels": int(manual_mask.sum()),
            }
        )
    return pd.DataFrame(rows)
