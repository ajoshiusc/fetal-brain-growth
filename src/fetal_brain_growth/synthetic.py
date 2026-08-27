"""Deterministic fetal-label phantom used only by automated tests."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def make_synthetic_phantom(
    image_path: str | Path,
    segmentation_path: str | Path,
    *,
    shape: tuple[int, int, int] = (144, 160, 136),
    spacing_mm: float = 0.8,
    seed: int = 2026,
) -> tuple[Path, Path]:
    """Create a public-safe, MRI-like RAS phantom containing FeTA labels 1–7."""

    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1)[:, None, None, None] / 2
    x, y, z = (grid - center) * spacing_mm
    outer = (x / 49) ** 2 + (y / 57) ** 2 + (z / 43) ** 2 <= 1
    tissue = (x / 45) ** 2 + (y / 53) ** 2 + (z / 39) ** 2 <= 1
    inner = (x / 38) ** 2 + (y / 46) ** 2 + (z / 33) ** 2 <= 1

    labels = np.zeros(shape, dtype=np.int16)
    labels[outer & ~tissue] = 1
    labels[tissue & ~inner] = 2
    labels[inner] = 3

    ventricles = (
        (((x - 8) / 4.5) ** 2 + ((y + 2) / 15) ** 2 + ((z - 3) / 5.5) ** 2 <= 1)
        | (((x + 8) / 4.5) ** 2 + ((y + 2) / 15) ** 2 + ((z - 3) / 5.5) ** 2 <= 1)
    )
    cerebellum = (
        (((x - 13) / 15) ** 2 + ((y + 37) / 16) ** 2 + ((z + 25) / 13) ** 2 <= 1)
        | (((x + 13) / 15) ** 2 + ((y + 37) / 16) ** 2 + ((z + 25) / 13) ** 2 <= 1)
    ) & outer
    deep_gray = (
        (((x - 12) / 7) ** 2 + ((y + 2) / 10) ** 2 + ((z + 1) / 8) ** 2 <= 1)
        | (((x + 12) / 7) ** 2 + ((y + 2) / 10) ** 2 + ((z + 1) / 8) ** 2 <= 1)
    )
    brainstem = ((x / 7) ** 2 + ((y + 15) / 8) ** 2 <= 1) & (z > -39) & (z < -15)
    labels[ventricles & tissue] = 4
    labels[cerebellum] = 5
    labels[deep_gray & tissue] = 6
    labels[brainstem & outer] = 7

    # Approximate fetal T2 tissue contrast with smooth bias and Rician-like noise.
    levels = np.asarray([0, 78, 92, 128, 170, 112, 103, 96], dtype=np.float32)
    image = levels[labels]
    bias = 1 + 0.12 * (x / 50) - 0.08 * (z / 45)
    image *= bias
    image = gaussian_filter(image, sigma=0.85)
    rng = np.random.default_rng(seed)
    real = image + rng.normal(0, 3.0, shape)
    imaginary = rng.normal(0, 3.0, shape)
    image = np.sqrt(real**2 + imaginary**2).astype(np.float32)
    image[~outer] = 0

    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = -0.5 * spacing_mm * (np.asarray(shape) - 1)
    image_path = Path(image_path)
    segmentation_path = Path(segmentation_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image, affine), str(image_path))
    nib.save(nib.Nifti1Image(labels, affine), str(segmentation_path))
    return image_path, segmentation_path
