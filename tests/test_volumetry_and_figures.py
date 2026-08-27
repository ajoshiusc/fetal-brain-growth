from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fetal_brain_growth.charts import save_growth_chart
from fetal_brain_growth.radiology import load_aligned_canonical, save_radiology_figure
from fetal_brain_growth.references import build_table_curves
from fetal_brain_growth.synthetic import make_synthetic_phantom
from fetal_brain_growth.volumetry import measure_segmentation


def test_affine_aware_volumetry(tmp_path: Path):
    data = np.zeros((8, 9, 10), dtype=np.int16)
    data[1:3, 1:4, 1:5] = 5
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    tissues, _, qc = measure_segmentation(path)
    cerebellum = tissues.loc[tissues.region == "cerebellum"].iloc[0]
    assert cerebellum.voxel_count == 24
    assert np.isclose(cerebellum.volume_ml, 24 * 6 / 1000)
    assert np.isclose(qc["voxel_volume_ml"], 0.006)


def test_synthetic_radiology_figure_is_ras_and_high_resolution(tmp_path: Path):
    image = tmp_path / "image.nii.gz"
    segmentation = tmp_path / "segmentation.nii.gz"
    make_synthetic_phantom(image, segmentation, shape=(96, 104, 92))
    _, labels, orientation = load_aligned_canonical(image, segmentation)
    assert orientation == ("R", "A", "S")
    assert set(np.unique(labels)) == set(range(8))
    output = save_radiology_figure(image, segmentation, tmp_path / "qc.png", dpi=150)
    loaded = nib.load(segmentation)
    assert loaded.shape == labels.shape
    assert output.stat().st_size > 20_000


def test_growth_chart_can_write_vector_output(tmp_path: Path):
    curves, _ = build_table_curves(grid_step_weeks=0.5)
    output = save_growth_chart(curves, tmp_path / "chart.svg", regions=["total_brain", "cerebellum"])
    assert output.read_text().lstrip().startswith("<?xml")
