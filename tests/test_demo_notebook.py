from __future__ import annotations

import json
from pathlib import Path


def test_demo_notebook_defers_optional_segmentation_import_and_shows_all_regions():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "Radiology_Meeting_Demo.ipynb").read_text())
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert "from fetal_brain_growth.segmentation import segment_image" not in code_cells[0]
    segmentation_cell = next(cell for cell in code_cells if "if not PREDICTED_PATH.exists()" in cell)
    assert "    from fetal_brain_growth.segmentation import segment_image" in segmentation_cell
    assert sum("regions=tuple(REFERENCE_GROUPS)" in cell for cell in code_cells) == 2
