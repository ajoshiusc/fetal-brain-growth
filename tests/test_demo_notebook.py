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


def test_meeting_notebooks_explain_quantile_provenance_and_limitations():
    root = Path(__file__).resolve().parents[1]
    demo = json.loads((root / "notebooks" / "Radiology_Meeting_Demo.ipynb").read_text())
    feta = json.loads((root / "notebooks" / "FeTA_10_Case_Meeting.ipynb").read_text())
    demo_markdown = "\n".join(
        "".join(cell["source"]) for cell in demo["cells"] if cell["cell_type"] == "markdown"
    )
    feta_markdown = "\n".join(
        "".join(cell["source"]) for cell in feta["cells"] if cell["cell_type"] == "markdown"
    )

    assert "reconstructed centiles" in demo_markdown
    assert "Q_q(t) = max(0" in demo_markdown
    assert "bounded to P3–P97" in demo_markdown
    assert "public case on FeTA-generated matched quantiles" in demo_markdown
    assert "no FeTA subject image" in demo_markdown
    assert "protocol-matched teaching reference" in feta_markdown
    assert "Q_q(GA) = exp" in feta_markdown
    assert "30 controls" in feta_markdown
    assert "same ten cases on Ren 2022 reconstructed quantiles" in feta_markdown

    feta_code = "\n".join(
        "".join(cell["source"]) for cell in feta["cells"] if cell["cell_type"] == "code"
    )
    assert "ten_case_growth_chart_ren2022.png" in feta_code
    assert "ren_scores = score_against_curves" in feta_code

    demo_code = "\n".join(
        "".join(cell["source"]) for cell in demo["cells"] if cell["cell_type"] == "code"
    )
    assert "growth_chart_feta_neurotypical.png" in demo_code
    assert "definition_guard=False" in demo_code
    assert "FeTA quantile plot skipped" in demo_code
