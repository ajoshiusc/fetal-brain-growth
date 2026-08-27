#!/usr/bin/env python3
"""Build the checked-in meeting notebook without embedding patient data."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Fetal brain segmentation and growth-chart demonstration\n\n"
            "Public-safe synthetic demonstration. Replace the two NIfTI paths and GA with a reviewed case. "
            "Outputs are research aids, not diagnoses."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "from fetal_brain_growth.synthetic import make_synthetic_phantom\n"
            "from fetal_brain_growth.radiology import save_radiology_figure\n"
            "from fetal_brain_growth.volumetry import measure_segmentation\n"
            "from fetal_brain_growth.references import build_table_curves, score_against_curves\n"
            "from fetal_brain_growth.charts import save_growth_chart"
        ),
        nbf.v4.new_code_cell(
            "out = Path('demo_outputs'); out.mkdir(exist_ok=True)\n"
            "image, segmentation = make_synthetic_phantom(out/'synthetic_svr.nii.gz', out/'synthetic_seg.nii.gz')\n"
            "figure = save_radiology_figure(image, segmentation, out/'segmentation_qc.png', "
            "subject_id='Synthetic demonstration', gestational_age_weeks=30.0, fill_alpha=0.08)\n"
            "display(Image(filename=str(figure), width=1200))"
        ),
        nbf.v4.new_markdown_cell(
            "## Affine-aware volumes\n\nVoxel volume is computed from the determinant of the NIfTI affine. "
            "The native seven tissues and definition-aware Ren aggregates are reported separately."
        ),
        nbf.v4.new_code_cell(
            "tissues, aggregates, qc = measure_segmentation(segmentation, subject_id='demo', gestational_age_weeks=30.0)\n"
            "display(tissues[['region','volume_ml']].round(2))\n"
            "display(aggregates[['region','volume_ml']].round(2))\n"
            "qc"
        ),
        nbf.v4.new_markdown_cell(
            "## Multi-quantile chart\n\nThe bundled Ren table mode reconstructs centiles from weekly mean/SD under an "
            "explicit Normal approximation. It is not a FetalSynthSeg-native clinical norm."
        ),
        nbf.v4.new_code_cell(
            "curves, metadata = build_table_curves(method='interpolate')\n"
            "scores = score_against_curves(aggregates, curves)\n"
            "chart = save_growth_chart(curves, out/'growth_chart.png', observations=scores, "
            "regions=['total_brain','intracranial_volume','external_csf','cerebellum'], "
            "subtitle='Ren 2022 summary table • reconstructed Normal-approximation centiles')\n"
            "display(Image(filename=str(chart), width=1200))\n"
            "display(scores[['region','volume_ml','estimated_percentile_bounded','status','interpretation_note']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Model choices\n\n"
            "- `build_table_curves(method='interpolate')`: reproduces weekly means; default.\n"
            "- `method='quadratic'`, `'cubic'`, or `'auto'`: smooths mean and log-SD; auto uses leave-one-week-out error.\n"
            "- `fit_local_quantile_reference(..., method='spline')`: preferred for a reviewed, protocol-matched control cohort.\n"
            "- `published_polynomial_curve('jarvis2016_total_brain')`: published mean equation only; no invented bands."
        ),
    ]
    nbf.write(notebook, root / "notebooks" / "Radiology_Meeting_Demo.ipynb")


if __name__ == "__main__":
    main()
