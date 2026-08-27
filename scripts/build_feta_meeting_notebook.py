#!/usr/bin/env python3
"""Build the local FeTA ten-case meeting notebook from reviewable cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Ten real FeTA fetal brains on nine matched growth charts\n\n"
            "This local meeting notebook uses FeTA 2.2 SVR images and expert segmentations, not synthetic "
            "data. It fits only QC-passing cases labeled `Neurotypical`, then displays five neurotypical "
            "and five pathological examples. A reference flag is a research screen, not a diagnosis. "
            "FeTA data and derived images remain subject to FeTA access terms and are not committed."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "if not (ROOT/'src/fetal_brain_growth').is_dir():\n"
            "    raise FileNotFoundError(f'Run this notebook from the fetal-brain-growth repository; cwd={ROOT}')\n"
            "sys.path.insert(0, str(ROOT/'src'))\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "from fetal_brain_growth.feta_gallery import build_feta_gallery\n"
            "from fetal_brain_growth.feta_reference import resolve_feta_root"
        ),
        nbf.v4.new_code_cell(
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "FETA_ROOT = resolve_feta_root()  # or resolve_feta_root('/path/to/feta_2.2')\n"
            "OUT = ROOT/'meeting_outputs/feta_10_cases_matched'\n"
            "if not (OUT/'case_summary.csv').exists():\n"
            "    paths = build_feta_gallery(FETA_ROOT, OUT, reference='feta-neurotypical', feta_degree=2)\n"
            "else:\n"
            "    paths = {\n"
            "        'summary': OUT/'case_summary.csv',\n"
            "        'scores': OUT/'reference_scores.csv',\n"
            "        'overview': OUT/'ten_case_overview.png',\n"
            "        'growth_chart': OUT/'ten_case_growth_chart.png',\n"
            "        'metadata': OUT/'reference_metadata.json',\n"
            "    }\n"
            "print('FeTA root:', FETA_ROOT)\n"
            "print('Meeting outputs:', OUT)"
        ),
        nbf.v4.new_markdown_cell(
            "## Cohort selection and independent labels\n\n"
            "The gallery deliberately includes both FeTA phenotype groups. Phenotype is supplied by the "
            "dataset; the volumetric screen is calculated here. A pathological phenotype may have all "
            "volumes within range, and a reference flag alone does not establish pathology."
        ),
        nbf.v4.new_code_cell(
            "summary = pd.read_csv(paths['summary'])\n"
            "display(summary[['subject_id','gestational_age_weeks','feta_phenotype',"
            "'volume_screen','reference_result_detail']])"
        ),
        nbf.v4.new_markdown_cell("## Expert segmentations in standard orientation"),
        nbf.v4.new_code_cell("display(Image(filename=str(paths['overview']), width=1500))"),
        nbf.v4.new_markdown_cell(
            "## Nine exact-label volume charts\n\n"
            "The panels show total brain, intracranial volume, external CSF, cortical gray matter, white "
            "matter, ventricles, cerebellum, deep gray matter, and brainstem with "
            "P3/P10/P25/P50/P75/P90/P97 bands. Red points fall outside P3–P97."
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(paths['growth_chart']), width=1500))"),
        nbf.v4.new_markdown_cell("## Flagged measurements and per-case report"),
        nbf.v4.new_code_cell(
            "scores = pd.read_csv(paths['scores'])\n"
            "flagged = scores[scores.status.isin(['low_reference_flag','high_reference_flag'])]\n"
            "display(flagged[['subject_id','gestational_age_weeks','region','volume_ml',"
            "'estimated_percentile_bounded','status']])\n"
            "display(Image(filename=str(OUT/'case_cards/sub-050_case_report.png'), width=1500))"
        ),
        nbf.v4.new_markdown_cell(
            "## How the reference was generated\n\n"
            "For each measure, the default model is quadratic in centered gestational age with log-volume "
            "as the response. Quantiles are `exp(fitted_log_volume + Normal_quantile × residual_SD)`. "
            "No biological-volume outliers are removed. Cubic fitting is available with `fbg "
            "feta-reference --degree 3`, but should be treated as a sensitivity analysis for this small cohort."
        ),
        nbf.v4.new_code_cell(
            "metadata = json.loads(Path(paths['metadata']).read_text())\n"
            "print({key: metadata[key] for key in [\n"
            "    'subjects','age_min_weeks','age_max_weeks','degree','quantiles',"
            "'segmentation_qc_excluded_cases']})\n"
            "diagnostics = pd.DataFrame(metadata['diagnostics']).T.reset_index(names='region')\n"
            "display(diagnostics[['region','log_volume_r_squared','leave_one_out_rmse_log_volume',"
            "'log_residual_sd']].style.format({\n"
            "    'log_volume_r_squared': '{:.3f}',\n"
            "    'leave_one_out_rmse_log_volume': '{:.3f}',\n"
            "    'log_residual_sd': '{:.3f}',\n"
            "}))"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation limits\n\n"
            "- The FeTA control set is small, cross-sectional, and partly reused as displayed normal examples.\n"
            "- Do not extrapolate outside 22.7–34.8 weeks.\n"
            "- Ventricular, deep-gray, and brainstem fits are especially uncertain; inspect diagnostics.\n"
            "- Review the complete 3-D image and segmentation, gestational-age uncertainty, morphology, and "
            "clinical context with a fetal neuroradiologist.\n"
            "- A larger independent cohort processed with the same frozen pipeline is required for clinical validation."
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"feta-meeting-{index:02d}"
    nbf.write(notebook, root / "notebooks" / "FeTA_10_Case_Meeting.ipynb")


if __name__ == "__main__":
    main()
