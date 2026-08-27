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
            "from fetal_brain_growth.charts import save_growth_chart\n"
            "from fetal_brain_growth.feta_gallery import build_feta_gallery\n"
            "from fetal_brain_growth.feta_reference import resolve_feta_root\n"
            "from fetal_brain_growth.labels import REFERENCE_GROUPS\n"
            "from fetal_brain_growth.references import build_table_curves, score_against_curves"
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
        nbf.v4.new_markdown_cell(
            "## The same ten cases on Ren 2022 reconstructed quantiles\n\n"
            "This second chart uses `mean(GA) + Φ⁻¹(q) × SD(GA)` reconstructed from the Ren weekly summary "
            "table. It is included to show how conclusions depend on the reference population and anatomical "
            "definitions. Green/red points are the four definition-aligned screens (total brain, intracranial "
            "volume, external CSF, and cerebellum); orange points are comparison-only. Do not interpret "
            "differences between the two charts as biological change in an individual fetus."
        ),
        nbf.v4.new_code_cell(
            "matched_scores = pd.read_csv(paths['scores'])\n"
            "base_volumes = matched_scores[['subject_id','gestational_age_weeks','region','volume_ml']].copy()\n"
            "subcortical = (\n"
            "    base_volumes[base_volumes.region.isin(['white_matter','deep_gray_matter'])]\n"
            "    .groupby(['subject_id','gestational_age_weeks'], as_index=False).volume_ml.sum()\n"
            ")\n"
            "subcortical['region'] = 'subcortical_brain_tissue'\n"
            "ren_volumes = pd.concat([\n"
            "    base_volumes[base_volumes.region.isin(REFERENCE_GROUPS) & (base_volumes.region != 'subcortical_brain_tissue')],\n"
            "    subcortical,\n"
            "], ignore_index=True)\n"
            "ren_curves, ren_metadata = build_table_curves(method='interpolate')\n"
            "ren_scores = score_against_curves(ren_volumes, ren_curves, definition_guard=True)\n"
            "ren_chart = save_growth_chart(\n"
            "    ren_curves, OUT/'ten_case_growth_chart_ren2022.png', observations=ren_scores,\n"
            "    regions=tuple(REFERENCE_GROUPS),\n"
            "    title='The same ten FeTA cases on Ren 2022 reconstructed quantiles',\n"
            "    subtitle='Normal approximation from weekly mean/SD • green/red = definition-aligned; orange = comparison-only',\n"
            "    dpi=240,\n"
            ")\n"
            "display(Image(filename=str(ren_chart), width=1500))\n"
            "display(ren_scores[['subject_id','gestational_age_weeks','region','volume_ml',"
            "'estimated_percentile_bounded','status']])"
        ),
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
            "This is a **protocol-matched teaching reference**, not a validated clinical norm and not the Ren "
            "summary-data reconstruction shown in the second chart. It is built as follows:\n\n"
            "1. Select FeTA rows whose `Pathology` field is exactly `Neurotypical` (case-insensitive).\n"
            "2. Measure the seven native expert-label volumes plus total brain and intracranial volume using "
            "`voxel count × abs(det(affine[:3,:3])) / 1000` mL.\n"
            "3. Exclude technical segmentation-QC failures, but remove no cases based on their measured volume. "
            "The mounted release contains 31 neurotypical rows; `sub-041` fails boundary QC, leaving 30 controls "
            "from 22.7–34.8 weeks.\n"
            "4. For each region fit `log(V) = β₀ + β₁(GA − mean(GA)) + β₂(GA − mean(GA))²`.\n"
            "5. Estimate one constant residual SD `s` in log-volume space and calculate "
            "`Q_q(GA) = exp(fitted_log_volume(GA) + Φ⁻¹(q) × s)` for P3/P10/P25/P50/P75/P90/P97.\n\n"
            "These are estimated population intervals under a log-Normal residual assumption, not confidence "
            "intervals for the fitted median. Case percentiles are linearly interpolated between the seven curves "
            "and bounded to P3–P97; values outside that interval receive a research flag. Cubic fitting is available "
            "with `fbg feta-reference --degree 3`, but is only a sensitivity analysis for this small cohort. Because "
            "some displayed normal cases also helped fit the curves, their positions are in-sample."
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
