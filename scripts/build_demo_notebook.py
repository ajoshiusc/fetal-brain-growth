#!/usr/bin/env python3
"""Build the real fetal MRI meeting notebook from reviewable cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Real fetal MRI segmentation and growth-chart demonstration\n\n"
            "This notebook uses the real 30-week fetal T2 atlas example distributed by FetalSynthSeg, "
            "not a synthetic phantom. The image and manual label map originate from the CC0 IMAGINE "
            "Fetal T2-weighted MRI Atlas (doi:10.7910/DVN/WE9JVR). The automatic labels are produced by "
            "the official FetalSynthSeg checkpoint. Results are research aids, not diagnoses."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n"
            "from fetal_brain_growth.case_report import save_case_report\n"
            "from fetal_brain_growth.charts import save_growth_chart\n"
            "from fetal_brain_growth.feta_reference import (\n"
            "    build_feta_matched_reference, resolve_feta_root, save_feta_matched_reference,\n"
            ")\n"
            "from fetal_brain_growth.labels import FETA_MATCHED_REFERENCE_REGIONS, REFERENCE_GROUPS\n"
            "from fetal_brain_growth.radiology import save_radiology_figure\n"
            "from fetal_brain_growth.references import build_table_curves, score_against_curves\n"
            "from fetal_brain_growth.validation import tissue_dice\n"
            "from fetal_brain_growth.volumetry import measure_segmentation"
        ),
        nbf.v4.new_code_cell(
            "START = Path.cwd().resolve()\n"
            "ROOT = next((p for p in (START, *START.parents) if (p/'pyproject.toml').exists() and (p/'notebooks').is_dir()), None)\n"
            "if ROOT is None:\n"
            "    raise RuntimeError('Could not locate the fetal-brain-growth repository. Start Jupyter from the repository or its notebooks directory.')\n"
            "EXAMPLE = ROOT/'third_party/FetalSynthSeg/data/sub-sta30/anat'\n"
            "IMAGE_PATH = EXAMPLE/'sub-sta30_rec-irtk_T2w.nii.gz'\n"
            "MANUAL_PATH = EXAMPLE/'sub-sta30_rec-irtk_T2w_dseg.nii.gz'\n"
            "CHECKPOINT = ROOT/'models/KISPI-all_fss.ckpt'\n"
            "OUT = ROOT/'demo_outputs/real_example'; OUT.mkdir(parents=True, exist_ok=True)\n"
            "PREDICTED_PATH = OUT/'sub-sta30_fetalsynthseg.nii.gz'\n"
            "required = [IMAGE_PATH, MANUAL_PATH] + ([] if PREDICTED_PATH.exists() else [CHECKPOINT])\n"
            "missing = [p for p in required if not p.exists()]\n"
            "if missing:\n"
            "    raise FileNotFoundError('Run ./scripts/install_fetalsynthseg.sh --accept-license first. Missing: ' + ', '.join(map(str, missing)))\n"
            "print('Real example:', IMAGE_PATH.name, '• gestational age 30 weeks')"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Run FetalSynthSeg and inspect standard orientation\n\n"
            "The wrapper reproduces the official 0.5-mm preprocessing and U-Net, verifies the checkpoint "
            "SHA-256 before deserialization, restores labels to native geometry, and records provenance."
        ),
        nbf.v4.new_code_cell(
            "if not PREDICTED_PATH.exists():\n"
            "    from fetal_brain_growth.segmentation import segment_image\n"
            "    metadata = segment_image(IMAGE_PATH, PREDICTED_PATH, CHECKPOINT, device_name='auto', "
            "metadata_path=OUT/'segmentation_provenance.json')\n"
            "    public_keys = ('checkpoint_sha256','official_repository','official_commit','device','elapsed_seconds','torch_version','monai_version','intended_use')\n"
            "    print(json.dumps({key: metadata[key] for key in public_keys}, indent=2))\n"
            "else:\n"
            "    print('Using previously generated prediction:', PREDICTED_PATH.name)\n"
            "qc_figure = save_radiology_figure(IMAGE_PATH, PREDICTED_PATH, OUT/'segmentation_qc.png', "
            "subject_id='IMAGINE atlas example sub-sta30', gestational_age_weeks=30.0, fill_alpha=0.05)\n"
            "display(Image(filename=str(qc_figure), width=1200))"
        ),
        nbf.v4.new_markdown_cell(
            "The three panels are reoriented to canonical RAS and shown in radiological convention. "
            "Contours preserve the underlying T2 anatomy; patient right appears on image left in axial and coronal views."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. One-case execution validation against the supplied manual labels\n\n"
            "Dice here checks model loading, preprocessing, inverse resampling, geometry, and label identity on one "
            "real labeled image. It is not a cohort-level accuracy claim."
        ),
        nbf.v4.new_code_cell(
            "dice = tissue_dice(PREDICTED_PATH, MANUAL_PATH)\n"
            "display(dice[['label','tissue','dice','predicted_voxels','manual_voxels']].style.format({'dice':'{:.3f}'}))\n"
            "print(f\"Mean tissue Dice: {dice.dice.mean():.3f}; range {dice.dice.min():.3f}–{dice.dice.max():.3f}\")"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Affine-aware tissue and aggregate volumes\n\n"
            "Each volume is `voxel count × abs(det(affine[:3,:3])) / 1000` mL. Native FetalSynthSeg tissues "
            "and definition-aware literature aggregates remain separate."
        ),
        nbf.v4.new_code_cell(
            "tissues, aggregates, qc = measure_segmentation(PREDICTED_PATH, subject_id='sub-sta30', gestational_age_weeks=30.0)\n"
            "display(tissues[['label','region','volume_ml']].style.format({'volume_ml':'{:.2f}'}))\n"
            "display(aggregates[['region','labels','volume_ml']].style.format({'volume_ml':'{:.2f}'}))\n"
            "qc"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Position the real case on multiple quantiles\n\n"
            "### Secondary literature comparison: Ren 2022 summary-data reconstruction\n\n"
            "These bands are **reconstructed centiles**, not quantiles directly fitted from the individual "
            "participants in this notebook and not confidence intervals around the mean. For each of the eight "
            "Ren measures, the repository contains the paper's weekly mean `μ(t)` and SD `σ(t)`. For quantile "
            "`q`, the curve is\n\n"
            "`Q_q(t) = max(0, μ(t) + Φ⁻¹(q) × σ(t))`.\n\n"
            "The displayed values are P3/P10/P25/P50/P75/P90/P97. The default `interpolate` method linearly "
            "interpolates mean and SD between completed weeks on a 0.05-week grid, preserving the published "
            "integer-week values. At 30 weeks, for example, Ren total-brain mean = 175.60 mL and SD = 6.09 mL, "
            "giving approximately P3 = 164.1, P10 = 167.8, P25 = 171.5, P50 = 175.6, P75 = 179.7, "
            "P90 = 183.4, and P97 = 187.1 mL.\n\n"
            "With `quadratic` or `cubic`, mean is fitted in volume space and log(SD) is fitted polynomially so "
            "SD stays positive. `auto` uses leave-one-week-out error and selects cubic only when its combined "
            "normalized score improves by more than 5%. Published mean-curve coefficients alone do not define "
            "centiles. The current public demo uses Ren 2022, not the BMJ Fetal & Neonatal paper."
        ),
        nbf.v4.new_code_cell(
            "curves, curve_metadata = build_table_curves(method='interpolate')\n"
            "scores = score_against_curves(aggregates, curves)\n"
            "chart = save_growth_chart(curves, OUT/'growth_chart.png', observations=scores, "
            "regions=tuple(REFERENCE_GROUPS), "
            "title='Real fetal MRI example across all Ren 2022 volume measures', "
            "subtitle='30-week IMAGINE atlas • green = definition-aligned screen; orange = comparison only')\n"
            "display(Image(filename=str(chart), width=1200))\n"
            "display(scores[['region','volume_ml','estimated_percentile_bounded','status','interpretation_note']])"
        ),
        nbf.v4.new_markdown_cell(
            "### Case position and reference flags\n\n"
            "At the fetus's exact gestational age, each expected quantile is interpolated from the curve. The "
            "reported percentile is then linearly interpolated between the seven quantile values and is **bounded "
            "to P3–P97**: `P3 bounded` means P3 or lower and `P97 bounded` means P97 or higher, not an exact tail "
            "probability. A research flag is produced below P3 or above P97. All eight measures are plotted, but "
            "only total brain, intracranial volume, external CSF, and cerebellum are definition-aligned enough for "
            "automated Ren flags; the four orange points are comparison-only because anatomical definitions differ."
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Primary local reference: FeTA-generated matched quantiles\n\n"
            "For FetalSynthSeg/FeTA-label analysis, this is the preferred local teaching comparison because its "
            "nine regions use the same label ontology. It uses 30 QC-passing "
            "expert segmentations labeled neurotypical, 22.7–34.8 weeks, with a quadratic centered-age model "
            "in log-volume space and a constant log-residual SD. Its quantiles are "
            "`Q_q(GA) = exp(fitted_log_volume(GA) + Φ⁻¹(q) × residual_SD)`. Unlike the Ren comparison, all nine "
            "panels use the native FeTA/FetalSynthSeg label definitions.\n\n"
            "The plot is generated when a cached local fit or the mounted FeTA dataset is available. It contains "
            "only aggregate quantile curves and the public IMAGINE atlas observation—no FeTA subject image. "
            "This small, partly in-sample reference is for teaching, not a validated clinical norm."
        ),
        nbf.v4.new_code_cell(
            "FETA_REFERENCE_DIR = ROOT/'meeting_outputs/feta_10_cases_matched'\n"
            "FETA_CURVES_PATH = FETA_REFERENCE_DIR/'feta_neurotypical_reference_curves.csv'\n"
            "FETA_METADATA_PATH = FETA_REFERENCE_DIR/'feta_neurotypical_reference_metadata.json'\n"
            "feta_curves = feta_metadata = None\n"
            "if FETA_CURVES_PATH.exists() and FETA_METADATA_PATH.exists():\n"
            "    feta_curves = pd.read_csv(FETA_CURVES_PATH)\n"
            "    feta_metadata = json.loads(FETA_METADATA_PATH.read_text())\n"
            "else:\n"
            "    try:\n"
            "        feta_root = resolve_feta_root()\n"
            "    except FileNotFoundError:\n"
            "        feta_root = None\n"
            "    if feta_root is not None:\n"
            "        feta_curves, feta_metadata, control_tissues, control_volumes, control_qc = build_feta_matched_reference(feta_root, degree=2)\n"
            "        save_feta_matched_reference(FETA_REFERENCE_DIR, feta_curves, feta_metadata, control_tissues, control_volumes, control_qc)\n"
            "if feta_curves is None:\n"
            "    display(Markdown('**FeTA quantile plot skipped:** mount FeTA 2.2 or copy a generated FeTA reference into `meeting_outputs/feta_10_cases_matched/`. The Ren demonstration above remains fully reproducible from public files.'))\n"
            "else:\n"
            "    feta_case_volumes = pd.concat([\n"
            "        aggregates[aggregates.region.isin(['total_brain','intracranial_volume'])],\n"
            "        tissues,\n"
            "    ], ignore_index=True, sort=False)\n"
            "    feta_scores = score_against_curves(feta_case_volumes, feta_curves, definition_guard=False)\n"
            "    feta_chart = save_growth_chart(\n"
            "        feta_curves, OUT/'growth_chart_feta_neurotypical.png', observations=feta_scores,\n"
            "        regions=FETA_MATCHED_REFERENCE_REGIONS,\n"
            "        title='Public fetal MRI example on FeTA-generated matched quantiles',\n"
            "        subtitle=f\"{feta_metadata['subjects']} QC-passing neurotypical FeTA controls • degree-{feta_metadata['degree']} log-volume model • teaching reference\",\n"
            "        dpi=300,\n"
            "    )\n"
            "    display(Image(filename=str(feta_chart), width=1200))\n"
            "    display(feta_scores[['region','volume_ml','estimated_percentile_bounded','status']])"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Primary-reference radiology case report\n\n"
            "When the FeTA-matched reference is available, this report uses its nine definition-compatible panels. "
            "Ren is used only as a fallback when the local FeTA fit is unavailable."
        ),
        nbf.v4.new_code_cell(
            "if feta_curves is not None:\n"
            "    report_curves, report_scores = feta_curves, feta_scores\n"
            "    report_regions = tuple(FETA_MATCHED_REFERENCE_REGIONS)\n"
            "    report_reference = 'FeTA-matched teaching reference'\n"
            "else:\n"
            "    report_curves, report_scores = curves, scores\n"
            "    report_regions = tuple(REFERENCE_GROUPS)\n"
            "    report_reference = 'Ren 2022 compatibility-limited comparison'\n"
            "report = save_case_report(IMAGE_PATH, PREDICTED_PATH, report_curves, report_scores, OUT/'case_report.png', "
            "subject_id='IMAGINE atlas example sub-sta30', gestational_age_weeks=30.0, dice=dice, "
            "segmentation_source=f'FetalSynthSeg prediction • {report_reference}', regions=report_regions)\n"
            "display(Image(filename=str(report), width=1400))"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation and model choices\n\n"
            "- A single scan is a cross-sectional position, not that fetus's longitudinal growth trajectory.\n"
            "- A reference flag is not an abnormality diagnosis, and within-range volume does not exclude abnormal morphology.\n"
            "- `interpolate` preserves published weekly means; `quadratic`, `cubic`, and leave-one-week-out `auto` are available.\n"
            "- The FeTA-generated chart is label-compatible but small and partly in-sample; prefer a larger reviewed cohort processed with the same frozen FetalSynthSeg model for validation.\n"
            "- Review the complete 3-D MRI/segmentation and clinical context with a fetal neuroradiologist."
        ),
    ]
    nbf.write(notebook, root / "notebooks" / "Radiology_Meeting_Demo.ipynb")


if __name__ == "__main__":
    main()
