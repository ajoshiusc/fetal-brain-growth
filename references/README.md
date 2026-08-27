# Reference provenance and construction

This directory contains transparent, machine-readable source values—never
serialized statistical models. The software keeps distinct reference-building
operations separate because they answer different questions.

## 1. FeTA 2.2 protocol-matched teaching reference (default)

FeTA and FetalSynthSeg share the same output ontology: background 0, external
CSF 1, cortical gray matter 2, white matter 3, ventricles 4, cerebellum 5, deep
gray matter 6, and brainstem 7. This permits exact-label tissue comparisons and
two deterministic aggregates:

- total brain = labels 2 + 3 + 5 + 6 + 7;
- intracranial volume = labels 1 through 7.

Run:

```bash
fbg feta-reference --degree 2 --output-dir outputs/feta_reference
```

The builder reads `participants.tsv`, retains only the case-insensitive exact
label `Pathology == Neurotypical`, measures expert segmentations using the
NIfTI affine, and excludes only technical segmentation-QC failures. It does not
remove cases based on their volume. On the current local FeTA 2.2 release, 31
rows pass the phenotype filter and 30 pass technical QC, spanning 22.7–34.8
weeks.

For region `r`, centered age `x = GA - mean(GA)`, and degree `d`:

```text
log(V_r) = beta_r,0 + beta_r,1*x + ... + beta_r,d*x^d + error_r
Q_r,q(GA) = exp(fitted_log(V_r) + Phi_inverse(q) * residual_SD_r)
```

The default is degree 2; degree 3 is available only as an explicit sensitivity
analysis. In the current 30-case fit, cubic leave-one-subject-out log-RMSE was
no better for any measure and was up to 8.4% worse. One constant residual SD is
estimated per region in log-volume space.
This restrained location-scale model is used because 30 controls are
insufficient for stable direct extreme-quantile regression. Outputs record the
coefficients, residual SD, log-scale R², leave-one-subject-out RMSE, included
subject IDs, QC exclusions, and P3/P10/P25/P50/P75/P90/P97 curves.

The age grid stops at the youngest and oldest included controls. These are
small, cross-sectional, in-sample teaching references—not clinical norms. FeTA
data and locally derived tables remain subject to FeTA research/education terms
and are excluded from Git.

Source: Payette K et al. *An automatic multi-tissue human fetal brain
segmentation benchmark using the Fetal Tissue Annotation Dataset.* Scientific
Data. 2021;8:167.
[doi:10.1038/s41597-021-00946-3](https://doi.org/10.1038/s41597-021-00946-3)

## 2. Ren 2022 weekly summary table

[`ren2022_weekly_mean_sd.csv`](ren2022_weekly_mean_sd.csv) is a manual,
double-checked transcription of the weekly **mean and standard deviation** in
Table 1 of:

Ren J-Y et al. *Quantification of Intracranial Structures Volume in Fetuses
Using 3-D Volumetric MRI: Normal Values at 19 to 37 Weeks' Gestation.* Frontiers
in Neuroscience. 2022;16:886083.
[doi:10.3389/fnins.2022.886083](https://doi.org/10.3389/fnins.2022.886083)

The study included 188 fetuses with normal-appearing brains and used manual
segmentation on 0.5-mm isotropic SVR reconstructions. These are published
summary statistics, not participant-level data.

Spreadsheet construction:

1. The eight measure blocks (TBV, ICV, GMV, SBV, e-CSFV, VV, CBV and BM) were
   transcribed from Table 1 for completed weeks 19–37.
2. Only `mean_ml` and `sd_ml` were retained because they are sufficient for a
   consistent set of requested centiles under an explicit distributional model.
3. The two rows printed as week 36 at the top of the GMV and e-CSFV blocks were
   assigned 37 then 36 because each block otherwise follows a complete 37-to-19
   sequence. No volume value was edited.
4. The CSV is validated for exactly one positive-SD row per measure/week.
5. `fbg build-reference` interpolates the weekly mean and SD, or fits a
   quadratic/cubic polynomial to mean and log(SD). It computes
   `Q_q(age) = mean(age) + Phi_inverse(q) * SD(age)` and floors impossible
   negative volumes at zero. Thus P3/P10/P25/P50/P75/P90/P97 are a **Normal
   approximation reconstructed from summary data**, not centiles published by
   the authors and not a refit of individual data.

`--method auto` compares quadratic and cubic mean/log-SD fits by
leave-one-week-out RMSE and only selects cubic when its combined normalized
score improves by more than 5%. `--method interpolate` is the conservative
default because it reproduces the transcribed weekly means exactly.

The repository includes a reproducible source check. After downloading the
official article HTML, run `python scripts/verify_ren_reference.py --html
article.html`; all 152 measure/week mean and SD pairs must match exactly.

## 3. Published polynomial coefficients

[`published_models.json`](published_models.json) stores coefficients in
ascending order (`intercept`, `GA`, `GA²`, ...). These models draw a predicted
**mean only**; the software deliberately does not invent centile bands.

- Jarvis et al. 2016: `TBV = 89.69 - 13.33 GA + 0.53 GA²`, 18–36 weeks,
  adjusted R² 0.974.
  [doi:10.1002/pd.4961](https://doi.org/10.1002/pd.4961)
- Ren et al. 2022: `TBV = 47.41 - 9.57 GA + 0.45 GA²`, 19–37 weeks, R² 0.98.
  [doi:10.3389/fnins.2022.886083](https://doi.org/10.3389/fnins.2022.886083)

Some other equations printed in Ren 2022 appear internally inconsistent (the
e-CSFV equation duplicates ICV and the BM equation yields negative volumes).
They are not silently repaired or included.

Bouachba et al.'s prospective 260-pregnancy MRI study (16–36 weeks) was
published in *Archives of Disease in Childhood: Fetal & Neonatal* in 2025
([doi:10.1136/archdischild-2024-328310](https://doi.org/10.1136/archdischild-2024-328310)).
Its abstract reports total brain and cerebellar manual volumes and a strong
age association, but the full coefficient/reference table is not openly
available under reusable terms. This repository does not infer values from its
figure; an authorized table can be supplied later as a separately named model.

## 4. Preferred large FetalSynthSeg-matched reference

For tissue-level charts, fit direct quantiles to a reviewed local control
cohort processed with the same frozen FetalSynthSeg checkpoint:

```bash
fbg fit-reference \
  --volumes control_tissue_volumes.csv \
  --method spline \
  --output outputs/local_reference.csv
```

The default directly fits P3/P10/P25/P50/P75/P90/P97 in log-volume space with
cubic B-spline quantile regression. `--method quadratic` and `--method cubic`
are available for a pre-specified polynomial analysis. The default minimum of
120 independent fetuses is a software guardrail, not a sample-size calculation.
Use one scan per fetus or an explicitly longitudinal model.

## 5. Other public MRI references evaluated

BOUNTI is the most useful larger public alternative located. It used reviewed,
manually corrected segmentations from 390 healthy controls across three
acquisition protocols and 21–38 weeks to produce P5/P50/P95 charts for nine
structures. Its 19-region parcellation is not identical to FeTA: cerebellar
vermis and several ventricular compartments are separated, among other
differences. The current GPL-3.0
[Multi-BOUNTI reporting code](https://github.com/SVRTK/perinatal-brain-mri-analysis)
contains polynomial reporting models. They should be used with the matching
BOUNTI pipeline, not copied into this MIT project and silently applied to FeTA
labels.

Uus AU et al. *BOUNTI: Brain vOlumetry and aUtomated parcellatioN for 3D feTal
MRI.* eLife reviewed preprint. 2023;12:RP88818.
[doi:10.7554/eLife.88818.1](https://doi.org/10.7554/eLife.88818.1)

Kyriakopoulou et al. published P5/P50/P95 MRI biometry from 127 normal fetuses
at 21–38 weeks and an online calculator. Its manual supratentorial, cortical,
ventricular, cerebellar, and extra-cerebral CSF definitions are informative but
not a drop-in FeTA/FetalSynthSeg seven-tissue reference.
[doi:10.1007/s00429-016-1342-6](https://doi.org/10.1007/s00429-016-1342-6)

Andescavage et al. reported MRI quantile-regression charts from 166 healthy
fetuses, but the accessible article presents several tissue trajectories as
figures rather than a reusable complete FeTA-compatible coefficient table. The
project does not digitize figures to manufacture precise reference curves.
[doi:10.1093/cercor/bhw306](https://doi.org/10.1093/cercor/bhw306)

## Label mapping and interpretation

| Ren measure | FetalSynthSeg/FeTA labels | Automated status? |
|---|---|---|
| TBV | 2 + 3 + 5 + 6 + 7 | research flag |
| ICV | 1 through 7 | research flag |
| GMV | 2 | comparison only |
| SBV | 3 + 6 | comparison only |
| e-CSFV | 1 | research flag |
| VV | 4 | comparison only |
| CBV | 5 | research flag |
| BM | 7 | comparison only |

The non-scoreable mappings differ materially in boundary definitions. In
particular, Ren measures lateral ventricles while FeTA label 4 is broader, and
FeTA label 7 may include spinal cord. A point outside P3–P97 is a reference flag,
not a diagnosis. All segmentations require visual review and local validation.
