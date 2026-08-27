# Reference provenance and construction

This directory contains transparent, machine-readable source values—never
serialized statistical models. The software keeps three different operations
separate because they answer different questions.

## 1. Ren 2022 weekly summary table

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

## 2. Published polynomial coefficients

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

## 3. Preferred FetalSynthSeg-matched reference

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
