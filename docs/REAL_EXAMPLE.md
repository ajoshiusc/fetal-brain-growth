# Real fetal MRI example provenance

The README and executed demonstration use the 30-week `sub-sta30` T2-weighted
fetal atlas example distributed by the official FetalSynthSeg repository. The
FetalSynthSeg authors identify the image and manual segmentation as borrowed
from:

Gholipour A, Velasco-Annis C, Rollins CK, Vasung L, Ouaalam A, Ortinau C,
Akhondi-Asl A, Clancy S, Yang E, Estroff J, Warfield SK. *IMAGINE Fetal
T2-weighted MRI Atlas.* Harvard Dataverse, V1, 2023.
[doi:10.7910/DVN/WE9JVR](https://doi.org/10.7910/DVN/WE9JVR)

The Harvard Dataverse API identifies this release as **CC0 1.0**. The checked-in
PNG figures are derived visualizations; no source or predicted NIfTI volume and
no FetalSynthSeg checkpoint is committed.

The displayed automatic segmentation was produced with the official
`KISPI-all_fss.ckpt` and compared with the example's manual seven-tissue label
map. [`real_example_dice.csv`](real_example_dice.csv) records the per-tissue
Dice. This is a one-case pipeline execution check, not a cohort performance
estimate.

Growth-chart positions use the automated label volumes, not the manual labels.
They are compared with reconstructed centiles from Ren et al. 2022 weekly
summary data. Cross-study acquisition, population, SVR, and boundary-definition
differences remain; the result is not a diagnosis.

This public example intentionally retains the Ren comparison because the FeTA
2.2 control data are access-controlled and their derived curves are not
committed. When FeTA is mounted locally, `fbg feta-gallery` and
`notebooks/FeTA_10_Case_Meeting.ipynb` instead run or reuse automatic
FetalSynthSeg predictions and use the resulting exact-label nine-measure FeTA
neurotypical teaching reference by default.
