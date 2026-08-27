"""FeTA/FetalSynthSeg labels and definition-aware reference mappings."""

from __future__ import annotations

from collections import OrderedDict


FETA_LABELS: OrderedDict[int, str] = OrderedDict(
    [
        (0, "background"),
        (1, "external_csf"),
        (2, "cortical_gray_matter"),
        (3, "white_matter"),
        (4, "ventricles"),
        (5, "cerebellum"),
        (6, "deep_gray_matter"),
        (7, "brainstem"),
    ]
)

FETA_TISSUE_REGIONS = tuple(name for label, name in FETA_LABELS.items() if label)

# Directly available from a FeTA/FetalSynthSeg label map. Total brain and ICV
# are deterministic aggregates; the remaining seven entries are native labels.
FETA_MATCHED_REFERENCE_REGIONS = (
    "total_brain",
    "intracranial_volume",
    *FETA_TISSUE_REGIONS,
)

LABEL_TITLES = {
    1: "External CSF",
    2: "Cortical gray matter",
    3: "White matter",
    4: "Ventricles",
    5: "Cerebellum",
    6: "Deep gray matter",
    7: "Brainstem",
}

# High-contrast colors designed for dark MRI backgrounds.
LABEL_COLORS = {
    1: "#43A5FF",
    2: "#F29E2E",
    3: "#45C46A",
    4: "#315BCE",
    5: "#E7439B",
    6: "#9A56C7",
    7: "#EB4A4A",
}

# Compositions approximating Ren et al. 2022 manual definitions.
REFERENCE_GROUPS: OrderedDict[str, tuple[int, ...]] = OrderedDict(
    [
        ("total_brain", (2, 3, 5, 6, 7)),
        ("intracranial_volume", (1, 2, 3, 4, 5, 6, 7)),
        ("cortical_gray_matter", (2,)),
        ("subcortical_brain_tissue", (3, 6)),
        ("external_csf", (1,)),
        ("ventricles", (4,)),
        ("cerebellum", (5,)),
        ("brainstem", (7,)),
    ]
)

REFERENCE_MEASURES = OrderedDict(
    [
        ("total_brain", "TBV"),
        ("intracranial_volume", "ICV"),
        ("cortical_gray_matter", "GMV"),
        ("subcortical_brain_tissue", "SBV"),
        ("external_csf", "e-CSFV"),
        ("ventricles", "VV"),
        ("cerebellum", "CBV"),
        ("brainstem", "BM"),
    ]
)

REGION_TITLES = {
    "total_brain": "Total brain",
    "intracranial_volume": "Intracranial volume",
    "cortical_gray_matter": "Cortical gray matter",
    "subcortical_brain_tissue": "White + deep gray matter",
    "external_csf": "External CSF",
    "ventricles": "Ventricles",
    "cerebellum": "Cerebellum",
    "brainstem": "Brainstem",
    **{name: title for name, title in zip(list(FETA_LABELS.values())[1:], LABEL_TITLES.values())},
}

# These four had acceptable protocol alignment in a manual-atlas check.
SCOREABLE_REGIONS = {
    "total_brain",
    "intracranial_volume",
    "external_csf",
    "cerebellum",
}

DEFINITION_MISMATCH_NOTES = {
    "cortical_gray_matter": "FeTA and reference cortical boundaries differ.",
    "subcortical_brain_tissue": "White/deep-gray allocation differs between protocols.",
    "ventricles": "The reference measures lateral ventricles; FeTA label 4 is broader.",
    "brainstem": "FeTA may include spinal cord with brainstem.",
}
