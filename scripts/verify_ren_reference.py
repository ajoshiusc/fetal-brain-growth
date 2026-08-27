#!/usr/bin/env python3
"""Cross-check bundled Ren 2022 mean/SD values against official article HTML."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


MEASURE = re.compile(r"(?:[A-H]:)?\s*(TBV|ICV|GMV|SBV|e-CSFV|VV|CBV|BM)", re.IGNORECASE)


def extract_official_table(html_or_url: str) -> pd.DataFrame:
    table = pd.read_html(html_or_url)[0]
    initial = MEASURE.search(str(table.columns[3][0]))
    if initial is None:
        raise ValueError("Could not identify the first measure block.")
    measure = initial.group(1)
    records = []
    for _, row in table.iterrows():
        marker = MEASURE.search(str(row.iloc[3]))
        if marker and pd.isna(row.iloc[0]):
            measure = marker.group(1)
            continue
        try:
            age = int(float(row.iloc[0]))
        except (TypeError, ValueError):
            continue
        if 19 <= age <= 37:
            records.append(
                {
                    "measure": measure,
                    "gestational_age_weeks": age,
                    "mean_official": float(row.iloc[6]),
                    "sd_official": float(row.iloc[7]),
                }
            )
    extracted = pd.DataFrame(records)
    # The source HTML repeats 36 at the top of GMV and e-CSFV; the article's
    # complete descending sequence makes the first row 37 (values unchanged).
    for measure_name in ("GMV", "e-CSFV"):
        group = extracted.loc[extracted.measure.str.casefold() == measure_name.casefold()]
        duplicates = group.index[group.gestational_age_weeks == 36]
        if len(duplicates) == 2 and not (group.gestational_age_weeks == 37).any():
            extracted.loc[duplicates[0], "gestational_age_weeks"] = 37
    extracted["measure"] = extracted["measure"].replace({"E-CSFV": "e-CSFV"})
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", required=True, help="Downloaded official HTML path or article URL")
    parser.add_argument(
        "--reference",
        default=str(Path(__file__).resolve().parents[1] / "references" / "ren2022_weekly_mean_sd.csv"),
    )
    args = parser.parse_args()
    official = extract_official_table(args.html)
    bundled = pd.read_csv(args.reference)
    merged = bundled.merge(official, on=["measure", "gestational_age_weeks"], how="outer", indicator=True)
    mismatches = merged.loc[
        (merged._merge != "both")
        | ~np.isclose(merged.mean_ml, merged.mean_official, equal_nan=False)
        | ~np.isclose(merged.sd_ml, merged.sd_official, equal_nan=False)
    ]
    if not mismatches.empty:
        print(mismatches.to_string(index=False))
        raise SystemExit("Reference verification failed.")
    print(f"Verified {len(merged)} measure/week rows against the official article HTML.")


if __name__ == "__main__":
    main()
