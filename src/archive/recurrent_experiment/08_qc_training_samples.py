from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_raw.csv"
)

OUT_DIR = Path("results/phase0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "training_sample_qc.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "aspect",
    "curvature",
    "distdrainage",
    "distriver",
    "distroads",
    "elevation",
    "ndvi",
    "rainfall_frequency",
    "slope",
    "twi",
]


print("=" * 80)
print("TRAINING SAMPLE QC")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print(f"\nRows: {len(df):,}")

print("\nClass counts:")
print(
    df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# BASIC CHECKS
# ============================================================

print("\nMissing values:")
print(df[FEATURES].isna().sum())

duplicates = df.duplicated(
    subset=["row", "col"]
).sum()

print(
    f"\nDuplicate grid cells: "
    f"{duplicates:,}"
)


# ============================================================
# CLASS-WISE DISTRIBUTION SUMMARY
# ============================================================

records = []

for feature in FEATURES:

    for cls in [0, 1]:

        values = df.loc[
            df["class"] == cls,
            feature
        ]

        records.append(
            {
                "feature": feature,
                "class": cls,
                "n": len(values),
                "min": values.min(),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "mean": values.mean(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "std": values.std(),
            }
        )


summary = pd.DataFrame(records)

summary.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# PRINT EASY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MEDIAN VALUES BY CLASS")
print("=" * 80)

for feature in FEATURES:

    nf = df.loc[
        df["class"] == 0,
        feature
    ].median()

    fl = df.loc[
        df["class"] == 1,
        feature
    ].median()

    print(
        f"{feature:<22}"
        f" non-flood={nf:>12.4f}"
        f"   flood={fl:>12.4f}"
    )


print("\n" + "=" * 80)
print("QC COMPLETE")
print("=" * 80)

print(
    f"\nSaved full summary:\n"
    f"{OUTPUT}"
)