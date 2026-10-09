from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# PATH
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_union_raw.csv"
)


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


print("=" * 75)
print("UNION SAMPLE QC")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

y = df["class"].to_numpy()


print(f"\nRows: {len(df):,}")
print(
    f"Flood: {(df['class'] == 1).sum():,}"
)
print(
    f"Background: {(df['class'] == 0).sum():,}"
)


# ============================================================
# SINGLE-FEATURE AUC
#
# AUC may be <0.5 when higher values indicate LESS flooding.
# We flip it to report discrimination strength:
#
# max(AUC, 1-AUC)
# ============================================================

print("\nSingle-feature discrimination:\n")

results = []


for feature in FEATURES:

    x = df[feature].to_numpy()

    auc_raw = roc_auc_score(
        y,
        x
    )

    auc_strength = max(
        auc_raw,
        1 - auc_raw
    )

    direction = (
        "higher → flood"
        if auc_raw >= 0.5
        else "lower → flood"
    )

    results.append(
        (
            feature,
            auc_strength,
            direction
        )
    )


results.sort(
    key=lambda z: z[1],
    reverse=True
)


for feature, auc, direction in results:

    print(
        f"{feature:<22}"
        f"AUC={auc:.4f}   "
        f"{direction}"
    )


# ============================================================
# BLOCK COUNTS
# ============================================================

print("\nSpatial blocks:")

for cls, label in [
    (1, "Flood"),
    (0, "Background"),
]:

    subset = df[
        df["class"] == cls
    ]

    counts = (
        subset["sampling_block"]
        .value_counts()
    )

    print(
        f"\n{label}"
    )

    print(
        f"  Blocks: {len(counts):,}"
    )

    print(
        f"  Median samples/block: "
        f"{counts.median():.1f}"
    )

    print(
        f"  Maximum samples/block: "
        f"{counts.max():,}"
    )


# ============================================================
# FEATURE RANGES
# ============================================================

print("\nClass medians:\n")


for feature in FEATURES:

    bg = df.loc[
        df["class"] == 0,
        feature
    ].median()

    fl = df.loc[
        df["class"] == 1,
        feature
    ].median()

    print(
        f"{feature:<22}"
        f"background={bg:>12.4f}   "
        f"flood={fl:>12.4f}"
    )


print("\nQC complete.")