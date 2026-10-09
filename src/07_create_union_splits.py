from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

INPUT = Path(
    "Data/samples/training_samples_union_raw.csv"
)

OUTPUT = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)


print("=" * 80)
print("CREATING RANDOM + SPATIAL VALIDATION SPLITS")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print(f"\nRows: {len(df):,}")

print(
    f"Flood: "
    f"{(df['class'] == 1).sum():,}"
)

print(
    f"Background: "
    f"{(df['class'] == 0).sum():,}"
)


# ============================================================
# 1. RANDOM STRATIFIED HOLD-OUT
#
# 77% train
# 23% test
#
# Same ratio we used before, so comparison remains easy.
# ============================================================

indices = np.arange(
    len(df)
)


train_idx, test_idx = train_test_split(
    indices,
    test_size=0.23,
    random_state=RANDOM_SEED,
    stratify=df["class"],
)


df["random_split"] = "train"

df.loc[
    test_idx,
    "random_split"
] = "test"


print("\n" + "-" * 80)
print("RANDOM STRATIFIED SPLIT")
print("-" * 80)


for split in [
    "train",
    "test"
]:

    part = df[
        df["random_split"] == split
    ]

    flood = int(
        (part["class"] == 1).sum()
    )

    background = int(
        (part["class"] == 0).sum()
    )

    print(
        f"{split:<6} "
        f"total={len(part):,}   "
        f"flood={flood:,}   "
        f"background={background:,}"
    )


# ============================================================
# 2. SPATIAL BLOCK CROSS-VALIDATION
#
# All samples belonging to the same 100-km block stay together.
# Therefore a block cannot appear in both training and
# validation for the same fold.
# ============================================================

X_dummy = np.zeros(
    (len(df), 1)
)

y = df["class"].to_numpy()

groups = (
    df["sampling_block"]
    .to_numpy()
)


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED,
)


df["spatial_fold"] = -1


print("\n" + "-" * 80)
print("5-FOLD SPATIAL BLOCK CV")
print("-" * 80)


for fold, (
    train_indices,
    validation_indices
) in enumerate(
    cv.split(
        X_dummy,
        y,
        groups
    ),
    start=1
):

    # --------------------------------------------
    # Assign validation fold number
    # --------------------------------------------

    df.loc[
        validation_indices,
        "spatial_fold"
    ] = fold


    # --------------------------------------------
    # Leakage check
    # --------------------------------------------

    train_blocks = set(
        df.iloc[
            train_indices
        ]["sampling_block"]
    )

    validation_blocks = set(
        df.iloc[
            validation_indices
        ]["sampling_block"]
    )


    overlap = (
        train_blocks
        & validation_blocks
    )


    if overlap:

        raise RuntimeError(
            f"Fold {fold}: "
            f"{len(overlap)} spatial blocks "
            "appear in BOTH train and validation."
        )


    validation = df.iloc[
        validation_indices
    ]


    flood = int(
        (
            validation["class"]
            == 1
        ).sum()
    )

    background = int(
        (
            validation["class"]
            == 0
        ).sum()
    )


    print(
        f"\nFold {fold}"
    )

    print(
        f"  Validation samples: "
        f"{len(validation):,}"
    )

    print(
        f"  Flood:              "
        f"{flood:,}"
    )

    print(
        f"  Background:         "
        f"{background:,}"
    )

    print(
        f"  Validation blocks:  "
        f"{len(validation_blocks):,}"
    )

    print(
        "  Train/test block overlap: 0"
    )


# ============================================================
# FINAL CHECKS
# ============================================================

if (
    df["spatial_fold"]
    == -1
).any():

    raise RuntimeError(
        "Some samples were not assigned "
        "to a spatial fold."
    )


if not set(
    df["spatial_fold"].unique()
) == {1, 2, 3, 4, 5}:

    raise RuntimeError(
        "Unexpected spatial fold labels."
    )


print("\n" + "-" * 80)
print("FINAL CHECK")
print("-" * 80)

print(
    f"Unique spatial blocks: "
    f"{df['sampling_block'].nunique():,}"
)

print(
    "✓ Every sample assigned "
    "to one spatial fold"
)

print(
    "✓ No spatial block can occur "
    "in both train and validation "
    "within a fold"
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


print(
    f"\nSaved:\n{OUTPUT}"
)