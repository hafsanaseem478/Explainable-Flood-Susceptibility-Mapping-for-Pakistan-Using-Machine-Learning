from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TEST_SIZE = 0.23

N_SPATIAL_FOLDS = 5

# 100 km spatial blocks
BLOCK_SIZE_M = 100_000


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_random_background_raw.csv"
)

OUTPUT = Path(
    "Data/samples/training_samples_random_background_with_splits.csv"
)


print("=" * 80)
print("CREATING VALIDATION SPLITS")
print("=" * 80)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(INPUT)

print(f"\nSamples: {len(df):,}")

print("\nClass counts:")
print(
    df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# BASIC SAFETY CHECKS
# ============================================================

if len(df) != 20_000:
    raise RuntimeError(
        f"Expected 20,000 samples, found {len(df):,}"
    )

if df["sample_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate sample IDs found."
    )

if df.duplicated(
    subset=["row", "col"]
).any():
    raise RuntimeError(
        "Duplicate raster cells found."
    )


# ============================================================
# 1. RANDOM STRATIFIED 77/23 SPLIT
# ============================================================

print("\n[1/2] Creating random stratified 77/23 split...")

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["class"],
)

df["random_split"] = ""

df.loc[
    train_idx,
    "random_split"
] = "train"

df.loc[
    test_idx,
    "random_split"
] = "test"


print("\nRandom split:")

random_summary = (
    df.groupby(
        ["random_split", "class"]
    )
    .size()
    .unstack(fill_value=0)
)

print(random_summary)


# ============================================================
# 2. CREATE SPATIAL BLOCKS
# ============================================================

print(
    "\n[2/2] Creating 100-km spatial blocks..."
)

# Original x/y coordinates are EPSG:3395
gdf = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(
        df["x"],
        df["y"]
    ),
    crs="EPSG:3395"
)

# Equal-area CRS with metre coordinates
gdf_ea = gdf.to_crs("EPSG:6933")

x_ea = gdf_ea.geometry.x.to_numpy()
y_ea = gdf_ea.geometry.y.to_numpy()


# ============================================================
# ASSIGN EACH SAMPLE TO A 100-km BLOCK
# ============================================================

block_x = np.floor(
    x_ea / BLOCK_SIZE_M
).astype(int)

block_y = np.floor(
    y_ea / BLOCK_SIZE_M
).astype(int)

df["spatial_block"] = [
    f"B_{bx}_{by}"
    for bx, by in zip(
        block_x,
        block_y
    )
]


n_blocks = df["spatial_block"].nunique()

print(
    f"Unique spatial blocks: {n_blocks:,}"
)

if n_blocks < N_SPATIAL_FOLDS:
    raise RuntimeError(
        "Not enough spatial blocks for "
        "5-fold spatial validation."
    )


# ============================================================
# STRATIFIED GROUP 5-FOLD CV
# ============================================================

sgkf = StratifiedGroupKFold(
    n_splits=N_SPATIAL_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

df["spatial_fold"] = 0

X_dummy = np.zeros(
    (len(df), 1)
)

y = df["class"].to_numpy()

groups = df[
    "spatial_block"
].to_numpy()


for fold, (_, test_fold_idx) in enumerate(
    sgkf.split(
        X_dummy,
        y,
        groups
    ),
    start=1,
):

    df.loc[
        test_fold_idx,
        "spatial_fold"
    ] = fold


# ============================================================
# VALIDATE SPATIAL FOLDS
# ============================================================

if (
    df["spatial_fold"] == 0
).any():
    raise RuntimeError(
        "Some samples were not assigned "
        "to a spatial fold."
    )


print("\nSpatial fold class counts:")

spatial_summary = (
    df.groupby(
        ["spatial_fold", "class"]
    )
    .size()
    .unstack(fill_value=0)
)

print(spatial_summary)


# ============================================================
# CHECK FOR BLOCK LEAKAGE
# ============================================================

print("\nChecking spatial block leakage...")

for fold in range(
    1,
    N_SPATIAL_FOLDS + 1
):

    test_blocks = set(
        df.loc[
            df["spatial_fold"] == fold,
            "spatial_block"
        ]
    )

    train_blocks = set(
        df.loc[
            df["spatial_fold"] != fold,
            "spatial_block"
        ]
    )

    overlap = (
        test_blocks
        & train_blocks
    )

    if overlap:
        raise RuntimeError(
            f"Spatial leakage detected "
            f"in fold {fold}."
        )


print(
    "✓ No spatial block appears "
    "in both training and testing "
    "within any spatial fold."
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION SPLITS COMPLETE")
print("=" * 80)

print(
    f"Random training samples: "
    f"{(df['random_split'] == 'train').sum():,}"
)

print(
    f"Random testing samples:  "
    f"{(df['random_split'] == 'test').sum():,}"
)

print(
    f"Spatial blocks:          "
    f"{n_blocks:,}"
)

print(
    f"Spatial folds:           "
    f"{N_SPATIAL_FOLDS}"
)

print(
    f"\nSaved:\n"
    f"{OUTPUT}"
)