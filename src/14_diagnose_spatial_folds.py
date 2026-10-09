from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)

OUT_DIR = Path(
    "results/phase0"
)

FIG_DIR = Path(
    "results/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
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


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)


print("=" * 80)
print("SPATIAL FOLD DIAGNOSTIC")
print("=" * 80)


# ============================================================
# CONVERT COORDINATES TO LAT/LON
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:3395",
    "EPSG:4326",
    always_xy=True
)


lon, lat = transformer.transform(
    df["x"].to_numpy(),
    df["y"].to_numpy()
)


df["longitude"] = np.asarray(lon)
df["latitude"] = np.asarray(lat)


# ============================================================
# 1. BASIC FOLD SUMMARY
# ============================================================

summary_rows = []


for fold in range(1, 6):

    part = df[
        df["spatial_fold"] == fold
    ]

    summary_rows.append({
        "fold": fold,

        "n_samples":
            len(part),

        "n_flood":
            int(
                (part["class"] == 1).sum()
            ),

        "n_background":
            int(
                (part["class"] == 0).sum()
            ),

        "n_blocks":
            part[
                "sampling_block"
            ].nunique(),

        "lat_min":
            part[
                "latitude"
            ].min(),

        "lat_max":
            part[
                "latitude"
            ].max(),

        "lat_median":
            part[
                "latitude"
            ].median(),

        "lon_min":
            part[
                "longitude"
            ].min(),

        "lon_max":
            part[
                "longitude"
            ].max(),

        "lon_median":
            part[
                "longitude"
            ].median(),
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUT_DIR
    / "spatial_fold_summary.csv",
    index=False
)


print(
    "\nFold geography:\n"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 2. FEATURE MEDIANS BY FOLD
# ============================================================

median_rows = []


for fold in range(1, 6):

    for cls, class_name in [
        (0, "background"),
        (1, "flood"),
    ]:

        subset = df[
            (df["spatial_fold"] == fold)
            & (df["class"] == cls)
        ]


        row = {
            "fold": fold,
            "class": class_name,
            "n": len(subset),
        }


        for feature in FEATURES:

            row[
                feature
            ] = subset[
                feature
            ].median()


        median_rows.append(
            row
        )


medians = pd.DataFrame(
    median_rows
)


medians.to_csv(
    OUT_DIR
    / "spatial_fold_feature_medians.csv",
    index=False
)


# ============================================================
# 3. FOLD 5 VS FOLDS 1–4
# ============================================================

fold5 = df[
    df["spatial_fold"] == 5
]

others = df[
    df["spatial_fold"] != 5
]


comparison_rows = []


for cls, class_name in [
    (0, "background"),
    (1, "flood"),
]:

    f5 = fold5[
        fold5["class"] == cls
    ]

    other = others[
        others["class"] == cls
    ]


    for feature in FEATURES:

        f5_median = f5[
            feature
        ].median()

        other_median = other[
            feature
        ].median()


        comparison_rows.append({
            "class": class_name,
            "feature": feature,

            "fold5_median":
                f5_median,

            "fold1_4_median":
                other_median,

            "difference":
                f5_median
                - other_median,
        })


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUT_DIR
    / "fold5_vs_other_folds.csv",
    index=False
)


print(
    "\n" + "=" * 80
)

print(
    "FOLD 5 VS FOLDS 1–4"
)

print(
    "=" * 80
)


for class_name in [
    "flood",
    "background"
]:

    print(
        f"\n{class_name.upper()}"
    )


    part = comparison[
        comparison[
            "class"
        ] == class_name
    ]


    for _, row in part.iterrows():

        print(
            f"{row['feature']:<22}"
            f"Fold5={row['fold5_median']:>12.4f}   "
            f"Others={row['fold1_4_median']:>12.4f}"
        )


# ============================================================
# 4. MAP THE FIVE FOLDS
# ============================================================

plt.figure(
    figsize=(8, 9)
)


scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["spatial_fold"],
    s=5,
    alpha=0.55
)


plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "Spatial Validation Folds"
)

plt.colorbar(
    scatter,
    label="Spatial fold"
)


plt.tight_layout()


FIG_OUTPUT = (
    FIG_DIR
    / "spatial_validation_folds.png"
)


plt.savefig(
    FIG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. FOLD 5 ONLY MAP
# ============================================================

plt.figure(
    figsize=(8, 9)
)


other_points = df[
    df["spatial_fold"] != 5
]


fold5_points = df[
    df["spatial_fold"] == 5
]


plt.scatter(
    other_points["longitude"],
    other_points["latitude"],
    s=3,
    alpha=0.15,
    label="Folds 1–4"
)


plt.scatter(
    fold5_points["longitude"],
    fold5_points["latitude"],
    s=8,
    alpha=0.7,
    label="Fold 5"
)


plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "Location of Spatial Validation Fold 5"
)

plt.legend()


plt.tight_layout()


FOLD5_FIG = (
    FIG_DIR
    / "spatial_fold5_location.png"
)


plt.savefig(
    FOLD5_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nSaved:\n"
    f"{OUT_DIR / 'spatial_fold_summary.csv'}"
)

print(
    f"{OUT_DIR / 'spatial_fold_feature_medians.csv'}"
)

print(
    f"{OUT_DIR / 'fold5_vs_other_folds.csv'}"
)

print(
    f"{FIG_OUTPUT}"
)

print(
    f"{FOLD5_FIG}"
)