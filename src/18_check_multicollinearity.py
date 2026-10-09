from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

CORRELATION_NOTE_THRESHOLD = 0.70

# These thresholds are for interpretation only.
# We are NOT automatically deleting variables.
VIF_LOW_THRESHOLD = 3.0
VIF_REVIEW_THRESHOLD = 5.0


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)

OUT_DIR = Path(
    "results/ale/tella_protocol/multicollinearity"
)

FIG_DIR = Path(
    "results/figures/ale"
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
# EXACT PREDICTORS USED BY THE MODELS
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
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


required = (
    FEATURES
    + ["random_split"]
)


missing = [
    col
    for col in required
    if col not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            missing
        )
    )


# ============================================================
# USE THE SAME TRAINING DATA USED FOR ALE
# ============================================================

train_df = df[
    df["random_split"] == "train"
].copy()


X = train_df[
    FEATURES
].copy()


if X.isna().any().any():

    raise ValueError(
        "Missing values detected in predictors."
    )


print("=" * 80)
print("MULTICOLLINEARITY DIAGNOSTIC")
print("PEARSON CORRELATION + VIF")
print("=" * 80)


print(
    f"\nTraining observations: "
    f"{len(X):,}"
)

print(
    f"Predictors: "
    f"{len(FEATURES)}"
)


# ============================================================
# 1. PEARSON CORRELATION MATRIX
# ============================================================

corr = X.corr(
    method="pearson"
)


CORR_OUTPUT = (
    OUT_DIR
    / "pearson_correlation_train.csv"
)


corr.to_csv(
    CORR_OUTPUT
)


# ============================================================
# EXTRACT UNIQUE FEATURE PAIRS
# ============================================================

pairs = []


for i in range(
    len(FEATURES)
):

    for j in range(
        i + 1,
        len(FEATURES)
    ):

        feature_1 = FEATURES[i]

        feature_2 = FEATURES[j]

        r = corr.loc[
            feature_1,
            feature_2
        ]


        pairs.append({
            "feature_1":
                feature_1,

            "feature_2":
                feature_2,

            "pearson_r":
                r,

            "abs_pearson_r":
                abs(r),
        })


pair_df = pd.DataFrame(
    pairs
)


pair_df = pair_df.sort_values(
    "abs_pearson_r",
    ascending=False
).reset_index(
    drop=True
)


PAIR_OUTPUT = (
    OUT_DIR
    / "pearson_pairwise_ranked_train.csv"
)


pair_df.to_csv(
    PAIR_OUTPUT,
    index=False
)


# ============================================================
# 2. VARIANCE INFLATION FACTOR
#
# Standard VIF definition:
#
# VIF_j = 1 / (1 - R²_j)
#
# where feature j is regressed against every other predictor.
#
# We standardize before these regressions only for numerical
# stability. Standardization does NOT change VIF.
# ============================================================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


vif_rows = []


for i, feature in enumerate(
    FEATURES
):

    y_feature = X_scaled[
        :,
        i
    ]


    other_indices = [
        j
        for j in range(
            len(FEATURES)
        )
        if j != i
    ]


    X_other = X_scaled[
        :,
        other_indices
    ]


    regression = LinearRegression(
        fit_intercept=True
    )


    regression.fit(
        X_other,
        y_feature
    )


    r_squared = regression.score(
        X_other,
        y_feature
    )


    if r_squared >= 1.0:

        vif = np.inf

    else:

        vif = (
            1.0
            / (
                1.0
                - r_squared
            )
        )


    if vif < VIF_LOW_THRESHOLD:

        interpretation = (
            "low"
        )

    elif vif < VIF_REVIEW_THRESHOLD:

        interpretation = (
            "review"
        )

    else:

        interpretation = (
            "elevated"
        )


    vif_rows.append({
        "feature":
            feature,

        "r_squared_against_other_predictors":
            r_squared,

        "vif":
            vif,

        "interpretation":
            interpretation,
    })


vif_df = pd.DataFrame(
    vif_rows
)


vif_df = vif_df.sort_values(
    "vif",
    ascending=False
).reset_index(
    drop=True
)


VIF_OUTPUT = (
    OUT_DIR
    / "vif_train.csv"
)


vif_df.to_csv(
    VIF_OUTPUT,
    index=False
)


# ============================================================
# 3. CORRELATION HEATMAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


image = ax.imshow(
    corr.values,
    vmin=-1,
    vmax=1,
)


ax.set_xticks(
    np.arange(
        len(FEATURES)
    )
)

ax.set_yticks(
    np.arange(
        len(FEATURES)
    )
)


ax.set_xticklabels(
    FEATURES,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    FEATURES
)


for i in range(
    len(FEATURES)
):

    for j in range(
        len(FEATURES)
    ):

        ax.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
        )


fig.colorbar(
    image,
    ax=ax,
    label="Pearson correlation (r)"
)


ax.set_title(
    "Pearson Correlation Matrix — ALE Training Data"
)


fig.tight_layout()


FIG_OUTPUT = (
    FIG_DIR
    / "pearson_correlation_train.png"
)


fig.savefig(
    FIG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 4. TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STRONGEST PAIRWISE CORRELATIONS"
)

print(
    "=" * 80
)


print(
    pair_df.head(
        10
    )[
        [
            "feature_1",
            "feature_2",
            "pearson_r",
            "abs_pearson_r",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# NOTE HIGHER PAIRWISE CORRELATIONS
#
# This is descriptive only.
# Tella et al. did not specify a hard Pearson-r cutoff
# for automatic predictor removal.
# ============================================================

notable_pairs = pair_df[
    pair_df["abs_pearson_r"]
    >= CORRELATION_NOTE_THRESHOLD
]


print(
    "\n"
    + "=" * 80
)

print(
    f"PAIRWISE CORRELATIONS WITH |r| >= "
    f"{CORRELATION_NOTE_THRESHOLD:.2f}"
)

print(
    "=" * 80
)


if len(notable_pairs) == 0:

    print(
        "\nNone."
    )

else:

    print(
        notable_pairs[
            [
                "feature_1",
                "feature_2",
                "pearson_r",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# VIF SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "VARIANCE INFLATION FACTOR"
)

print(
    "=" * 80
)


print(
    vif_df[
        [
            "feature",
            "vif",
            "interpretation",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# OVERALL DIAGNOSTIC
# ============================================================

max_corr = pair_df.iloc[
    0
]["abs_pearson_r"]


max_corr_row = pair_df.iloc[
    0
]


max_vif = vif_df.iloc[
    0
]["vif"]


max_vif_feature = vif_df.iloc[
    0
]["feature"]


print(
    "\n"
    + "=" * 80
)

print(
    "SUMMARY"
)

print(
    "=" * 80
)


print(
    f"\nLargest absolute pairwise correlation:"
)

print(
    f"  {max_corr_row['feature_1']} "
    f"vs "
    f"{max_corr_row['feature_2']}"
)

print(
    f"  |r| = "
    f"{max_corr:.4f}"
)


print(
    f"\nLargest VIF:"
)

print(
    f"  {max_vif_feature}"
)

print(
    f"  VIF = "
    f"{max_vif:.4f}"
)


print(
    "\nRESULT:"
)


if max_vif < 3.0:

    print(
        "Overall multicollinearity is low: "
        "all predictors have VIF < 3."
    )

elif max_vif < 5.0:

    print(
        "Overall multicollinearity is moderate: "
        "one or more predictors have VIF between 3 and 5."
    )

else:

    print(
        "Elevated multicollinearity detected: "
        "one or more predictors have VIF >= 5."
    )


if len(notable_pairs) > 0:

    print(
        "\nPairwise correlation note:"
    )

    for _, row in notable_pairs.iterrows():

        print(
            f"  {row['feature_1']} vs "
            f"{row['feature_2']}: "
            f"r = {row['pearson_r']:.4f}"
        )

    print(
        "\nThese relationships should be considered "
        "when interpreting ALE, but they do not "
        "by themselves require predictor removal."
    )


print(
    "\nSaved files:"
)

print(
    f"  Correlation matrix:\n"
    f"  {CORR_OUTPUT}"
)

print(
    f"\n  Ranked correlations:\n"
    f"  {PAIR_OUTPUT}"
)

print(
    f"\n  VIF table:\n"
    f"  {VIF_OUTPUT}"
)

print(
    f"\n  Figure:\n"
    f"  {FIG_OUTPUT}"
)