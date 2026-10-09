from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


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
# EXACT MODEL PREDICTORS
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
# LOAD FULL MODELLING DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing:

    raise ValueError(
        "Missing predictors:\n"
        + "\n".join(
            missing
        )
    )


X = df[
    FEATURES
].copy()


# ============================================================
# BASIC QC
# ============================================================

if X.isna().any().any():

    bad = X.columns[
        X.isna().any()
    ].tolist()

    raise ValueError(
        "Missing values detected in:\n"
        + "\n".join(
            bad
        )
    )


if not np.isfinite(
    X.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Non-finite predictor values detected."
    )


print("=" * 80)
print("FULL-DATA MULTICOLLINEARITY DIAGNOSTIC")
print("PEARSON CORRELATION + VIF")
print("=" * 80)


print(
    f"\nObservations: "
    f"{len(X):,}"
)

print(
    f"Predictors: "
    f"{len(FEATURES)}"
)


if len(X) != 20_000:

    print(
        "\nWARNING:"
        f" expected 20,000 samples,"
        f" found {len(X):,}."
    )


# ============================================================
# 1. PEARSON CORRELATION MATRIX
# ============================================================

corr = X.corr(
    method="pearson"
)


CORR_OUTPUT = (
    OUT_DIR
    / "pearson_correlation_full.csv"
)


corr.to_csv(
    CORR_OUTPUT
)


# ============================================================
# 2. UNIQUE PAIRWISE CORRELATIONS
# ============================================================

pair_rows = []


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


        pair_rows.append({
            "feature_1":
                feature_1,

            "feature_2":
                feature_2,

            "pearson_r":
                float(r),

            "abs_pearson_r":
                float(abs(r)),
        })


pair_df = pd.DataFrame(
    pair_rows
)


pair_df = pair_df.sort_values(
    "abs_pearson_r",
    ascending=False
).reset_index(
    drop=True
)


PAIR_OUTPUT = (
    OUT_DIR
    / "pearson_pairwise_ranked_full.csv"
)


pair_df.to_csv(
    PAIR_OUTPUT,
    index=False
)


# ============================================================
# 3. VARIANCE INFLATION FACTOR
#
# Standard definition:
#
# VIF_j = 1 / (1 - R²_j)
#
# For each predictor j:
# regress predictor j against all remaining predictors.
#
# IMPORTANT:
# We use the original predictor values here.
# No scaling or transformation is applied.
# ============================================================

vif_rows = []


X_values = X.to_numpy(
    dtype=float
)


for i, feature in enumerate(
    FEATURES
):

    y_feature = X_values[
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


    X_other = X_values[
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


    # Numerical protection only
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


    vif_rows.append({
        "feature":
            feature,

        "r_squared_against_other_predictors":
            float(r_squared),

        "vif":
            float(vif),
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
    / "vif_full.csv"
)


vif_df.to_csv(
    VIF_OUTPUT,
    index=False
)


# ============================================================
# 4. CORRELATION HEATMAP
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
    "Pearson Correlation Matrix — Full Modelling Dataset"
)


fig.tight_layout()


FIG_OUTPUT = (
    FIG_DIR
    / "pearson_correlation_full.png"
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
# 5. CHECK CORRELATION MATRIX INTEGRITY
# ============================================================

if not np.allclose(
    corr.values,
    corr.values.T,
    atol=1e-12
):

    raise RuntimeError(
        "Correlation matrix is not symmetric."
    )


if not np.allclose(
    np.diag(
        corr.values
    ),
    1.0,
    atol=1e-12
):

    raise RuntimeError(
        "Correlation matrix diagonal is not 1."
    )


expected_pairs = (
    len(FEATURES)
    * (
        len(FEATURES) - 1
    )
    // 2
)


if len(pair_df) != expected_pairs:

    raise RuntimeError(
        f"Expected {expected_pairs} unique pairs, "
        f"found {len(pair_df)}."
    )


# ============================================================
# 6. PRINT RESULTS
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
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


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
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 7. PURELY DESCRIPTIVE SUMMARY
#
# No arbitrary Pearson cut-off is used.
# No predictors are removed automatically.
# ============================================================

max_corr_row = pair_df.iloc[
    0
]


max_vif_row = vif_df.iloc[
    0
]


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
    "\nStrongest pairwise relationship:"
)

print(
    f"  {max_corr_row['feature_1']} "
    f"vs "
    f"{max_corr_row['feature_2']}"
)

print(
    f"  r = "
    f"{max_corr_row['pearson_r']:.6f}"
)


print(
    "\nLargest VIF:"
)

print(
    f"  {max_vif_row['feature']}"
)

print(
    f"  VIF = "
    f"{max_vif_row['vif']:.6f}"
)


print(
    "\nVIF values:"
)


for _, row in vif_df.iterrows():

    print(
        f"  {row['feature']:<22}"
        f"{row['vif']:.6f}"
    )


print(
    "\nNo predictors were removed or modified."
)


print(
    "\nSaved files:"
)

print(
    f"  {CORR_OUTPUT}"
)

print(
    f"  {PAIR_OUTPUT}"
)

print(
    f"  {VIF_OUTPUT}"
)

print(
    f"  {FIG_OUTPUT}"
)