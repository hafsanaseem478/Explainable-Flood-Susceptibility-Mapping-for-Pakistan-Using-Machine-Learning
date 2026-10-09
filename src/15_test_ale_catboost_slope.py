from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier


# ============================================================
# SETTINGS
# ============================================================

FEATURE = "slope"

N_BINS = 20


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)

MODEL_PATH = Path(
    "models/catboost_union_random.cbm"
)

OUT_DIR = Path(
    "results/ale"
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
# FEATURES — EXACT MODEL ORDER
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
    INPUT
)


# ALE reference data:
# use only the data used to train the saved random-split model

train_df = df[
    df["random_split"] == "train"
].copy()


X = train_df[
    FEATURES
].copy()


print("=" * 80)
print("ALE TEST — CATBOOST + SLOPE")
print("=" * 80)

print(
    f"\nTraining samples used for ALE: "
    f"{len(X):,}"
)

print(
    f"Slope range: "
    f"{X[FEATURE].min():.4f}° "
    f"to "
    f"{X[FEATURE].max():.4f}°"
)


# ============================================================
# LOAD SAVED CATBOOST MODEL
# ============================================================

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)

print(
    "✓ Saved CatBoost model loaded"
)


# ============================================================
# CREATE QUANTILE BIN EDGES
# ============================================================

quantiles = np.linspace(
    0,
    1,
    N_BINS + 1
)


edges = np.quantile(
    X[FEATURE],
    quantiles
)


# Remove duplicate edges.
# This matters because slope has many nearly identical
# low values.

edges = np.unique(
    edges
)


if len(edges) < 3:

    raise RuntimeError(
        "Not enough unique values "
        "to calculate ALE."
    )


actual_bins = (
    len(edges) - 1
)


print(
    f"Requested bins: {N_BINS}"
)

print(
    f"Actual unique bins: "
    f"{actual_bins}"
)


# ============================================================
# ASSIGN EACH SAMPLE TO A BIN
# ============================================================

values = X[
    FEATURE
].to_numpy()


bin_ids = np.searchsorted(
    edges,
    values,
    side="right"
) - 1


# Maximum value belongs to final bin

bin_ids = np.clip(
    bin_ids,
    0,
    actual_bins - 1
)


# ============================================================
# CALCULATE LOCAL EFFECT WITHIN EACH BIN
#
# For samples in one bin:
#
# predict once with slope = lower edge
# predict once with slope = upper edge
#
# Difference = local model response to slope
# ============================================================

local_effects = []

bin_counts = []

bin_centers = []


for i in range(
    actual_bins
):

    mask = (
        bin_ids == i
    )

    X_bin = X.loc[
        mask
    ].copy()


    n = len(
        X_bin
    )


    if n == 0:

        local_effects.append(
            np.nan
        )

        bin_counts.append(
            0
        )

        bin_centers.append(
            (
                edges[i]
                + edges[i + 1]
            )
            / 2
        )

        continue


    lower = edges[i]

    upper = edges[i + 1]


    X_low = X_bin.copy()

    X_high = X_bin.copy()


    X_low[
        FEATURE
    ] = lower

    X_high[
        FEATURE
    ] = upper


    p_low = model.predict_proba(
        X_low
    )[:, 1]


    p_high = model.predict_proba(
        X_high
    )[:, 1]


    effect = np.mean(
        p_high - p_low
    )


    local_effects.append(
        effect
    )

    bin_counts.append(
        n
    )

    bin_centers.append(
        (
            lower
            + upper
        )
        / 2
    )


local_effects = np.asarray(
    local_effects,
    dtype=float
)

bin_counts = np.asarray(
    bin_counts,
    dtype=int
)

bin_centers = np.asarray(
    bin_centers,
    dtype=float
)


# ============================================================
# ACCUMULATE LOCAL EFFECTS
# ============================================================

if np.isnan(
    local_effects
).any():

    raise RuntimeError(
        "Empty ALE bins detected."
    )


# ALE value at each bin:
# cumulative sum of local effects

ale_values = np.cumsum(
    local_effects
)


# ============================================================
# CENTER ALE AROUND ZERO
#
# ALE is relative, not an absolute probability.
# Zero = average model prediction level.
# Positive = increases susceptibility relative to average.
# Negative = decreases susceptibility relative to average.
# ============================================================

weighted_mean = np.average(
    ale_values,
    weights=bin_counts
)


ale_centered = (
    ale_values
    - weighted_mean
)


# ============================================================
# SAVE TABLE
# ============================================================

result = pd.DataFrame({
    "bin": np.arange(
        1,
        actual_bins + 1
    ),

    "lower_slope_deg":
        edges[:-1],

    "upper_slope_deg":
        edges[1:],

    "slope_midpoint_deg":
        bin_centers,

    "n_samples":
        bin_counts,

    "local_effect":
        local_effects,

    "ale":
        ale_centered,
})


CSV_OUTPUT = (
    OUT_DIR
    / "catboost_slope_ale_test.csv"
)


result.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# PRINT TABLE
# ============================================================

print(
    "\nALE results:\n"
)


print(
    result[
        [
            "lower_slope_deg",
            "upper_slope_deg",
            "n_samples",
            "ale",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    result["slope_midpoint_deg"],
    result["ale"],
    marker="o"
)


plt.axhline(
    0,
    linewidth=1
)


plt.xlabel(
    "Slope (degrees)"
)

plt.ylabel(
    "ALE on predicted flood probability"
)

plt.title(
    "CatBoost Accumulated Local Effect — Slope"
)


plt.tight_layout()


FIG_OUTPUT = (
    FIG_DIR
    / "ale_test_catboost_slope.png"
)


plt.savefig(
    FIG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SIMPLE DIAGNOSTICS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "ALE TEST COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nMinimum samples in a bin: "
    f"{bin_counts.min():,}"
)

print(
    f"Maximum samples in a bin: "
    f"{bin_counts.max():,}"
)


min_idx = np.argmin(
    ale_centered
)

max_idx = np.argmax(
    ale_centered
)


print(
    f"\nLowest ALE:"
    f"\n  slope ≈ "
    f"{bin_centers[min_idx]:.4f}°"
    f"\n  ALE = "
    f"{ale_centered[min_idx]:.4f}"
)


print(
    f"\nHighest ALE:"
    f"\n  slope ≈ "
    f"{bin_centers[max_idx]:.4f}°"
    f"\n  ALE = "
    f"{ale_centered[max_idx]:.4f}"
)


print(
    f"\nSaved table:\n"
    f"{CSV_OUTPUT}"
)


print(
    f"\nSaved figure:\n"
    f"{FIG_OUTPUT}"
)