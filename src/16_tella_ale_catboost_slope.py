from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skexplain

from catboost import CatBoostClassifier


# ============================================================
# TELLA ET AL. (2026) ALE SETTINGS
# ============================================================

FEATURE = "slope"

N_BINS = 20
N_BOOTSTRAP = 150
SUBSAMPLE = 10_000
RANDOM_SEED = 42


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
    "results/ale/tella_protocol"
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
# EXACT MODEL FEATURE ORDER
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

df = pd.read_csv(INPUT)


# ============================================================
# TELLA:
# ALE COMPUTED USING TRAINING DATA
# ============================================================

train_df = df[
    df["random_split"] == "train"
].copy()


X_train = train_df[
    FEATURES
].copy()


y_train = (
    train_df["class"]
    .astype(int)
    .to_numpy()
)


print("=" * 80)
print("TELLA-STYLE FIRST-ORDER ALE")
print("CATBOOST — SLOPE TEST")
print("=" * 80)


print(
    f"\nTraining observations: "
    f"{len(X_train):,}"
)

print(
    f"Features: "
    f"{X_train.shape[1]}"
)

print(
    f"Slope range: "
    f"{X_train[FEATURE].min():.4f}° "
    f"to "
    f"{X_train[FEATURE].max():.4f}°"
)


# ============================================================
# LOAD THE ALREADY-FITTED CATBOOST MODEL
# ============================================================

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)


print(
    "✓ Saved CatBoost model loaded"
)


# ============================================================
# CREATE SKEXPLAIN TOOLKIT
#
# Same ALE library used by Tella et al.
# ============================================================

explainer = skexplain.ExplainToolkit(
    estimators=(
        "CatBoost",
        model
    ),
    X=X_train,
    y=y_train,
)


# ============================================================
# FIRST-ORDER ALE
#
# TELLA SETTINGS:
#   20 intervals
#   150 bootstrap repetitions
#   10,000-instance subsample
# ============================================================

print(
    "\nComputing ALE..."
)

print(
    f"  Feature:       {FEATURE}"
)

print(
    f"  Intervals:     {N_BINS}"
)

print(
    f"  Bootstraps:    {N_BOOTSTRAP}"
)

print(
    f"  Subsample:     {SUBSAMPLE:,}"
)


ale_ds = explainer.ale(
    features=[FEATURE],
    n_bins=N_BINS,
    n_bootstrap=N_BOOTSTRAP,
    subsample=SUBSAMPLE,
    n_jobs=1,
    random_seed=RANDOM_SEED,
    class_index=1,
)


print(
    "\n✓ ALE calculation complete"
)


# ============================================================
# SAVE RAW ALE RESULT
#
# Pickle preserves the complete xarray/skexplain object.
# NetCDF is avoided because the scipy NetCDF backend cannot
# serialize some Unicode metadata produced by skexplain.
# ============================================================

import pickle


PKL_OUTPUT = (
    OUT_DIR
    / "catboost_slope_ale.pkl"
)


with open(
    PKL_OUTPUT,
    "wb"
) as f:

    pickle.dump(
        ale_ds,
        f
    )


print(
    "✓ Raw ALE object saved"
)


# ============================================================
# PLOT USING SKEXPLAIN
#
# This also displays bootstrap uncertainty.
# ============================================================

fig, axes = explainer.plot_ale(
    ale=ale_ds,
    features=[FEATURE],
)


FIG_OUTPUT = (
    FIG_DIR
    / "catboost_slope_tella_protocol.png"
)


fig.savefig(
    FIG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("TELLA-STYLE ALE TEST COMPLETE")
print("=" * 80)


print(
    "\nMethod:"
)

print(
    "  First-order ALE"
)

print(
    "  Training data only"
)

print(
    f"  {N_BINS} percentile intervals"
)

print(
    f"  {N_BOOTSTRAP} bootstrap repetitions"
)

print(
    f"  {SUBSAMPLE:,}-instance subsample"
)

print(
    "  Library: scikit-explain"
)


print(
    f"\nRaw ALE result:\n"
    f"{PKL_OUTPUT}"
)


print(
    f"\nFigure:\n"
    f"{FIG_OUTPUT}"
)


print(
    "\nALE dataset structure:"
)

print(
    ale_ds
)