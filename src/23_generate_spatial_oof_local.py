from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "Data"
    / "samples"
    / "training_samples_union_with_splits.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "section_3_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "spatial_oof_predictions_local.csv"
)


# Existing results used ONLY for QC.
EXISTING_METRICS = {
    "LR":
        ROOT / "results/models/lr_union/spatial_fold_metrics.csv",

    "RF":
        ROOT / "results/models/rf_union/spatial_fold_metrics.csv",

    "XGBoost":
        ROOT / "results/models/xgb_union/spatial_fold_metrics.csv",

    "CatBoost":
        ROOT / "results/models/catboost_union/spatial_fold_metrics.csv",
}


# ============================================================
# EXACT PREDICTORS USED IN MAIN STUDY
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

print("=" * 80)
print("SPATIAL OOF PREDICTIONS — LOCAL MODELS")
print("=" * 80)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(f"\nLoaded: {len(df):,} samples")


# ============================================================
# BASIC QC
# ============================================================

if len(df) != 20000:
    raise RuntimeError(
        f"Expected 20,000 samples, found {len(df):,}"
    )

if "sample_id" not in df.columns:
    raise RuntimeError(
        "sample_id column is missing."
    )

if not df["sample_id"].is_unique:
    raise RuntimeError(
        "sample_id is not unique."
    )

if "spatial_fold" not in df.columns:
    raise RuntimeError(
        "spatial_fold column is missing."
    )


# ============================================================
# FIND TARGET COLUMN SAFELY
# ============================================================

TARGET_CANDIDATES = [
    "y",
    "target",
    "label",
    "class",
    "flood",
    "flooded",
    "flood_label",
]


target_col = None

for candidate in TARGET_CANDIDATES:

    if candidate not in df.columns:
        continue

    values = set(
        pd.Series(df[candidate])
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if values.issubset({0, 1}) and len(values) == 2:
        target_col = candidate
        break


if target_col is None:

    raise RuntimeError(
        "\nCould not identify the binary target column.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


print(f"Target column: {target_col}")


# ============================================================
# FEATURE QC
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise RuntimeError(
        "Missing predictor columns:\n"
        f"{missing_features}"
    )


X = df[FEATURES].copy()

y = (
    df[target_col]
    .astype(int)
    .copy()
)


if X.isna().any().any():
    raise RuntimeError(
        "Missing values found in predictor matrix."
    )


print(
    f"Class 0: {(y == 0).sum():,}"
)

print(
    f"Class 1: {(y == 1).sum():,}"
)


if (y == 0).sum() != 10000:
    raise RuntimeError(
        "Expected exactly 10,000 class-0 samples."
    )

if (y == 1).sum() != 10000:
    raise RuntimeError(
        "Expected exactly 10,000 class-1 samples."
    )


# ============================================================
# CHECK SPATIAL FOLDS
# ============================================================

expected_fold_counts = {
    1: 3928,
    2: 3940,
    3: 4049,
    4: 4082,
    5: 4001,
}


observed_fold_counts = (
    df["spatial_fold"]
    .value_counts()
    .sort_index()
    .to_dict()
)


print("\nSpatial fold counts:")

for fold in range(1, 6):

    observed = int(
        observed_fold_counts.get(
            fold,
            0
        )
    )

    expected = expected_fold_counts[fold]

    print(
        f"Fold {fold}: "
        f"{observed:,}"
    )

    if observed != expected:

        raise RuntimeError(
            f"\nFold {fold} contains "
            f"{observed:,} samples, "
            f"but current union experiment expects "
            f"{expected:,}.\n"
            "STOP — wrong dataset/fold assignments."
        )


print(
    "\n✓ Fold assignments match current union experiment"
)


# ============================================================
# MODEL FACTORIES
# ============================================================

def make_lr():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def make_rf():

    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def make_xgb():

    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def make_catboost():

    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


MODEL_FACTORIES = {
    "LR": make_lr,
    "RF": make_rf,
    "XGBoost": make_xgb,
    "CatBoost": make_catboost,
}


# ============================================================
# LOAD EXISTING FOLD RESULTS FOR QC
# ============================================================

existing_auc = {}

for model_name, metric_file in EXISTING_METRICS.items():

    if not metric_file.exists():

        raise FileNotFoundError(
            f"Missing existing metric file:\n"
            f"{metric_file}"
        )

    tmp = pd.read_csv(
        metric_file
    )

    required = {
        "fold",
        "roc_auc",
        "n_validation",
    }

    if not required.issubset(
        tmp.columns
    ):

        raise RuntimeError(
            f"{metric_file} is missing required columns."
        )


    existing_auc[model_name] = {
        int(row["fold"]):
            float(row["roc_auc"])

        for _, row
        in tmp.iterrows()
    }


# ============================================================
# OUTPUT BASE
# ============================================================

oof_df = pd.DataFrame({
    "sample_id":
        df["sample_id"],

    "y_true":
        y,

    "spatial_fold":
        df["spatial_fold"].astype(int),
})


# ============================================================
# RUN EACH MODEL
# ============================================================

qc_rows = []


for model_name, make_model in MODEL_FACTORIES.items():

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{model_name} — 5-FOLD SPATIAL OOF"
    )

    print(
        "=" * 80
    )


    oof_probability = np.full(
        len(df),
        np.nan,
        dtype=float,
    )


    oof_prediction = np.full(
        len(df),
        -1,
        dtype=int,
    )


    for fold in range(1, 6):

        validation_mask = (
            df["spatial_fold"]
            == fold
        )

        training_mask = (
            df["spatial_fold"]
            != fold
        )


        X_train = X.loc[
            training_mask
        ]

        y_train = y.loc[
            training_mask
        ]

        X_validation = X.loc[
            validation_mask
        ]

        y_validation = y.loc[
            validation_mask
        ]


        print(
            f"\nFold {fold}"
            f" | train={len(X_train):,}"
            f" | validation={len(X_validation):,}"
        )


        model = make_model()


        model.fit(
            X_train,
            y_train,
        )


        probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )


        prediction = (
            probability >= 0.5
        ).astype(int)


        # ----------------------------------------
        # Reproduce fold AUC
        # ----------------------------------------

        new_auc = roc_auc_score(
            y_validation,
            probability,
        )


        old_auc = (
            existing_auc[
                model_name
            ][fold]
        )


        difference = abs(
            new_auc
            - old_auc
        )


        print(
            f"Existing AUC : {old_auc:.10f}"
        )

        print(
            f"New AUC      : {new_auc:.10f}"
        )

        print(
            f"Difference   : {difference:.10f}"
        )


        qc_rows.append({
            "model":
                model_name,

            "fold":
                fold,

            "existing_auc":
                old_auc,

            "new_auc":
                new_auc,

            "absolute_difference":
                difference,

            "n_validation":
                int(
                    validation_mask.sum()
                ),
        })


        # ----------------------------------------
        # STRICT QC
        # ----------------------------------------

        if difference > 1e-6:

            raise RuntimeError(
                "\n"
                + "!" * 70
                + "\n"
                + f"{model_name} fold {fold} "
                + "DOES NOT reproduce the original result.\n\n"
                + f"Existing AUC = {old_auc:.10f}\n"
                + f"New AUC      = {new_auc:.10f}\n"
                + f"Difference   = {difference:.10f}\n\n"
                + "STOPPED. No final OOF file will be accepted.\n"
                + "!" * 70
            )


        # ----------------------------------------
        # Save held-out predictions in their
        # original sample positions
        # ----------------------------------------

        validation_indices = np.where(
            validation_mask.to_numpy()
        )[0]


        oof_probability[
            validation_indices
        ] = probability


        oof_prediction[
            validation_indices
        ] = prediction


    # ========================================================
    # MODEL-WIDE OOF QC
    # ========================================================

    if np.isnan(
        oof_probability
    ).any():

        raise RuntimeError(
            f"{model_name}: some samples never received "
            "an OOF probability."
        )


    if np.any(
        oof_prediction < 0
    ):

        raise RuntimeError(
            f"{model_name}: some samples never received "
            "an OOF class prediction."
        )


    # Every observation must have exactly one OOF prediction.
    if len(
        oof_probability
    ) != 20000:

        raise RuntimeError(
            f"{model_name}: incorrect OOF length."
        )


    col_prefix = {
        "LR":
            "lr",

        "RF":
            "rf",

        "XGBoost":
            "xgboost",

        "CatBoost":
            "catboost",
    }[model_name]


    oof_df[
        f"{col_prefix}_prob"
    ] = oof_probability


    oof_df[
        f"{col_prefix}_pred"
    ] = oof_prediction


    pooled_auc = roc_auc_score(
        y,
        oof_probability,
    )


    print(
        f"\n{model_name} pooled OOF AUC: "
        f"{pooled_auc:.4f}"
    )


# ============================================================
# FINAL QC
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL OOF QC"
)

print(
    "=" * 80
)


expected_columns = [
    "sample_id",
    "y_true",
    "spatial_fold",
    "lr_prob",
    "lr_pred",
    "rf_prob",
    "rf_pred",
    "xgboost_prob",
    "xgboost_pred",
    "catboost_prob",
    "catboost_pred",
]


if list(
    oof_df.columns
) != expected_columns:

    raise RuntimeError(
        "Unexpected final OOF column structure."
    )


if len(
    oof_df
) != 20000:

    raise RuntimeError(
        "Final OOF file does not contain exactly 20,000 rows."
    )


if not oof_df[
    "sample_id"
].is_unique:

    raise RuntimeError(
        "Final OOF sample_id is not unique."
    )


if oof_df.isna().any().any():

    raise RuntimeError(
        "Final OOF file contains missing values."
    )


print(
    "✓ 20,000 samples"
)

print(
    "✓ each sample has one held-out prediction"
)

print(
    "✓ no missing probabilities"
)

print(
    "✓ no duplicate sample IDs"
)

print(
    "✓ all fold AUCs reproduce saved results"
)


# ============================================================
# SAVE QC TABLE
# ============================================================

qc_df = pd.DataFrame(
    qc_rows
)


QC_FILE = (
    OUTPUT_DIR
    / "spatial_oof_local_reproduction_qc.csv"
)


qc_df.to_csv(
    QC_FILE,
    index=False,
)


# ============================================================
# SAVE FINAL LOCAL OOF PREDICTIONS
# ============================================================

oof_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\n"
    + "=" * 80
)

print(
    "SUCCESS"
)

print(
    "=" * 80
)

print(
    f"\nOOF predictions:\n{OUTPUT_FILE}"
)

print(
    f"\nReproduction QC:\n{QC_FILE}"
)