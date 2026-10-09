from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SECTION_DIR = (
    ROOT
    / "results"
    / "section_3_1"
)

LOCAL_FILE = (
    SECTION_DIR
    / "spatial_oof_predictions_local.csv"
)

TABPFN_FILE = (
    SECTION_DIR
    / "spatial_oof_predictions_tabpfn.csv"
)


# Existing fold metrics
FOLD_METRIC_FILES = {
    "LR":
        ROOT
        / "results/models/lr_union/spatial_fold_metrics.csv",

    "RF":
        ROOT
        / "results/models/rf_union/spatial_fold_metrics.csv",

    "XGBoost":
        ROOT
        / "results/models/xgb_union/spatial_fold_metrics.csv",

    "CatBoost":
        ROOT
        / "results/models/catboost_union/spatial_fold_metrics.csv",

    "TabPFN":
        ROOT
        / "results/models/tabpfn_union/tabpfn_spatial_fold_metrics.csv",
}


# Outputs
MERGED_OUTPUT = (
    SECTION_DIR
    / "spatial_oof_predictions_all5.csv"
)

OVERALL_OUTPUT = (
    SECTION_DIR
    / "spatial_performance_all5.csv"
)

CLASSWISE_OUTPUT = (
    SECTION_DIR
    / "spatial_classwise_metrics_all5.csv"
)

CONFUSION_OUTPUT = (
    SECTION_DIR
    / "spatial_confusion_matrix_all5.csv"
)

FOLD_SUMMARY_OUTPUT = (
    SECTION_DIR
    / "spatial_fold_auc_summary_all5.csv"
)


# ============================================================
# MODEL COLUMNS
# ============================================================

MODELS = {
    "LR": {
        "prob": "lr_prob",
        "pred": "lr_pred",
    },

    "RF": {
        "prob": "rf_prob",
        "pred": "rf_pred",
    },

    "XGBoost": {
        "prob": "xgboost_prob",
        "pred": "xgboost_pred",
    },

    "CatBoost": {
        "prob": "catboost_prob",
        "pred": "catboost_pred",
    },

    "TabPFN": {
        "prob": "tabpfn_prob",
        "pred": "tabpfn_pred",
    },
}


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FINAL FIVE-MODEL SPATIAL PERFORMANCE")
print("=" * 80)

if not LOCAL_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{LOCAL_FILE}"
    )

if not TABPFN_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{TABPFN_FILE}"
    )


local = pd.read_csv(
    LOCAL_FILE
)

tab = pd.read_csv(
    TABPFN_FILE
)


print(
    f"\nLocal rows : {len(local):,}"
)

print(
    f"TabPFN rows: {len(tab):,}"
)


# ============================================================
# STRICT QC BEFORE MERGING
# ============================================================

if len(local) != 20000:
    raise RuntimeError(
        f"Expected 20,000 local rows, "
        f"found {len(local):,}"
    )

if len(tab) != 20000:
    raise RuntimeError(
        f"Expected 20,000 TabPFN rows, "
        f"found {len(tab):,}"
    )

if not local["sample_id"].is_unique:
    raise RuntimeError(
        "Duplicate sample_id in local OOF file."
    )

if not tab["sample_id"].is_unique:
    raise RuntimeError(
        "Duplicate sample_id in TabPFN OOF file."
    )


# ============================================================
# MERGE BY SAMPLE ID
# ============================================================

merged = local.merge(
    tab,
    on="sample_id",
    how="outer",
    suffixes=(
        "_local",
        "_tabpfn",
    ),
    indicator=True,
)


if len(merged) != 20000:
    raise RuntimeError(
        "Merged file does not contain 20,000 samples."
    )

if not (
    merged["_merge"]
    == "both"
).all():

    raise RuntimeError(
        "Some sample IDs did not match."
    )


# ============================================================
# VERIFY TARGET AND FOLD ARE IDENTICAL
# ============================================================

if not np.array_equal(
    merged[
        "y_true_local"
    ].to_numpy(),
    merged[
        "y_true_tabpfn"
    ].to_numpy(),
):

    raise RuntimeError(
        "y_true mismatch between local and TabPFN OOF files."
    )


if not np.array_equal(
    merged[
        "spatial_fold_local"
    ].to_numpy(),
    merged[
        "spatial_fold_tabpfn"
    ].to_numpy(),
):

    raise RuntimeError(
        "spatial_fold mismatch between local and TabPFN files."
    )


merged = (
    merged
    .drop(
        columns=[
            "_merge",
            "y_true_tabpfn",
            "spatial_fold_tabpfn",
        ]
    )
    .rename(
        columns={
            "y_true_local":
                "y_true",

            "spatial_fold_local":
                "spatial_fold",
        }
    )
)


print(
    "\n✓ 20,000 sample IDs matched"
)

print(
    "✓ y_true matches"
)

print(
    "✓ spatial_fold matches"
)


# ============================================================
# FINAL TARGET CHECK
# ============================================================

y = (
    merged[
        "y_true"
    ]
    .astype(int)
    .to_numpy()
)


if np.sum(
    y == 0
) != 10000:

    raise RuntimeError(
        "Expected 10,000 Non-flooded samples."
    )


if np.sum(
    y == 1
) != 10000:

    raise RuntimeError(
        "Expected 10,000 Flooded samples."
    )


print(
    "✓ Non-flooded = 10,000"
)

print(
    "✓ Flooded     = 10,000"
)


# ============================================================
# CALCULATE POOLED OOF PERFORMANCE
# ============================================================

overall_rows = []

classwise_rows = []

confusion_rows = []


for model, cols in MODELS.items():

    probability = (
        merged[
            cols["prob"]
        ]
        .astype(float)
        .to_numpy()
    )

    prediction = (
        merged[
            cols["pred"]
        ]
        .astype(int)
        .to_numpy()
    )


    if np.isnan(
        probability
    ).any():

        raise RuntimeError(
            f"{model}: missing probabilities."
        )


    # --------------------------------------------------------
    # Pooled metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        probability,
    )

    pr_auc = average_precision_score(
        y,
        probability,
    )

    accuracy = accuracy_score(
        y,
        prediction,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        prediction,
    )

    precision = precision_score(
        y,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    brier = brier_score_loss(
        y,
        probability,
    )


    overall_rows.append({
        "model":
            model,

        "pooled_oof_roc_auc":
            roc_auc,

        "pooled_oof_pr_auc":
            pr_auc,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision_flooded":
            precision,

        "recall_flooded":
            recall,

        "f1_flooded":
            f1,

        "brier":
            brier,

        "n":
            len(y),
    })


    # --------------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------------

    (
        class_precision,
        class_recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        y,
        prediction,
        labels=[
            0,
            1,
        ],
        zero_division=0,
    )


    for i, label in enumerate(
        [
            0,
            1,
        ]
    ):

        classwise_rows.append({
            "model":
                model,

            "class_label":
                label,

            "class_name":
                (
                    "Non-flooded"
                    if label == 0
                    else "Flooded"
                ),

            "precision":
                class_precision[i],

            "recall":
                class_recall[i],

            "f1":
                class_f1[i],

            "support":
                int(
                    support[i]
                ),
        })


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            prediction,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )


    confusion_rows.append({
        "model":
            model,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    })


# ============================================================
# FOLD-WISE AUC MEAN ± SD
# ============================================================

fold_summary_rows = []


for model, path in FOLD_METRIC_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold metrics:\n{path}"
        )


    fold_df = pd.read_csv(
        path
    )


    if len(
        fold_df
    ) != 5:

        raise RuntimeError(
            f"{model}: expected 5 fold rows."
        )


    auc_values = (
        fold_df[
            "roc_auc"
        ]
        .astype(float)
    )


    fold_summary_rows.append({
        "model":
            model,

        "fold1_auc":
            auc_values.iloc[0],

        "fold2_auc":
            auc_values.iloc[1],

        "fold3_auc":
            auc_values.iloc[2],

        "fold4_auc":
            auc_values.iloc[3],

        "fold5_auc":
            auc_values.iloc[4],

        "mean_fold_auc":
            auc_values.mean(),

        # Same convention as pandas .std():
        # sample SD across the five folds.
        "sd_fold_auc":
            auc_values.std(
                ddof=1
            ),
    })


# ============================================================
# DATAFRAMES
# ============================================================

overall_df = pd.DataFrame(
    overall_rows
)

classwise_df = pd.DataFrame(
    classwise_rows
)

confusion_df = pd.DataFrame(
    confusion_rows
)

fold_summary_df = pd.DataFrame(
    fold_summary_rows
)


# ============================================================
# MERGE FOLD SUMMARY INTO MAIN TABLE
# ============================================================

overall_df = overall_df.merge(
    fold_summary_df[
        [
            "model",
            "mean_fold_auc",
            "sd_fold_auc",
        ]
    ],
    on="model",
    how="left",
)


# ============================================================
# QC AGAINST ESTABLISHED SPATIAL AUC RESULTS
# ============================================================

EXPECTED_MEAN_AUC = {
    "LR": 0.8395,
    "RF": 0.9157,
    "XGBoost": 0.9159,
    "CatBoost": 0.9177,
    "TabPFN": 0.9102,
}


for model, expected in EXPECTED_MEAN_AUC.items():

    observed = float(
        overall_df.loc[
            overall_df[
                "model"
            ] == model,
            "mean_fold_auc",
        ].iloc[0]
    )


    if abs(
        observed
        - expected
    ) > 0.0001:

        raise RuntimeError(
            f"{model}: fold-mean AUC mismatch.\n"
            f"Expected ≈ {expected:.4f}\n"
            f"Observed = {observed:.4f}"
        )


print(
    "\n✓ Existing five-model spatial AUC results reproduced"
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    MERGED_OUTPUT,
    index=False,
)

overall_df.to_csv(
    OVERALL_OUTPUT,
    index=False,
)

classwise_df.to_csv(
    CLASSWISE_OUTPUT,
    index=False,
)

confusion_df.to_csv(
    CONFUSION_OUTPUT,
    index=False,
)

fold_summary_df.to_csv(
    FOLD_SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FINAL SPATIAL PERFORMANCE — POOLED OOF"
)

print(
    "=" * 100
)


print(
    overall_df[
        [
            "model",
            "pooled_oof_roc_auc",
            "accuracy",
            "precision_flooded",
            "recall_flooded",
            "f1_flooded",
            "mean_fold_auc",
            "sd_fold_auc",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "CLASS-WISE SPATIAL PERFORMANCE"
)

print(
    "=" * 100
)


print(
    classwise_df[
        [
            "model",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OVERALL_OUTPUT
)

print(
    CLASSWISE_OUTPUT
)

print(
    CONFUSION_OUTPUT
)

print(
    FOLD_SUMMARY_OUTPUT
)

print(
    MERGED_OUTPUT
)