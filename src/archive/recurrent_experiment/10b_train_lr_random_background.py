from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

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
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_random_background_with_splits.csv"
)

RESULT_DIR = Path("results/models/lr")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def make_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_SEED
            )
        )
    ])


def calculate_metrics(
    y_true,
    probability,
    prediction
):

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),

        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "brier":
            brier_score_loss(
                y_true,
                probability
            ),
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 80)

df = pd.read_csv(INPUT)

X = df[FEATURES]
y = df["class"]


# ============================================================
# 1. RANDOM 77/23 VALIDATION
# ============================================================

print("\n[1/2] Random 77/23 validation...")

train_mask = (
    df["random_split"] == "train"
)

test_mask = (
    df["random_split"] == "test"
)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]


random_model = make_model()

random_model.fit(
    X_train,
    y_train
)

random_probability = (
    random_model.predict_proba(
        X_test
    )[:, 1]
)

random_prediction = (
    random_model.predict(
        X_test
    )
)

random_metrics = calculate_metrics(
    y_test,
    random_probability,
    random_prediction
)


print("\nRandom validation metrics:")

for metric, value in random_metrics.items():

    print(
        f"  {metric:<20} "
        f"{value:.4f}"
    )


# ============================================================
# SAVE RANDOM PREDICTIONS
# ============================================================

random_predictions = df.loc[
    test_mask,
    [
        "sample_id",
        "class",
        "x",
        "y",
        "row",
        "col",
    ]
].copy()

random_predictions[
    "probability"
] = random_probability

random_predictions[
    "prediction"
] = random_prediction

random_predictions.to_csv(
    RESULT_DIR
    / "random_predictions.csv",
    index=False
)


# ============================================================
# SAVE RANDOM-SPLIT MODEL
# ============================================================

joblib.dump(
    random_model,
    MODEL_DIR / "lr_random_background.joblib"
)


# ============================================================
# 2. SPATIAL 5-FOLD VALIDATION
# ============================================================

print(
    "\n[2/2] Spatial 5-fold validation..."
)

spatial_records = []
spatial_predictions = []


for fold in range(1, 6):

    print(f"\nFold {fold}")

    train_mask = (
        df["spatial_fold"] != fold
    )

    test_mask = (
        df["spatial_fold"] == fold
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]


    # IMPORTANT:
    # new model + new scaler for every fold
    model = make_model()

    model.fit(
        X_train,
        y_train
    )

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        model.predict(
            X_test
        )
    )

    metrics = calculate_metrics(
        y_test,
        probability,
        prediction
    )

    metrics["fold"] = fold
    metrics["n_test"] = len(y_test)

    metrics["n_flood"] = int(
        (y_test == 1).sum()
    )

    metrics["n_nonflood"] = int(
        (y_test == 0).sum()
    )

    spatial_records.append(
        metrics
    )


    fold_predictions = df.loc[
        test_mask,
        [
            "sample_id",
            "class",
            "x",
            "y",
            "row",
            "col",
            "spatial_block",
            "spatial_fold",
        ]
    ].copy()

    fold_predictions[
        "probability"
    ] = probability

    fold_predictions[
        "prediction"
    ] = prediction

    spatial_predictions.append(
        fold_predictions
    )


    print(
        f"  ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:  "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"  Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )


# ============================================================
# SPATIAL RESULTS
# ============================================================

spatial_df = pd.DataFrame(
    spatial_records
)

spatial_df.to_csv(
    RESULT_DIR
    / "spatial_fold_metrics.csv",
    index=False
)


all_spatial_predictions = pd.concat(
    spatial_predictions,
    ignore_index=True
)

all_spatial_predictions.to_csv(
    RESULT_DIR
    / "spatial_predictions.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metric_columns = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier",
]


summary_rows = []


# Random results
random_row = {
    "validation":
        "Random 77/23"
}

random_row.update(
    random_metrics
)

summary_rows.append(
    random_row
)


# Spatial mean
spatial_mean = {
    "validation":
        "Spatial CV mean"
}

for metric in metric_columns:

    spatial_mean[metric] = (
        spatial_df[metric].mean()
    )

summary_rows.append(
    spatial_mean
)


# Spatial standard deviation
spatial_std = {
    "validation":
        "Spatial CV std"
}

for metric in metric_columns:

    spatial_std[metric] = (
        spatial_df[metric].std()
    )

summary_rows.append(
    spatial_std
)


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    RESULT_DIR
    / "lr_summary.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 80)

print("\nRandom 77/23:")

for metric in metric_columns:

    print(
        f"  {metric:<20} "
        f"{random_metrics[metric]:.4f}"
    )


print("\nSpatial 5-fold mean ± SD:")

for metric in metric_columns:

    mean = spatial_df[
        metric
    ].mean()

    std = spatial_df[
        metric
    ].std()

    print(
        f"  {metric:<20} "
        f"{mean:.4f} ± {std:.4f}"
    )


print(
    "\nSaved results to:\n"
    f"{RESULT_DIR}"
)

print(
    "\nSaved random-split model to:\n"
    f"{MODEL_DIR / 'lr_random_background.joblib'}"
)