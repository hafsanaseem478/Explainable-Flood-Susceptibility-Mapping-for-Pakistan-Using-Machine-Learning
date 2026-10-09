from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNClassifier

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
# PATHS
# ============================================================

INPUT = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)

RESULT_DIR = Path(
    "results/models/tabpfn_union"
)

RESULT_DIR.mkdir(
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
# METRICS
# ============================================================

def calculate_metrics(y_true, probability):

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probability
        ),

        "pr_auc": average_precision_score(
            y_true,
            probability
        ),

        "accuracy": accuracy_score(
            y_true,
            prediction
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction
        ),

        "precision": precision_score(
            y_true,
            prediction
        ),

        "recall": recall_score(
            y_true,
            prediction
        ),

        "f1": f1_score(
            y_true,
            prediction
        ),

        "brier": brier_score_loss(
            y_true,
            probability
        ),
    }


# ============================================================
# DEVICE CHECK
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU not available. "
        "Run this script in Google Colab "
        "with a GPU runtime."
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT
)

X = df[FEATURES]

y = df["class"].astype(int)


print("=" * 70)
print("TABPFN — UNION FLOOD INVENTORY")
print("=" * 70)

print(
    f"\nSamples: {len(df):,}"
)

print(
    f"Flood: {(y == 1).sum():,}"
)

print(
    f"Background: {(y == 0).sum():,}"
)


# ============================================================
# RANDOM HOLD-OUT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM HOLD-OUT")
print("=" * 70)


train_mask = (
    df["random_split"] == "train"
)

test_mask = (
    df["random_split"] == "test"
)


X_train = X.loc[
    train_mask
]

y_train = y.loc[
    train_mask
]

X_test = X.loc[
    test_mask
]

y_test = y.loc[
    test_mask
]


model = TabPFNClassifier(
    device="cuda",
    random_state=42,
    show_progress_bar=True,
)


print(
    "\nFitting TabPFN..."
)

model.fit(
    X_train,
    y_train
)


print(
    "\nPredicting..."
)

probability = model.predict_proba(
    X_test
)[:, 1]


random_metrics = calculate_metrics(
    y_test,
    probability
)


for metric, value in random_metrics.items():

    print(
        f"{metric:<20}"
        f"{value:.4f}"
    )


pd.DataFrame([
    random_metrics
]).to_csv(
    RESULT_DIR
    / "tabpfn_random_metrics.csv",
    index=False
)


# ============================================================
# CLEAN GPU MEMORY
# ============================================================

del model

torch.cuda.empty_cache()


# ============================================================
# 5-FOLD SPATIAL CV
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD SPATIAL BLOCK CV")
print("=" * 70)


spatial_results = []


for fold in range(1, 6):

    print(
        "\n" + "=" * 70
    )

    print(
        f"FOLD {fold}"
    )

    print(
        "=" * 70
    )


    train_mask = (
        df["spatial_fold"] != fold
    )

    validation_mask = (
        df["spatial_fold"] == fold
    )


    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]


    X_validation = X.loc[
        validation_mask
    ]

    y_validation = y.loc[
        validation_mask
    ]


    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_validation.shape
    )


    model = TabPFNClassifier(
        device="cuda",
        random_state=42,
        show_progress_bar=True,
    )


    print(
        "\nFitting..."
    )

    model.fit(
        X_train,
        y_train
    )


    print(
        "Predicting..."
    )

    probability = model.predict_proba(
        X_validation
    )[:, 1]


    metrics = calculate_metrics(
        y_validation,
        probability
    )


    metrics["fold"] = fold

    metrics["n_validation"] = len(
        y_validation
    )


    spatial_results.append(
        metrics
    )


    print(
        "\nResults:"
    )


    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
    ]:

        print(
            f"{metric:<20}"
            f"{metrics[metric]:.4f}"
        )


    del model

    torch.cuda.empty_cache()


# ============================================================
# SPATIAL SUMMARY
# ============================================================

spatial_df = pd.DataFrame(
    spatial_results
)


spatial_df.to_csv(
    RESULT_DIR
    / "tabpfn_spatial_fold_metrics.csv",
    index=False
)


summary_rows = []


print("\n" + "=" * 70)
print("TABPFN SPATIAL CV MEAN ± SD")
print("=" * 70)


for metric in [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier",
]:

    mean = spatial_df[
        metric
    ].mean()

    sd = spatial_df[
        metric
    ].std(ddof=1)


    print(
        f"{metric:<20}"
        f"{mean:.4f} ± {sd:.4f}"
    )


    summary_rows.append({
        "metric": metric,
        "mean": mean,
        "sd": sd,
    })


pd.DataFrame(
    summary_rows
).to_csv(
    RESULT_DIR
    / "tabpfn_spatial_summary.csv",
    index=False
)


print(
    "\nTabPFN complete."
)