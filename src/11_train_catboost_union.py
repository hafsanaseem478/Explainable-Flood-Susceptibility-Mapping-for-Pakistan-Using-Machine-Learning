from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier

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
    "results/models/catboost_union"
)

MODEL_DIR = Path(
    "models"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
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
# BASELINE CATBOOST
# No tuning yet
# ============================================================

def make_model():

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


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability
):

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
# LOAD DATA
# ============================================================

print("=" * 80)
print("CATBOOST — UNION FLOOD INVENTORY")
print("=" * 80)

df = pd.read_csv(
    INPUT
)

X = df[FEATURES]

y = df["class"].astype(int)


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

print("\n" + "=" * 80)
print("RANDOM HOLD-OUT")
print("=" * 80)


train_mask = (
    df["random_split"] == "train"
)

test_mask = (
    df["random_split"] == "test"
)


model = make_model()


model.fit(
    X.loc[train_mask],
    y.loc[train_mask]
)


probability = model.predict_proba(
    X.loc[test_mask]
)[:, 1]


random_metrics = calculate_metrics(
    y.loc[test_mask],
    probability
)


for metric, value in random_metrics.items():

    print(
        f"{metric:<20}"
        f"{value:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_OUTPUT = (
    MODEL_DIR
    / "catboost_union_random.cbm"
)

model.save_model(
    str(MODEL_OUTPUT)
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance.to_csv(
    RESULT_DIR
    / "random_feature_importance.csv",
    index=False
)


print(
    "\nCatBoost feature importance:"
)


for _, row in importance.iterrows():

    print(
        f"{row['feature']:<22}"
        f"{row['importance']:.4f}"
    )


# ============================================================
# 5-FOLD SPATIAL CV
# ============================================================

print("\n" + "=" * 80)
print("5-FOLD SPATIAL BLOCK CV")
print("=" * 80)


spatial_results = []


for fold in range(1, 6):

    validation_mask = (
        df["spatial_fold"] == fold
    )

    training_mask = (
        df["spatial_fold"] != fold
    )


    spatial_model = make_model()


    spatial_model.fit(
        X.loc[training_mask],
        y.loc[training_mask]
    )


    probability = (
        spatial_model.predict_proba(
            X.loc[validation_mask]
        )[:, 1]
    )


    metrics = calculate_metrics(
        y.loc[validation_mask],
        probability
    )


    metrics["fold"] = fold

    metrics["n_validation"] = int(
        validation_mask.sum()
    )


    spatial_results.append(
        metrics
    )


    print(
        f"\nFold {fold}"
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
            f"  {metric:<18}"
            f"{metrics[metric]:.4f}"
        )


# ============================================================
# SPATIAL SUMMARY
# ============================================================

spatial_df = pd.DataFrame(
    spatial_results
)


spatial_df.to_csv(
    RESULT_DIR
    / "spatial_fold_metrics.csv",
    index=False
)


print("\n" + "-" * 80)
print("SPATIAL CV MEAN ± SD")
print("-" * 80)


summary_rows = []


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
    / "spatial_summary.csv",
    index=False
)


# ============================================================
# SAVE RANDOM METRICS
# ============================================================

pd.DataFrame([
    random_metrics
]).to_csv(
    RESULT_DIR
    / "random_metrics.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CATBOOST COMPLETE")
print("=" * 80)


print(
    f"\nRandom ROC-AUC: "
    f"{random_metrics['roc_auc']:.4f}"
)


print(
    f"Spatial ROC-AUC: "
    f"{spatial_df['roc_auc'].mean():.4f}"
    f" ± "
    f"{spatial_df['roc_auc'].std(ddof=1):.4f}"
)


print(
    f"\nSaved model:\n"
    f"{MODEL_OUTPUT}"
)


print(
    f"\nResults:\n"
    f"{RESULT_DIR}"
)