from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
    "Data/samples/training_samples_with_splits.csv"
)

OUT_DIR = Path(
    "results/models/lr/diagnostics"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("DIAGNOSING HIGH LOGISTIC REGRESSION PERFORMANCE")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

train = df[
    df["random_split"] == "train"
].copy()

test = df[
    df["random_split"] == "test"
].copy()

y_train = train["class"]
y_test = test["class"]


print(f"\nTraining samples: {len(train):,}")
print(f"Testing samples:  {len(test):,}")


# ============================================================
# 1. SINGLE-FEATURE LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("SINGLE-FEATURE TEST AUC")
print("=" * 80)

single_results = []


for feature in FEATURES:

    X_train = train[[feature]]
    X_test = test[[feature]]

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_SEED
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probability
    )

    coefficient = (
        model.named_steps["lr"]
        .coef_[0][0]
    )

    single_results.append({
        "feature": feature,
        "roc_auc": auc,
        "standardized_coefficient": coefficient,
        "direction":
            "higher → more flood"
            if coefficient > 0
            else "higher → less flood"
    })


single_df = pd.DataFrame(
    single_results
).sort_values(
    "roc_auc",
    ascending=False
)


for _, row in single_df.iterrows():

    print(
        f"{row['feature']:<22}"
        f"AUC={row['roc_auc']:.4f}   "
        f"coef={row['standardized_coefficient']:+.4f}   "
        f"{row['direction']}"
    )


single_df.to_csv(
    OUT_DIR
    / "single_feature_auc.csv",
    index=False
)


# ============================================================
# 2. FULL MULTIVARIABLE LR
# ============================================================

print("\n" + "=" * 80)
print("FULL LR STANDARDIZED COEFFICIENTS")
print("=" * 80)


X_train = train[FEATURES]
X_test = test[FEATURES]


full_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lr",
        LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_SEED
        )
    )
])


full_model.fit(
    X_train,
    y_train
)


probability = full_model.predict_proba(
    X_test
)[:, 1]


full_auc = roc_auc_score(
    y_test,
    probability
)


coefficients = (
    full_model
    .named_steps["lr"]
    .coef_[0]
)


coef_df = pd.DataFrame({
    "feature": FEATURES,
    "standardized_coefficient":
        coefficients
})


coef_df[
    "absolute_coefficient"
] = np.abs(
    coef_df[
        "standardized_coefficient"
    ]
)


coef_df[
    "direction"
] = np.where(
    coef_df[
        "standardized_coefficient"
    ] > 0,
    "higher → more flood",
    "higher → less flood"
)


coef_df = coef_df.sort_values(
    "absolute_coefficient",
    ascending=False
)


print(
    f"\nFull model test ROC-AUC: "
    f"{full_auc:.4f}\n"
)


for _, row in coef_df.iterrows():

    print(
        f"{row['feature']:<22}"
        f"coef="
        f"{row['standardized_coefficient']:+.4f}   "
        f"{row['direction']}"
    )


coef_df.to_csv(
    OUT_DIR
    / "full_lr_coefficients.csv",
    index=False
)


# ============================================================
# 3. SIMPLE CLASS DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 80)
print("CLASS MEDIANS")
print("=" * 80)


median_rows = []

for feature in FEATURES:

    nonflood = df.loc[
        df["class"] == 0,
        feature
    ]

    flood = df.loc[
        df["class"] == 1,
        feature
    ]

    nf_med = nonflood.median()
    fl_med = flood.median()

    median_rows.append({
        "feature": feature,
        "nonflood_median": nf_med,
        "flood_median": fl_med
    })

    print(
        f"{feature:<22}"
        f"non-flood={nf_med:>12.4f}   "
        f"flood={fl_med:>12.4f}"
    )


pd.DataFrame(
    median_rows
).to_csv(
    OUT_DIR
    / "class_medians.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    f"\nFull LR ROC-AUC: "
    f"{full_auc:.4f}"
)

print(
    "\nSaved diagnostics to:\n"
    f"{OUT_DIR}"
)