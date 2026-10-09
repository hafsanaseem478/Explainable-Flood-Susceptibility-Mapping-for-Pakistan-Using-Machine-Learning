# ============================================================
# 21_tella_31_local_metrics.py
#
# Section 3.1 metrics for the four LOCAL models:
#   LR, RF, XGBoost, CatBoost
#
# Uses the frozen 77/23 random holdout already created in:
#   Data/samples/training_samples_union_with_splits.csv
#
# Computes:
#   - Accuracy
#   - ROC-AUC
#   - 95% AUC CI (stratified percentile bootstrap, 2000 reps)
#   - Class-wise precision, recall, F1 for:
#       class 0 = background / non-flood
#       class 1 = historical flood
#   - Confusion matrix
#   - ROC curve
#   - Exact two-sided McNemar tests between model predictions
#
# IMPORTANT:
#   Do NOT recreate the train/test split here.
#   Do NOT refit or tune the models.
#   This script evaluates the already-fitted random-holdout models.
# ============================================================

from pathlib import Path
from itertools import combinations
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import binomtest


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "Data"
    / "samples"
    / "training_samples_union_with_splits.csv"
)

MODELS_DIR = ROOT / "models"

OUT_DIR = (
    ROOT
    / "results"
    / "section_3_1"
)

FIG_DIR = (
    ROOT
    / "results"
    / "figures"
    / "section_3_1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN FEATURES
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
# SETTINGS
# ============================================================

BOOTSTRAP_REPS = 2000
RANDOM_SEED = 42
CLASS_THRESHOLD = 0.5


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "\nFrozen split file not found:\n"
        f"{DATA_FILE}\n\n"
        "This script intentionally does NOT recreate the split."
    )


df = pd.read_csv(
    DATA_FILE
)


required_columns = (
    ["sample_id", "class", "random_split"]
    + FEATURES
)

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns in split file: "
        f"{missing_columns}"
    )


train_mask = (
    df["random_split"]
    .astype(str)
    .str.lower()
    .eq("train")
)

test_mask = (
    df["random_split"]
    .astype(str)
    .str.lower()
    .eq("test")
)


n_train = int(
    train_mask.sum()
)

n_test = int(
    test_mask.sum()
)


# Frozen QC from the existing project design.
if n_train != 15400:
    raise RuntimeError(
        f"Unexpected training size: {n_train}. "
        "Expected 15,400."
    )

if n_test != 4600:
    raise RuntimeError(
        f"Unexpected test size: {n_test}. "
        "Expected 4,600."
    )


X_test = (
    df.loc[
        test_mask,
        FEATURES
    ]
    .copy()
)

y_test = (
    df.loc[
        test_mask,
        "class"
    ]
    .astype(int)
    .to_numpy()
)

sample_ids = (
    df.loc[
        test_mask,
        "sample_id"
    ]
    .astype(str)
    .to_numpy()
)


class_counts = pd.Series(
    y_test
).value_counts().to_dict()

if class_counts.get(0, 0) != 2300:
    raise RuntimeError(
        "Random test set no longer contains "
        "2,300 background samples."
    )

if class_counts.get(1, 0) != 2300:
    raise RuntimeError(
        "Random test set no longer contains "
        "2,300 flood samples."
    )


print(
    "=" * 80
)

print(
    "SECTION 3.1 — TELLA-STYLE RANDOM HOLDOUT METRICS"
)

print(
    "=" * 80
)

print(
    f"\nFrozen split file: {DATA_FILE}"
)

print(
    f"Train: {n_train:,}"
)

print(
    f"Test : {n_test:,} "
    f"(class 0 = 2,300; class 1 = 2,300)"
)


# ============================================================
# MODEL DISCOVERY
# ============================================================

if not MODELS_DIR.exists():
    raise FileNotFoundError(
        f"Models directory not found: {MODELS_DIR}"
    )


def all_model_files():
    allowed = {
        ".joblib",
        ".pkl",
        ".pickle",
        ".cbm",
        ".json",
        ".ubj",
    }

    return sorted(
        p
        for p in MODELS_DIR.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in allowed
        )
    )


MODEL_FILES = all_model_files()


def score_candidate(
    path,
    include_groups,
):
    """
    Each include group contains alternatives.
    Example:
      [["xgb", "xgboost"], ["union"], ["random"]]

    One token from each group must occur in the path.
    """

    text = str(
        path.relative_to(
            MODELS_DIR
        )
    ).lower()

    for group in include_groups:

        if not any(
            token in text
            for token in group
        ):
            return None

    # Prefer filenames containing "random" and "union",
    # and penalise fold/spatial models.
    score = 0

    if "random" in text:
        score += 10

    if "union" in text:
        score += 5

    if "spatial" in text:
        score -= 20

    if "fold" in text:
        score -= 20

    return score


def find_model_file(
    model_name,
    include_groups,
    preferred_names,
):

    # First: exact preferred basenames.
    preferred_lower = {
        n.lower()
        for n in preferred_names
    }

    preferred_hits = [
        p
        for p in MODEL_FILES
        if p.name.lower()
        in preferred_lower
    ]

    if len(
        preferred_hits
    ) == 1:
        return preferred_hits[0]


    # Second: token-based discovery.
    scored = []

    for p in MODEL_FILES:

        s = score_candidate(
            p,
            include_groups
        )

        if s is not None:
            scored.append(
                (s, p)
            )


    if not scored:

        print(
            "\nAvailable model files:"
        )

        for p in MODEL_FILES:
            print(
                "  ",
                p.relative_to(
                    ROOT
                )
            )

        raise FileNotFoundError(
            f"\nCould not locate the fitted "
            f"{model_name} random-holdout model."
        )


    scored.sort(
        key=lambda x: (
            -x[0],
            str(x[1]).lower(),
        )
    )

    best_score = scored[0][0]

    best = [
        p
        for s, p in scored
        if s == best_score
    ]


    if len(best) != 1:

        raise RuntimeError(
            f"\nAmbiguous {model_name} model files:\n"
            + "\n".join(
                f"  {p.relative_to(ROOT)}"
                for p in best
            )
            + "\n\nKeep only one random-holdout "
              "candidate or rename it clearly."
        )

    return best[0]


model_paths = {
    "LR":
        find_model_file(
            "LR",
            [
                ["lr", "logistic"],
                ["union"],
                ["random"],
            ],
            [
                "lr_union_random.joblib",
            ],
        ),

    "RF":
        find_model_file(
            "RF",
            [
                ["rf", "randomforest", "random_forest"],
                ["union"],
                ["random"],
            ],
            [
                "rf_union_random.joblib",
            ],
        ),

    "XGBoost":
        find_model_file(
            "XGBoost",
            [
                ["xgb", "xgboost"],
                ["union"],
                ["random"],
            ],
            [
                "xgboost_union_random.joblib",
                "xgb_union_random.joblib",
                "xgboost_union_random.json",
                "xgb_union_random.json",
                "xgboost_union_random.ubj",
                "xgb_union_random.ubj",
            ],
        ),

    "CatBoost":
        find_model_file(
            "CatBoost",
            [
                ["catboost", "cat"],
                ["union"],
                ["random"],
            ],
            [
                "catboost_union_random.joblib",
                "cat_union_random.joblib",
                "catboost_union_random.cbm",
                "cat_union_random.cbm",
            ],
        ),
}


print(
    "\nModels detected:"
)

for name, path in model_paths.items():

    print(
        f"  {name:<10} "
        f"{path.relative_to(ROOT)}"
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_saved_model(
    model_name,
    path,
):

    suffix = path.suffix.lower()


    if suffix in {
        ".joblib",
        ".pkl",
        ".pickle",
    }:

        return joblib.load(
            path
        )


    if (
        model_name == "CatBoost"
        and suffix == ".cbm"
    ):

        from catboost import (
            CatBoostClassifier
        )

        model = (
            CatBoostClassifier()
        )

        model.load_model(
            str(path)
        )

        return model


    if (
        model_name == "XGBoost"
        and suffix in {
            ".json",
            ".ubj",
        }
    ):

        from xgboost import (
            XGBClassifier
        )

        model = (
            XGBClassifier()
        )

        model.load_model(
            str(path)
        )

        return model


    raise RuntimeError(
        f"Unsupported saved model format "
        f"for {model_name}: {path}"
    )


# ============================================================
# PREDICTION HELPER
# ============================================================

def positive_probability(
    model,
    X,
):

    if not hasattr(
        model,
        "predict_proba"
    ):
        raise RuntimeError(
            "Loaded model does not provide "
            "predict_proba()."
        )


    p = model.predict_proba(
        X
    )


    if p.ndim != 2:
        raise RuntimeError(
            "predict_proba output is not 2-D."
        )


    # Respect class ordering rather than
    # blindly assuming column 1.
    classes = getattr(
        model,
        "classes_",
        None
    )


    if classes is None:

        if p.shape[1] != 2:
            raise RuntimeError(
                "Cannot identify positive-class "
                "probability column."
            )

        positive_col = 1

    else:

        classes = list(
            classes
        )

        if 1 not in classes:
            raise RuntimeError(
                f"Model classes do not contain "
                f"positive class 1: {classes}"
            )

        positive_col = (
            classes.index(1)
        )


    probability = np.asarray(
        p[:, positive_col],
        dtype=float,
    )


    if not np.isfinite(
        probability
    ).all():
        raise RuntimeError(
            "Non-finite predicted probabilities."
        )


    return probability


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

probabilities = {}
predictions = {}

for model_name, model_path in model_paths.items():

    print(
        f"\nEvaluating {model_name}..."
    )

    model = load_saved_model(
        model_name,
        model_path,
    )

    probability = (
        positive_probability(
            model,
            X_test,
        )
    )

    prediction = (
        probability
        >= CLASS_THRESHOLD
    ).astype(int)


    probabilities[
        model_name
    ] = probability

    predictions[
        model_name
    ] = prediction


    auc = roc_auc_score(
        y_test,
        probability,
    )

    print(
        f"  ROC-AUC: {auc:.4f}"
    )


# ============================================================
# SANITY CHECK AGAINST EXISTING PROJECT RESULTS
#
# These are NOT used as inputs. They only protect against
# accidentally loading the wrong saved model/split.
# ============================================================

EXPECTED_AUC = {
    "LR": 0.8414,
    "RF": 0.9338,
    "XGBoost": 0.9352,
    "CatBoost": 0.9342,
}

AUC_TOLERANCE = 0.0020


for model_name, expected in EXPECTED_AUC.items():

    observed = roc_auc_score(
        y_test,
        probabilities[
            model_name
        ],
    )

    if abs(
        observed - expected
    ) > AUC_TOLERANCE:

        raise RuntimeError(
            f"\n{model_name} AUC sanity check failed.\n"
            f"Expected approximately: {expected:.4f}\n"
            f"Observed:               {observed:.4f}\n\n"
            "STOP: this likely means the wrong model "
            "file or split was loaded."
        )


print(
    "\n✓ Existing random-holdout AUCs reproduced."
)


# ============================================================
# SHARED STRATIFIED BOOTSTRAP INDICES
#
# Same bootstrap samples are used for every model.
# ============================================================

positive_idx = np.flatnonzero(
    y_test == 1
)

negative_idx = np.flatnonzero(
    y_test == 0
)


rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_indices = []

for _ in range(
    BOOTSTRAP_REPS
):

    sampled_positive = rng.choice(
        positive_idx,
        size=len(
            positive_idx
        ),
        replace=True,
    )

    sampled_negative = rng.choice(
        negative_idx,
        size=len(
            negative_idx
        ),
        replace=True,
    )

    idx = np.concatenate(
        [
            sampled_negative,
            sampled_positive,
        ]
    )

    bootstrap_indices.append(
        idx
    )


# ============================================================
# OVERALL + CLASS-WISE METRICS
# ============================================================

overall_rows = []
classwise_rows = []
confusion_rows = []


for model_name in model_paths:

    probability = probabilities[
        model_name
    ]

    prediction = predictions[
        model_name
    ]


    auc = roc_auc_score(
        y_test,
        probability,
    )


    bootstrap_auc = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )


    for i, idx in enumerate(
        bootstrap_indices
    ):

        bootstrap_auc[i] = (
            roc_auc_score(
                y_test[idx],
                probability[idx],
            )
        )


    ci_lower, ci_upper = np.percentile(
        bootstrap_auc,
        [2.5, 97.5],
    )


    accuracy = accuracy_score(
        y_test,
        prediction,
    )


    overall_rows.append({
        "model":
            model_name,

        "accuracy":
            accuracy,

        "roc_auc":
            auc,

        "auc_ci_lower_95":
            ci_lower,

        "auc_ci_upper_95":
            ci_upper,

        "threshold":
            CLASS_THRESHOLD,

        "n_test":
            len(y_test),

        "n_class_0":
            int(
                (y_test == 0).sum()
            ),

        "n_class_1":
            int(
                (y_test == 1).sum()
            ),
    })


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            prediction,
            labels=[0, 1],
            zero_division=0,
        )
    )


    class_names = {
        0: "Not Flooded",
        1: "Flooded",
    }


    for j, label in enumerate(
        [0, 1]
    ):

        classwise_rows.append({
            "model":
                model_name,

            "class_label":
                label,

            "class_name":
                class_names[
                    label
                ],

            "precision":
                precision[j],

            "recall":
                recall[j],

            "f1":
                f1[j],

            "support":
                int(
                    support[j]
                ),
        })


    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )


    confusion_rows.append({
        "model":
            model_name,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    })


overall_df = pd.DataFrame(
    overall_rows
)

classwise_df = pd.DataFrame(
    classwise_rows
)

confusion_df = pd.DataFrame(
    confusion_rows
)


# ============================================================
# EXACT TWO-SIDED McNEMAR TEST
#
# McNemar compares paired correctness on the SAME 4,600
# holdout samples.
#
# b = A correct, B wrong
# c = A wrong, B correct
#
# Exact two-sided binomial version:
# H0: b and c are equally likely.
# ============================================================

mcnemar_rows = []


for model_a, model_b in combinations(
    model_paths.keys(),
    2,
):

    correct_a = (
        predictions[
            model_a
        ] == y_test
    )

    correct_b = (
        predictions[
            model_b
        ] == y_test
    )


    b = int(
        np.sum(
            correct_a
            & ~correct_b
        )
    )

    c = int(
        np.sum(
            ~correct_a
            & correct_b
        )
    )

    discordant = (
        b + c
    )


    if discordant == 0:

        p_value = 1.0

    else:

        p_value = (
            binomtest(
                k=min(
                    b,
                    c
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            )
            .pvalue
        )


    mcnemar_rows.append({
        "model_a":
            model_a,

        "model_b":
            model_b,

        "a_correct_b_wrong":
            b,

        "a_wrong_b_correct":
            c,

        "discordant_total":
            discordant,

        "p_exact_two_sided":
            p_value,
    })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


# ============================================================
# SAVE TEST PREDICTIONS
#
# This file is essential for merging TabPFN later.
# ============================================================

predictions_df = pd.DataFrame({
    "sample_id":
        sample_ids,

    "y_true":
        y_test,
})


for model_name in model_paths:

    safe_name = (
        model_name
        .lower()
        .replace(
            "boost",
            "boost"
        )
    )

    predictions_df[
        f"{safe_name}_prob"
    ] = probabilities[
        model_name
    ]

    predictions_df[
        f"{safe_name}_pred"
    ] = predictions[
        model_name
    ]


# ============================================================
# ROC CURVE TABLE + FIGURE
# ============================================================

roc_rows = []


fig, ax = plt.subplots(
    figsize=(7.5, 6.5)
)


for model_name in model_paths:

    fpr, tpr, thresholds = (
        roc_curve(
            y_test,
            probabilities[
                model_name
            ],
        )
    )

    auc = roc_auc_score(
        y_test,
        probabilities[
            model_name
        ],
    )


    ax.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC={auc:.3f})"
        ),
    )


    for x, y, t in zip(
        fpr,
        tpr,
        thresholds,
    ):

        roc_rows.append({
            "model":
                model_name,

            "fpr":
                x,

            "tpr":
                y,

            "threshold":
                t,
        })


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random",
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curve — Random Holdout"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "roc_curve_local_models.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIG_DIR
    / "roc_curve_local_models.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)


roc_df = pd.DataFrame(
    roc_rows
)


# ============================================================
# TELLA-STYLE CLASS METRIC FIGURE
#
# One figure; class 0 and class 1 are displayed separately
# for each model using precision, recall and F1.
# ============================================================

plot_df = classwise_df.copy()

model_order = list(
    model_paths.keys()
)

class_order = [
    "Not Flooded",
    "Flooded",
]

metric_names = [
    "precision",
    "recall",
    "f1",
]


labels = []

values = {
    metric: []
    for metric in metric_names
}


for model_name in model_order:

    for class_name in class_order:

        row = (
            plot_df[
                (
                    plot_df["model"]
                    == model_name
                )
                & (
                    plot_df["class_name"]
                    == class_name
                )
            ]
            .iloc[0]
        )

        labels.append(
            f"{model_name}\n{class_name}"
        )

        for metric in metric_names:

            values[
                metric
            ].append(
                row[
                    metric
                ]
            )


x = np.arange(
    len(labels)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(12, 6)
)


for j, metric in enumerate(
    metric_names
):

    offset = (
        j - 1
    ) * width

    ax.bar(
        x + offset,
        values[
            metric
        ],
        width,
        label=metric.capitalize(),
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    rotation=30,
    ha="right",
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.02
)

ax.set_title(
    "Class-wise Performance — Random Holdout"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "classwise_metrics_local_models.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIG_DIR
    / "classwise_metrics_local_models.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# SAVE OUTPUT TABLES
# ============================================================

overall_df.to_csv(
    OUT_DIR
    / "random_holdout_overall_metrics_local.csv",
    index=False,
)

classwise_df.to_csv(
    OUT_DIR
    / "random_holdout_classwise_metrics_local.csv",
    index=False,
)

confusion_df.to_csv(
    OUT_DIR
    / "random_holdout_confusion_matrix_local.csv",
    index=False,
)

mcnemar_df.to_csv(
    OUT_DIR
    / "mcnemar_local_models.csv",
    index=False,
)

predictions_df.to_csv(
    OUT_DIR
    / "random_holdout_predictions_local.csv",
    index=False,
)

roc_df.to_csv(
    OUT_DIR
    / "roc_curve_points_local.csv",
    index=False,
)


# ============================================================
# METHOD RECORD
# ============================================================

method_record = {
    "section":
        "3.1 Comparative Analysis of Model Performance",

    "evaluation_set":
        "Frozen 77/23 random holdout",

    "n_train":
        n_train,

    "n_test":
        n_test,

    "test_class_balance":
        {
            "class_0_not_flooded":
                int(
                    (y_test == 0).sum()
                ),

            "class_1_flooded":
                int(
                    (y_test == 1).sum()
                ),
        },

    "classification_threshold":
        CLASS_THRESHOLD,

    "reported_metrics":
        [
            "accuracy",
            "ROC-AUC",
            "class-wise precision",
            "class-wise recall",
            "class-wise F1",
        ],

    "auc_ci":
        {
            "method":
                "stratified percentile bootstrap",

            "repetitions":
                BOOTSTRAP_REPS,

            "confidence_level":
                0.95,

            "random_seed":
                RANDOM_SEED,

            "note":
                (
                    "Tella et al. report 95% AUC confidence "
                    "intervals, but the cited Results/Accuracy "
                    "Assessment text does not specify the exact "
                    "CI estimator. This project therefore uses "
                    "a documented stratified percentile bootstrap."
                ),
        },

    "mcnemar":
        {
            "method":
                "exact two-sided McNemar test via binomial test",

            "paired_unit":
                "same random-holdout sample",

            "note":
                (
                    "Compares whether two models differ in "
                    "their correct/incorrect classification "
                    "decisions on the same holdout observations."
                ),
        },

    "model_files":
        {
            model:
                str(
                    path.relative_to(
                        ROOT
                    )
                )
            for model, path
            in model_paths.items()
        },
}


with open(
    OUT_DIR
    / "section_3_1_method_local.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        method_record,
        f,
        indent=2,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option(
    "display.width",
    180
)

pd.set_option(
    "display.max_columns",
    20
)


print(
    "\n"
    + "=" * 80
)

print(
    "OVERALL RANDOM-HOLDOUT METRICS"
)

print(
    "=" * 80
)

print(
    overall_df[
        [
            "model",
            "accuracy",
            "roc_auc",
            "auc_ci_lower_95",
            "auc_ci_upper_95",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "CLASS-WISE PRECISION / RECALL / F1"
)

print(
    "=" * 80
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
    "\n"
    + "=" * 80
)

print(
    "EXACT PAIRWISE McNEMAR TESTS"
)

print(
    "=" * 80
)

print(
    mcnemar_df
    .round(
        {
            "p_exact_two_sided":
                6
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\nSaved tables to:"
)

print(
    f"  {OUT_DIR}"
)

print(
    "\nSaved figures to:"
)

print(
    f"  {FIG_DIR}"
)


print(
    "\nIMPORTANT:"
)

print(
    "This completes LR/RF/XGBoost/CatBoost only."
)

print(
    "TabPFN must be added using its probabilities "
    "from the SAME 4,600-sample holdout before "
    "the final five-model McNemar table/figures."
)
