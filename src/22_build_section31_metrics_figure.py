# ============================================================
# 22_build_section31_metrics_figure.py
#
# FINAL SECTION 3.1 TABLES + TELLA-STYLE PERFORMANCE FIGURE
#
# INPUTS:
#   results/section_3_1/random_holdout_predictions_local.csv
#   results/section_3_1/tabpfn_random_holdout_predictions.csv
#
# OUTPUTS:
#   results/section_3_1/random_holdout_overall_metrics_all5.csv
#   results/section_3_1/random_holdout_classwise_metrics_all5.csv
#   results/section_3_1/random_holdout_confusion_matrix_all5.csv
#
#   results/figures/section_3_1/
#       classwise_metrics_all5.png
#       classwise_metrics_all5.pdf
#
# IMPORTANT:
#   This script does NOT retrain any model.
#   It uses the already-saved predictions on the SAME
#   frozen 4,600-sample random holdout.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "section_3_1"
)

FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
    / "section_3_1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOCAL_PREDICTIONS = (
    RESULTS_DIR
    / "random_holdout_predictions_local.csv"
)

TABPFN_PREDICTIONS = (
    RESULTS_DIR
    / "tabpfn_random_holdout_predictions.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OVERALL_OUTPUT = (
    RESULTS_DIR
    / "random_holdout_overall_metrics_all5.csv"
)

CLASSWISE_OUTPUT = (
    RESULTS_DIR
    / "random_holdout_classwise_metrics_all5.csv"
)

CONFUSION_OUTPUT = (
    RESULTS_DIR
    / "random_holdout_confusion_matrix_all5.csv"
)

MERGED_OUTPUT = (
    RESULTS_DIR
    / "random_holdout_predictions_all5.csv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "classwise_metrics_all5.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "classwise_metrics_all5.pdf"
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_COLUMNS = {
    "LR": {
        "prob": "lr_prob",
        "pred": "lr_pred",
        "display": "Logistic Regression",
    },

    "RF": {
        "prob": "rf_prob",
        "pred": "rf_pred",
        "display": "Random Forest",
    },

    "XGBoost": {
        "prob": "xgboost_prob",
        "pred": "xgboost_pred",
        "display": "XGBoost",
    },

    "CatBoost": {
        "prob": "catboost_prob",
        "pred": "catboost_pred",
        "display": "CatBoost",
    },

    "TabPFN": {
        "prob": "tabpfn_prob",
        "pred": "tabpfn_pred",
        "display": "TabPFN",
    },
}


# Figure order:
# top to bottom
FIGURE_MODEL_ORDER = [
    "TabPFN",
    "CatBoost",
    "XGBoost",
    "RF",
    "LR",
]


# ============================================================
# LOAD INPUT FILES
# ============================================================

print("=" * 80)
print("SECTION 3.1 — FINAL FIVE-MODEL METRICS + FIGURE")
print("=" * 80)


if not LOCAL_PREDICTIONS.exists():

    raise FileNotFoundError(
        "\nMissing local-model predictions:\n"
        f"{LOCAL_PREDICTIONS}"
    )


if not TABPFN_PREDICTIONS.exists():

    raise FileNotFoundError(
        "\nMissing TabPFN predictions:\n"
        f"{TABPFN_PREDICTIONS}\n\n"
        "Put tabpfn_random_holdout_predictions.csv "
        "inside results/section_3_1/"
    )


local = pd.read_csv(
    LOCAL_PREDICTIONS
)

tabpfn = pd.read_csv(
    TABPFN_PREDICTIONS
)


print(
    f"\nLocal prediction rows : {len(local):,}"
)

print(
    f"TabPFN prediction rows: {len(tabpfn):,}"
)


# ============================================================
# STRICT INPUT CHECKS
# ============================================================

required_local_columns = {
    "sample_id",
    "y_true",
    "lr_prob",
    "lr_pred",
    "rf_prob",
    "rf_pred",
    "xgboost_prob",
    "xgboost_pred",
    "catboost_prob",
    "catboost_pred",
}


required_tabpfn_columns = {
    "sample_id",
    "y_true",
    "tabpfn_prob",
    "tabpfn_pred",
}


missing_local = (
    required_local_columns
    - set(
        local.columns
    )
)

missing_tabpfn = (
    required_tabpfn_columns
    - set(
        tabpfn.columns
    )
)


if missing_local:

    raise RuntimeError(
        "Missing columns from local prediction file: "
        f"{sorted(missing_local)}"
    )


if missing_tabpfn:

    raise RuntimeError(
        "Missing columns from TabPFN prediction file: "
        f"{sorted(missing_tabpfn)}"
    )


if len(local) != 4600:

    raise RuntimeError(
        f"Local file has {len(local)} rows. "
        "Expected exactly 4,600."
    )


if len(tabpfn) != 4600:

    raise RuntimeError(
        f"TabPFN file has {len(tabpfn)} rows. "
        "Expected exactly 4,600."
    )


if not local["sample_id"].is_unique:

    raise RuntimeError(
        "Duplicate sample_id values found "
        "in local prediction file."
    )


if not tabpfn["sample_id"].is_unique:

    raise RuntimeError(
        "Duplicate sample_id values found "
        "in TabPFN prediction file."
    )


# ============================================================
# MERGE BY SAMPLE_ID
#
# Never rely on row order.
# ============================================================

merged = local.merge(
    tabpfn,
    on="sample_id",
    how="outer",
    suffixes=(
        "_local",
        "_tabpfn",
    ),
    indicator=True,
)


if len(merged) != 4600:

    raise RuntimeError(
        "Merged data does not contain "
        "exactly 4,600 observations."
    )


if not (
    merged["_merge"]
    == "both"
).all():

    problem_rows = (
        merged[
            merged["_merge"]
            != "both"
        ]
    )

    raise RuntimeError(
        "Some sample IDs did not match "
        "between local and TabPFN files.\n"
        f"{problem_rows[['sample_id', '_merge']].head()}"
    )


# ============================================================
# VERIFY TARGET LABELS ARE IDENTICAL
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
        "y_true values differ between "
        "local and TabPFN files."
    )


merged = (
    merged
    .drop(
        columns=[
            "_merge",
            "y_true_tabpfn",
        ]
    )
    .rename(
        columns={
            "y_true_local":
                "y_true"
        }
    )
)


y_true = (
    merged[
        "y_true"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# FINAL TARGET QC
# ============================================================

n_background = int(
    np.sum(
        y_true == 0
    )
)

n_flooded = int(
    np.sum(
        y_true == 1
    )
)


if n_background != 2300:

    raise RuntimeError(
        f"Expected 2,300 background observations, "
        f"found {n_background}."
    )


if n_flooded != 2300:

    raise RuntimeError(
        f"Expected 2,300 flooded observations, "
        f"found {n_flooded}."
    )


print(
    "\n✓ All 4,600 sample IDs matched"
)

print(
    "✓ Target labels match exactly"
)

print(
    "✓ Background = 2,300"
)

print(
    "✓ Flooded    = 2,300"
)


# ============================================================
# CHECK PREDICTION VALUES
# ============================================================

for model_name, cols in MODEL_COLUMNS.items():

    prob_col = cols[
        "prob"
    ]

    pred_col = cols[
        "pred"
    ]


    if prob_col not in merged.columns:

        raise RuntimeError(
            f"Missing {prob_col}"
        )


    if pred_col not in merged.columns:

        raise RuntimeError(
            f"Missing {pred_col}"
        )


    probability = (
        merged[
            prob_col
        ]
        .astype(float)
        .to_numpy()
    )


    prediction = (
        merged[
            pred_col
        ]
        .astype(int)
        .to_numpy()
    )


    if not np.isfinite(
        probability
    ).all():

        raise RuntimeError(
            f"{model_name}: "
            "non-finite probabilities found."
        )


    if not np.isin(
        prediction,
        [
            0,
            1,
        ]
    ).all():

        raise RuntimeError(
            f"{model_name}: "
            "predictions contain values other "
            "than 0 and 1."
        )


# ============================================================
# CALCULATE OVERALL + CLASS-WISE METRICS
# ============================================================

overall_rows = []

classwise_rows = []

confusion_rows = []


for model_name, cols in MODEL_COLUMNS.items():

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


    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        prediction
    )


    auc = roc_auc_score(
        y_true,
        probability
    )


    overall_rows.append({
        "model":
            model_name,

        "display_name":
            cols["display"],

        "accuracy":
            accuracy,

        "roc_auc":
            auc,

        "n_test":
            len(
                y_true
            ),
    })


    # --------------------------------------------------------
    # Class-wise precision / recall / F1
    # --------------------------------------------------------

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
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
                model_name,

            "display_name":
                cols["display"],

            "class_label":
                label,

            # NOTE:
            # class 0 is sampled background,
            # not proof that a location never flooded.
            # Displayed as "Non-flooded" for readability.
            "class_name":
                (
                    "Non-flooded"
                    if label == 0
                    else "Flooded"
                ),

            "precision":
                precision[i],

            "recall":
                recall[i],

            "f1":
                f1[i],

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
            y_true,
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
            model_name,

        "display_name":
            cols["display"],

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
# CREATE DATAFRAMES
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


# ============================================================
# SAFETY CHECK AGAINST OUR ESTABLISHED AUC RESULTS
# ============================================================

EXPECTED_AUC = {
    "LR": 0.8414,
    "RF": 0.9338,
    "XGBoost": 0.9352,
    "CatBoost": 0.9342,
    "TabPFN": 0.9523,
}


for model_name, expected_auc in EXPECTED_AUC.items():

    observed_auc = float(
        overall_df.loc[
            overall_df[
                "model"
            ] == model_name,
            "roc_auc",
        ].iloc[0]
    )


    if abs(
        observed_auc
        - expected_auc
    ) > 0.001:

        raise RuntimeError(
            f"\n{model_name} AUC does not match "
            "the established result.\n"
            f"Expected approximately: {expected_auc:.4f}\n"
            f"Observed:               {observed_auc:.4f}\n\n"
            "STOP. Do not make the figure."
        )


print(
    "\n✓ All five established AUC values reproduced"
)


# ============================================================
# SAVE FINAL CSV FILES
# ============================================================

overall_df.to_csv(
    OVERALL_OUTPUT,
    index=False
)

classwise_df.to_csv(
    CLASSWISE_OUTPUT,
    index=False
)

confusion_df.to_csv(
    CONFUSION_OUTPUT,
    index=False
)

merged.to_csv(
    MERGED_OUTPUT,
    index=False
)


print(
    "\nCSV files created:"
)

print(
    f"  {OVERALL_OUTPUT}"
)

print(
    f"  {CLASSWISE_OUTPUT}"
)

print(
    f"  {CONFUSION_OUTPUT}"
)

print(
    f"  {MERGED_OUTPUT}"
)


# ============================================================
# PRINT METRICS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "OVERALL METRICS"
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
    "CLASS-WISE METRICS"
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


# ============================================================
# CLEAN PUBLICATION-STYLE PERFORMANCE FIGURE
# ============================================================

MODEL_ORDER = [
    "TabPFN",
    "CatBoost",
    "XGBoost",
    "RF",
    "LR",
]

DISPLAY_NAMES = {
    "TabPFN": "TabPFN",
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "RF": "Random Forest",
    "LR": "Logistic Regression",
}

CLASS_ORDER = [
    "Flooded",
    "Non-flooded",
]

METRICS = [
    "f1",
    "recall",
    "precision",
]

METRIC_LABELS = {
    "f1": "F1-Score",
    "recall": "Recall",
    "precision": "Precision",
}

COLORS = {
    "f1": "#3E7D22",
    "recall": "#F04F50",
    "precision": "#178A8A",
}

# ============================================================
# BUILD ROW POSITIONS
# ============================================================

row_gap = 0.78
group_gap = 0.48

y_positions = {}
group_centers = {}

y = 0

for model in MODEL_ORDER:

    positions_this_model = []

    for cls in CLASS_ORDER:

        y_positions[(model, cls)] = y
        positions_this_model.append(y)

        y += row_gap

    group_centers[model] = np.mean(
        positions_this_model
    )

    y += group_gap


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 7.2)
)

X_MIN = 0.70
X_MAX = 1.00

bar_height = 0.15

offsets = {
    "f1": -0.22,
    "recall": 0.00,
    "precision": 0.22,
}


# ============================================================
# DRAW BARS
# ============================================================

legend_added = set()

for model in MODEL_ORDER:

    for cls in CLASS_ORDER:

        row = classwise_df[
            (classwise_df["model"] == model)
            &
            (classwise_df["class_name"] == cls)
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected one row for {model} / {cls}, "
                f"found {len(row)}"
            )

        row = row.iloc[0]

        base_y = y_positions[
            (model, cls)
        ]

        for metric in METRICS:

            value = float(
                row[metric]
            )

            label = (
                METRIC_LABELS[metric]
                if metric not in legend_added
                else None
            )

            ax.barh(
                base_y + offsets[metric],
                value - X_MIN,
                left=X_MIN,
                height=bar_height,
                color=COLORS[metric],
                edgecolor="white",
                linewidth=0.6,
                label=label,
            )

            legend_added.add(
                metric
            )

            # Value label
            text_x = value + 0.008

            if value > 0.965:
                text_x = value - 0.008
                ha = "right"
            else:
                ha = "left"

            ax.text(
                text_x,
                base_y + offsets[metric],
                f"{value:.2f}",
                va="center",
                ha=ha,
                fontsize=8,
            )


# ============================================================
# CLASS LABELS
# ============================================================

yticks = []
yticklabels = []

for model in MODEL_ORDER:

    for cls in CLASS_ORDER:

        yticks.append(
            y_positions[
                (model, cls)
            ]
        )

        yticklabels.append(
            cls
        )


ax.set_yticks(
    yticks
)

ax.set_yticklabels(
    yticklabels,
    fontsize=9,
)


# ============================================================
# MODEL + AUC LABELS
# ============================================================

for model in MODEL_ORDER:

    auc = float(
        overall_df.loc[
            overall_df["model"] == model,
            "roc_auc",
        ].iloc[0]
    )

    label = (
        f"{DISPLAY_NAMES[model]}\n"
        f"AUC = {auc:.3f}"
    )

    ax.text(
        X_MIN - 0.045,
        group_centers[model],
        label,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


# ============================================================
# MODEL SEPARATORS
# ============================================================

for i in range(
    len(MODEL_ORDER) - 1
):

    model_now = MODEL_ORDER[i]
    model_next = MODEL_ORDER[i + 1]

    y1 = y_positions[
        (model_now, "Non-flooded")
    ]

    y2 = y_positions[
        (model_next, "Flooded")
    ]

    separator = (
        y1 + y2
    ) / 2

    ax.axhline(
        separator,
        linewidth=0.7,
        color="0.82",
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlim(
    X_MIN,
    X_MAX
)

ax.set_xticks(
    np.arange(
        0.70,
        1.001,
        0.05
    )
)

ax.set_xlabel(
    "Score",
    fontsize=11
)

ax.set_ylabel(
    ""
)

ax.invert_yaxis()

ax.grid(
    axis="x",
    linewidth=0.5,
    alpha=0.25
)

ax.set_axisbelow(
    True
)


# ============================================================
# CLEAN SPINES
# ============================================================

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

ax.spines["left"].set_visible(
    False
)

ax.tick_params(
    axis="y",
    length=0,
    pad=8
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=3,
    frameon=False,
    fontsize=10,
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Random Holdout Classification Performance",
    fontsize=12,
    pad=32,
)


# ============================================================
# SAVE
# ============================================================

plt.subplots_adjust(
    left=0.29,
    right=0.97,
    top=0.88,
    bottom=0.10,
)

OUTPUT_PNG = (
    FIGURE_DIR
    / "classwise_metrics_all5_clean.png"
)

OUTPUT_PDF = (
    FIGURE_DIR
    / "classwise_metrics_all5_clean.pdf"
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(OUTPUT_PNG)
print(OUTPUT_PDF)