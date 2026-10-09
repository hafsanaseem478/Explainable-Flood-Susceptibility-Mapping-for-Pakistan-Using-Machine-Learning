from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SECTION_DIR = (
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

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_FILE = (
    SECTION_DIR
    / "random_holdout_overall_metrics_all5.csv"
)

SPATIAL_FILE = (
    SECTION_DIR
    / "spatial_performance_all5.csv"
)

OUTPUT_CSV = (
    SECTION_DIR
    / "random_vs_spatial_auc_all5.csv"
)

OUTPUT_PNG = (
    FIGURE_DIR
    / "random_vs_spatial_auc_all5.png"
)

OUTPUT_PDF = (
    FIGURE_DIR
    / "random_vs_spatial_auc_all5.pdf"
)


# ============================================================
# MODEL ORDER
# ============================================================

MODEL_ORDER = [
    "LR",
    "RF",
    "XGBoost",
    "CatBoost",
    "TabPFN",
]

DISPLAY_NAMES = {
    "LR": "LR",
    "RF": "RF",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "TabPFN": "TabPFN",
}


# ============================================================
# FROZEN RANDOM-HOLDOUT 95% AUC CI
#
# These are the ORIGINAL bootstrap CIs already calculated
# for Section 3.1. Do not recompute them here.
# ============================================================

RANDOM_CI = {
    "LR": (
        0.8298,
        0.8531,
    ),

    "RF": (
        0.9267,
        0.9407,
    ),

    "XGBoost": (
        0.9284,
        0.9417,
    ),

    "CatBoost": (
        0.9273,
        0.9407,
    ),

    "TabPFN": (
        0.9462,
        0.9581,
    ),
}


# ============================================================
# LOAD
# ============================================================

if not RANDOM_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{RANDOM_FILE}"
    )

if not SPATIAL_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{SPATIAL_FILE}"
    )


random_df = pd.read_csv(
    RANDOM_FILE
)

spatial_df = pd.read_csv(
    SPATIAL_FILE
)


# ============================================================
# COLUMN CHECKS
# ============================================================

required_random = {
    "model",
    "roc_auc",
}

required_spatial = {
    "model",
    "mean_fold_auc",
    "sd_fold_auc",
}


if not required_random.issubset(
    random_df.columns
):
    raise RuntimeError(
        f"Random CSV missing columns: "
        f"{required_random - set(random_df.columns)}"
    )


if not required_spatial.issubset(
    spatial_df.columns
):
    raise RuntimeError(
        f"Spatial CSV missing columns: "
        f"{required_spatial - set(spatial_df.columns)}"
    )


# ============================================================
# BUILD FINAL COMPARISON TABLE
# ============================================================

rows = []


for model in MODEL_ORDER:

    random_match = random_df[
        random_df["model"] == model
    ]

    spatial_match = spatial_df[
        spatial_df["model"] == model
    ]


    if len(random_match) != 1:
        raise RuntimeError(
            f"Expected exactly one random row for {model}"
        )

    if len(spatial_match) != 1:
        raise RuntimeError(
            f"Expected exactly one spatial row for {model}"
        )


    random_auc = float(
        random_match["roc_auc"].iloc[0]
    )

    spatial_auc = float(
        spatial_match["mean_fold_auc"].iloc[0]
    )

    spatial_sd = float(
        spatial_match["sd_fold_auc"].iloc[0]
    )

    ci_low, ci_high = RANDOM_CI[
        model
    ]


    rows.append({
        "model":
            model,

        "random_auc":
            random_auc,

        "random_ci_low":
            ci_low,

        "random_ci_high":
            ci_high,

        "spatial_mean_auc":
            spatial_auc,

        "spatial_sd":
            spatial_sd,

        "delta_auc":
            spatial_auc
            - random_auc,
    })


comparison = pd.DataFrame(
    rows
)


# ============================================================
# QC AGAINST FROZEN RESULTS
# ============================================================

EXPECTED_RANDOM = {
    "LR": 0.8414,
    "RF": 0.9338,
    "XGBoost": 0.9352,
    "CatBoost": 0.9342,
    "TabPFN": 0.9523,
}

EXPECTED_SPATIAL = {
    "LR": 0.8395,
    "RF": 0.9157,
    "XGBoost": 0.9159,
    "CatBoost": 0.9177,
    "TabPFN": 0.9102,
}


for _, row in comparison.iterrows():

    model = row["model"]

    if abs(
        row["random_auc"]
        - EXPECTED_RANDOM[model]
    ) > 0.0001:

        raise RuntimeError(
            f"{model}: random AUC mismatch"
        )


    if abs(
        row["spatial_mean_auc"]
        - EXPECTED_SPATIAL[model]
    ) > 0.0001:

        raise RuntimeError(
            f"{model}: spatial AUC mismatch"
        )


print(
    "✓ Random and spatial AUC values match "
    "the frozen Section 3.1 results"
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# PREPARE FIGURE
# ============================================================

x = np.arange(
    len(MODEL_ORDER)
)

offset = 0.11


random_auc = comparison[
    "random_auc"
].to_numpy()

spatial_auc = comparison[
    "spatial_mean_auc"
].to_numpy()


# Asymmetric 95% CI for random holdout
random_lower_error = (
    random_auc
    - comparison[
        "random_ci_low"
    ].to_numpy()
)

random_upper_error = (
    comparison[
        "random_ci_high"
    ].to_numpy()
    - random_auc
)

random_yerr = np.vstack([
    random_lower_error,
    random_upper_error,
])


# Spatial ± SD
spatial_yerr = comparison[
    "spatial_sd"
].to_numpy()


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9.5,
        5.8,
    )
)


RANDOM_COLOR = "#2166AC"
SPATIAL_COLOR = "#E66101"


# ------------------------------------------------------------
# Connecting lines
#
# Neutral gray: the line only marks the pairing between the
# two points for the same model, so it should not carry its
# own color meaning.
# ------------------------------------------------------------

for i in range(
    len(x)
):

    ax.plot(
        [
            x[i] - offset,
            x[i] + offset,
        ],
        [
            random_auc[i],
            spatial_auc[i],
        ],
        color="0.65",
        linewidth=1.1,
        alpha=0.8,
        zorder=1,
    )


# ------------------------------------------------------------
# Random holdout
# ------------------------------------------------------------

ax.errorbar(
    x - offset,
    random_auc,
    yerr=random_yerr,
    fmt="o",
    color=RANDOM_COLOR,
    markersize=7,
    capsize=4,
    linewidth=1.4,
    label="Random holdout (95% CI)",
    zorder=3,
)


# ------------------------------------------------------------
# Spatial CV
# ------------------------------------------------------------

ax.errorbar(
    x + offset,
    spatial_auc,
    yerr=spatial_yerr,
    fmt="s",
    color=SPATIAL_COLOR,
    markersize=7,
    capsize=4,
    linewidth=1.4,
    label="Spatial CV (mean ± SD)",
    zorder=3,
)


# ============================================================
# DELTA LABELS
# ============================================================

for i, row in comparison.iterrows():

    delta = row[
        "delta_auc"
    ]

    y_text = max(
        row["random_ci_high"],
        row["spatial_mean_auc"]
        + row["spatial_sd"],
    ) + 0.009


    ax.text(
        x[i],
        y_text,
        f"Δ {delta:+.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="0.25",
    )


# ============================================================
# AXIS FORMATTING
# ============================================================

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        DISPLAY_NAMES[m]
        for m
        in MODEL_ORDER
    ],
    fontsize=10,
)


ax.set_ylabel(
    "ROC-AUC",
    fontsize=11,
)

ax.set_xlabel(
    "Model",
    fontsize=11,
)


# Fit the axis to the actual data instead of a fixed 0.75-1.00
# range, so the real differences between points aren't
# compressed into a small band of the plot.
data_low = min(
    comparison["random_ci_low"].min(),
    (
        comparison["spatial_mean_auc"]
        - comparison["spatial_sd"]
    ).min(),
)

data_high = max(
    comparison["random_ci_high"].max(),
    (
        comparison["spatial_mean_auc"]
        + comparison["spatial_sd"]
    ).max(),
)

y_low = 0.02 * np.floor(
    (data_low - 0.015) / 0.02
)

y_high = 0.02 * np.ceil(
    (data_high + 0.03) / 0.02
)

ax.set_ylim(
    y_low,
    y_high,
)


ax.set_yticks(
    np.arange(
        y_low,
        y_high + 0.001,
        0.02,
    )
)


ax.grid(
    axis="y",
    linewidth=0.6,
    alpha=0.25,
)

ax.set_axisbelow(
    True
)


# ============================================================
# CLEAN SPINES
# ============================================================

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    loc="upper left",
    frameon=False,
    fontsize=9,
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Random Holdout vs Spatial Cross-Validation",
    fontsize=12,
    pad=12,
)


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# PRINT TABLE
# ============================================================

print(
    "\n"
    + comparison.round(4).to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_PNG
)

print(
    OUTPUT_PDF
)                                                                                                                                                       