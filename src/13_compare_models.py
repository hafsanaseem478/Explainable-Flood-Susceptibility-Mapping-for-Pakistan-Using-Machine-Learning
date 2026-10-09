from pathlib import Path
import pandas as pd


RESULT_ROOT = Path("results/models")

MODELS = {
    "Logistic Regression": {
        "folder": "lr_union",
        "random": "random_metrics.csv",
        "spatial": "spatial_summary.csv",
    },
    "Random Forest": {
        "folder": "rf_union",
        "random": "random_metrics.csv",
        "spatial": "spatial_summary.csv",
    },
    "XGBoost": {
        "folder": "xgb_union",
        "random": "random_metrics.csv",
        "spatial": "spatial_summary.csv",
    },
    "CatBoost": {
        "folder": "catboost_union",
        "random": "random_metrics.csv",
        "spatial": "spatial_summary.csv",
    },
    "TabPFN": {
        "folder": "tabpfn_union",
        "random": "tabpfn_random_metrics.csv",
        "spatial": "tabpfn_spatial_summary.csv",
    },
}


rows = []


for model_name, info in MODELS.items():

    folder = RESULT_ROOT / info["folder"]

    random_file = folder / info["random"]
    spatial_file = folder / info["spatial"]

    if not random_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{random_file}"
        )

    if not spatial_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{spatial_file}"
        )


    random_df = pd.read_csv(
        random_file
    )

    spatial_df = pd.read_csv(
        spatial_file
    )


    # --------------------------------------------
    # Random metrics
    # --------------------------------------------

    random_row = random_df.iloc[0]


    # --------------------------------------------
    # Spatial summary is stored long-format:
    #
    # metric | mean | sd
    # --------------------------------------------

    spatial_lookup = (
        spatial_df
        .set_index("metric")
    )


    row = {
        "model": model_name,

        "random_roc_auc":
            random_row["roc_auc"],

        "spatial_roc_auc_mean":
            spatial_lookup.loc[
                "roc_auc",
                "mean"
            ],

        "spatial_roc_auc_sd":
            spatial_lookup.loc[
                "roc_auc",
                "sd"
            ],

        "random_pr_auc":
            random_row["pr_auc"],

        "spatial_pr_auc_mean":
            spatial_lookup.loc[
                "pr_auc",
                "mean"
            ],

        "random_accuracy":
            random_row["accuracy"],

        "spatial_accuracy_mean":
            spatial_lookup.loc[
                "accuracy",
                "mean"
            ],

        "random_f1":
            random_row["f1"],

        "spatial_f1_mean":
            spatial_lookup.loc[
                "f1",
                "mean"
            ],

        "random_brier":
            random_row["brier"],

        "spatial_brier_mean":
            spatial_lookup.loc[
                "brier",
                "mean"
            ],
    }


    row[
        "auc_random_spatial_gap"
    ] = (
        row["random_roc_auc"]
        - row["spatial_roc_auc_mean"]
    )


    rows.append(row)


# ============================================================
# MASTER TABLE
# ============================================================

comparison = pd.DataFrame(
    rows
)


comparison = comparison.sort_values(
    "spatial_roc_auc_mean",
    ascending=False
)


OUT_DIR = Path(
    "results/tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT = (
    OUT_DIR
    / "model_comparison.csv"
)


comparison.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)


print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)


display = comparison.copy()


for col in display.columns:

    if col != "model":
        display[col] = (
            display[col]
            .map(
                lambda x: f"{x:.4f}"
            )
        )


print(
    display.to_string(
        index=False
    )
)


print(
    f"\nSaved:\n{OUTPUT}"
)