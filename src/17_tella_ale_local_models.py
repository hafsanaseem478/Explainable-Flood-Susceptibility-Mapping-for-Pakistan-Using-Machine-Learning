from pathlib import Path
import pickle
import json
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skexplain

from catboost import CatBoostClassifier


# ============================================================
# TELLA ET AL. (2026) ALE SETTINGS
# ============================================================

N_BINS = 20
N_BOOTSTRAP = 150
SUBSAMPLE = 10_000
RANDOM_SEED = 42

# Keep one job for reproducibility and stability on this PC.
N_JOBS = 1


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "Data/samples/training_samples_union_with_splits.csv"
)

MODEL_PATHS = {
    "LR": Path(
        "models/lr_union_random.joblib"
    ),

    "RF": Path(
        "models/rf_union_random.joblib"
    ),

    "XGBoost": Path(
        "models/xgb_union_random.joblib"
    ),

    "CatBoost": Path(
        "models/catboost_union_random.cbm"
    ),
}


ALE_DIR = Path(
    "results/ale/tella_protocol"
)

TABLE_DIR = (
    ALE_DIR
    / "tables"
)

FIG_DIR = Path(
    "results/figures/ale"
)


ALE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXACT FEATURE ORDER USED FOR MODEL TRAINING
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
# CHECK REQUIRED FILES
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Training dataset not found:\n"
        f"{DATA_PATH}"
    )


for model_name, model_path in MODEL_PATHS.items():

    if not model_path.exists():

        raise FileNotFoundError(
            f"{model_name} model not found:\n"
            f"{model_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


required_columns = (
    FEATURES
    + [
        "class",
        "random_split",
    ]
)


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# TELLA:
# ALE IS COMPUTED USING TRAINING DATA
# ============================================================

train_df = df[
    df["random_split"] == "train"
].copy()


X_train = train_df[
    FEATURES
].copy()


y_train = (
    train_df["class"]
    .astype(int)
    .to_numpy()
)


if len(X_train) != 15_400:

    print(
        "\nWARNING:"
        f" Expected 15,400 training observations,"
        f" found {len(X_train):,}."
    )


if X_train.isna().any().any():

    raise ValueError(
        "Missing values detected in ALE predictors."
    )


print("=" * 80)
print("TELLA-STYLE FIRST-ORDER ALE")
print("LOCAL MODELS — ALL PREDICTORS")
print("=" * 80)


print(
    f"\nTraining observations: "
    f"{len(X_train):,}"
)

print(
    f"Predictors: "
    f"{len(FEATURES)}"
)

print(
    "\nALE protocol:"
)

print(
    f"  First-order ALE"
)

print(
    f"  Training data only"
)

print(
    f"  Intervals:     {N_BINS}"
)

print(
    f"  Bootstraps:    {N_BOOTSTRAP}"
)

print(
    f"  Subsample:     {SUBSAMPLE:,}"
)

print(
    f"  Random seed:   {RANDOM_SEED}"
)

print(
    f"  Library:       scikit-explain "
    f"{version('scikit-explain')}"
)


# ============================================================
# SAVE METHOD CONFIGURATION
# ============================================================

config = {
    "method": "first-order ALE",
    "reference": "Tella et al. (2026)",
    "dataset": str(DATA_PATH),
    "data_subset": "random training split only",
    "training_n": int(
        len(X_train)
    ),
    "features": FEATURES,
    "n_bins": N_BINS,
    "n_bootstrap": N_BOOTSTRAP,
    "subsample": SUBSAMPLE,
    "random_seed": RANDOM_SEED,
    "class_index": 1,
    "estimator_output": "probability",
    "library": "scikit-explain",
    "library_version": version(
        "scikit-explain"
    ),
}


CONFIG_OUTPUT = (
    ALE_DIR
    / "ale_method_config.json"
)


with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# MODEL LOADER
# ============================================================

def load_model(
    model_name,
    model_path
):

    if model_name == "CatBoost":

        model = CatBoostClassifier()

        model.load_model(
            str(model_path)
        )

        return model


    return joblib.load(
        model_path
    )


# ============================================================
# EXTRACT ALE TABLES
#
# skexplain returns:
#
# feature__MODEL__ale
#     dimensions:
#     bootstrap x bins
#
# feature__bin_values
#     physical predictor values
#
# We save:
#   mean ALE
#   median ALE
#   2.5th percentile
#   97.5th percentile
#
# These are summaries of the 150 bootstrap ALE curves.
# ============================================================

def save_ale_tables(
    ale_ds,
    model_name
):

    for feature in FEATURES:

        ale_var = (
            f"{feature}__"
            f"{model_name}__ale"
        )

        bin_var = (
            f"{feature}"
            f"__bin_values"
        )


        if ale_var not in ale_ds:

            raise KeyError(
                f"Expected ALE variable missing:\n"
                f"{ale_var}\n\n"
                f"Available variables:\n"
                f"{list(ale_ds.data_vars)}"
            )


        if bin_var not in ale_ds:

            raise KeyError(
                f"Expected bin variable missing:\n"
                f"{bin_var}"
            )


        ale_values = np.asarray(
            ale_ds[
                ale_var
            ].values,
            dtype=float
        )


        bin_values = np.asarray(
            ale_ds[
                bin_var
            ].values,
            dtype=float
        )


        # Expected shape:
        # 150 bootstrap curves x 20 bins

        if ale_values.ndim != 2:

            raise ValueError(
                f"Unexpected ALE shape for "
                f"{model_name} / {feature}: "
                f"{ale_values.shape}"
            )


        if ale_values.shape[0] != N_BOOTSTRAP:

            raise ValueError(
                f"{model_name} / {feature}: "
                f"expected {N_BOOTSTRAP} bootstraps, "
                f"found {ale_values.shape[0]}"
            )


        if ale_values.shape[1] != len(
            bin_values
        ):

            raise ValueError(
                f"Bin count mismatch for "
                f"{model_name} / {feature}"
            )


        table = pd.DataFrame({
            "model":
                model_name,

            "feature":
                feature,

            "bin_value":
                bin_values,

            "ale_mean":
                np.mean(
                    ale_values,
                    axis=0
                ),

            "ale_median":
                np.median(
                    ale_values,
                    axis=0
                ),

            "ale_lower_95":
                np.percentile(
                    ale_values,
                    2.5,
                    axis=0
                ),

            "ale_upper_95":
                np.percentile(
                    ale_values,
                    97.5,
                    axis=0
                ),
        })


        output = (
            TABLE_DIR
            / (
                f"{model_name.lower()}"
                f"_{feature}_ale.csv"
            )
        )


        table.to_csv(
            output,
            index=False
        )


# ============================================================
# RUN ALE MODEL BY MODEL
#
# Running separately is intentional:
#
# 1. each model gets exactly the same ALE settings
# 2. one failure cannot destroy results already completed
# 3. raw outputs remain easy to inspect
# 4. CatBoost can use its native saved model
# ============================================================

completed_models = []


for model_name, model_path in MODEL_PATHS.items():

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_name,
        model_path
    )


    print(
        f"✓ {model_name} model loaded"
    )


    # --------------------------------------------------------
    # ExplainToolkit
    # --------------------------------------------------------

    explainer = skexplain.ExplainToolkit(
        estimators=(
            model_name,
            model
        ),
        X=X_train,
        y=y_train,
    )


    # --------------------------------------------------------
    # TELLA-STYLE FIRST-ORDER ALE
    # --------------------------------------------------------

    print(
        "\nComputing ALE for:"
    )


    for feature in FEATURES:

        print(
            f"  - {feature}"
        )


    ale_ds = explainer.ale(
        features=FEATURES,
        n_bins=N_BINS,
        n_bootstrap=N_BOOTSTRAP,
        subsample=SUBSAMPLE,
        n_jobs=N_JOBS,
        random_seed=RANDOM_SEED,
        class_index=1,
    )


    print(
        f"\n✓ {model_name} ALE calculation complete"
    )


    # --------------------------------------------------------
    # VALIDATE RESULT
    # --------------------------------------------------------

    if (
        ale_ds.attrs.get(
            "method"
        )
        != "ale"
    ):

        raise ValueError(
            f"{model_name}: returned object "
            f"is not labelled as ALE."
        )


    for feature in FEATURES:

        expected_ale_var = (
            f"{feature}__"
            f"{model_name}__ale"
        )

        expected_bin_var = (
            f"{feature}"
            f"__bin_values"
        )


        if expected_ale_var not in ale_ds:

            raise KeyError(
                f"{model_name}: missing "
                f"{expected_ale_var}"
            )


        if expected_bin_var not in ale_ds:

            raise KeyError(
                f"{model_name}: missing "
                f"{expected_bin_var}"
            )


    print(
        "✓ ALE output structure verified"
    )


    # --------------------------------------------------------
    # SAVE COMPLETE RAW SKEXPLAIN OBJECT
    # --------------------------------------------------------

    PKL_OUTPUT = (
        ALE_DIR
        / (
            f"{model_name.lower()}"
            f"_all_features_ale.pkl"
        )
    )


    with open(
        PKL_OUTPUT,
        "wb"
    ) as f:

        pickle.dump(
            ale_ds,
            f
        )


    print(
        f"✓ Raw ALE saved:\n"
        f"  {PKL_OUTPUT}"
    )


    # --------------------------------------------------------
    # SAVE NUMERIC TABLE FOR EVERY FEATURE
    # --------------------------------------------------------

    save_ale_tables(
        ale_ds,
        model_name
    )


    print(
        f"✓ ALE CSV tables saved for "
        f"all {len(FEATURES)} predictors"
    )


    # --------------------------------------------------------
    # SKEXPLAIN FIGURE
    #
    # Use the same plotting framework that generated
    # the successful CatBoost+slope test.
    # --------------------------------------------------------

    fig, axes = explainer.plot_ale(
        ale=ale_ds,
        features=FEATURES,
    )


    FIG_OUTPUT = (
        FIG_DIR
        / (
            f"{model_name.lower()}"
            f"_all_features_tella_protocol.png"
        )
    )


    fig.savefig(
        FIG_OUTPUT,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"✓ Figure saved:\n"
        f"  {FIG_OUTPUT}"
    )


    # --------------------------------------------------------
    # REPORT SHAPE
    # --------------------------------------------------------

    first_feature = FEATURES[0]

    example_var = (
        f"{first_feature}__"
        f"{model_name}__ale"
    )


    shape = ale_ds[
        example_var
    ].shape


    print(
        f"\nBootstrap × bins check: "
        f"{shape}"
    )


    if shape != (
        N_BOOTSTRAP,
        N_BINS
    ):

        raise ValueError(
            f"Expected "
            f"({N_BOOTSTRAP}, {N_BINS}), "
            f"received {shape}"
        )


    completed_models.append(
        model_name
    )


    print(
        f"\n✓ {model_name} PASSED"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TELLA-PROTOCOL LOCAL ALE COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nModels completed:"
)


for model_name in completed_models:

    print(
        f"  ✓ {model_name}"
    )


print(
    f"\nPredictors per model: "
    f"{len(FEATURES)}"
)

print(
    f"Total model-feature combinations: "
    f"{len(completed_models) * len(FEATURES)}"
)


print(
    "\nFrozen ALE protocol:"
)

print(
    "  First-order ALE"
)

print(
    "  Training data only"
)

print(
    f"  {N_BINS} intervals"
)

print(
    f"  {N_BOOTSTRAP} bootstrap repetitions"
)

print(
    f"  {SUBSAMPLE:,}-instance subsample"
)

print(
    "  Probability output, class = 1"
)

print(
    "  Library = scikit-explain"
)


print(
    "\nOutputs:"
)

print(
    f"  Raw ALE objects: {ALE_DIR}"
)

print(
    f"  Numeric tables:  {TABLE_DIR}"
)

print(
    f"  Figures:         {FIG_DIR}"
)