"""
20_compare_ale_all_models.py

Compare first-order ALE responses across five flood-susceptibility models:
LR, RF, XGBoost, CatBoost, and TabPFN.

Important methodological note
-----------------------------
LR/RF/XGBoost/CatBoost ALE tables come from the exact Tella-style
scikit-explain protocol (training data, 20 requested bins, 150 bootstraps,
10,000-row subsample).

TabPFN uses the same first-order ALE design but a cached-local-effect
bootstrap for computational feasibility. The central TabPFN implementation
was separately validated against scikit-explain for slope.

Because the TabPFN subsample/bin locations can differ slightly from the
four local models, this script NEVER assumes identical x coordinates across
all five models. Shape-comparison metrics are calculated after interpolation
within the common observed x-range only; no extrapolation is used.
"""

from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

LOCAL_TABLE_DIR = ROOT / "results" / "ale" / "tella_protocol" / "tables"

# Preferred location after extracting tabpfn_ale_all.zip
TABPFN_TABLE_DIR = ROOT / "results" / "ale" / "tabpfn" / "tables"

# Optional fallback if the extracted folder was kept under its ZIP name
if not TABPFN_TABLE_DIR.exists():
    fallback = ROOT / "results" / "ale" / "tabpfn_ale_all" / "tables"
    if fallback.exists():
        TABPFN_TABLE_DIR = fallback

OUT_DIR = ROOT / "results" / "ale" / "all_models_comparison"
FIG_DIR = ROOT / "results" / "figures" / "ale" / "combined_all5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# FROZEN MODEL / FEATURE NAMES
# ---------------------------------------------------------------------
MODELS = ["LR", "RF", "XGBoost", "CatBoost", "TabPFN"]
NONLINEAR_MODELS = ["RF", "XGBoost", "CatBoost", "TabPFN"]

PREFIX = {
    "LR": "lr",
    "RF": "rf",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "TabPFN": "tabpfn",
}

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

INTERP_POINTS = 401


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def expected_path(model: str, feature: str) -> Path:
    base = TABPFN_TABLE_DIR if model == "TabPFN" else LOCAL_TABLE_DIR
    return base / f"{PREFIX[model]}_{feature}_ale.csv"


def load_curve(model: str, feature: str) -> pd.DataFrame:
    path = expected_path(model, feature)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ALE table:\n{path}\n\n"
            "Expected local-model tables under:\n"
            f"  {LOCAL_TABLE_DIR}\n"
            "and TabPFN tables under:\n"
            f"  {TABPFN_TABLE_DIR}"
        )

    df = pd.read_csv(path).copy()

    required = {
        "model",
        "feature",
        "bin_value",
        "ale_mean",
        "ale_lower_95",
        "ale_upper_95",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    if len(df) < 2:
        raise ValueError(f"{path.name}: fewer than two ALE bins")

    numeric_cols = [
        "bin_value",
        "ale_mean",
        "ale_lower_95",
        "ale_upper_95",
    ]
    vals = df[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(f"{path.name}: NaN/inf detected")

    if not (df["ale_lower_95"] <= df["ale_upper_95"]).all():
        raise ValueError(f"{path.name}: invalid confidence bounds")

    df = df.sort_values("bin_value").reset_index(drop=True)

    # Duplicate x values would make interpolation ambiguous.
    if df["bin_value"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated bin_value entries")

    return df


def zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Linear interpolation of all mean-ALE zero crossings."""
    out = []
    for i in range(len(y) - 1):
        if y[i] == 0:
            out.append(float(x[i]))
        elif y[i] * y[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            z = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
            out.append(float(z))

    if y[-1] == 0:
        out.append(float(x[-1]))

    # Remove numerically duplicated crossings while preserving order.
    cleaned = []
    for z in out:
        if not cleaned or not np.isclose(z, cleaned[-1], rtol=1e-9, atol=1e-12):
            cleaned.append(z)
    return cleaned


def interpolate_pair(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Interpolate two curves only across their common x-range."""
    xa = df_a["bin_value"].to_numpy(float)
    ya = df_a["ale_mean"].to_numpy(float)
    xb = df_b["bin_value"].to_numpy(float)
    yb = df_b["ale_mean"].to_numpy(float)

    lo = max(xa.min(), xb.min())
    hi = min(xa.max(), xb.max())
    if not hi > lo:
        raise ValueError("Curves have no overlapping x-range")

    grid = np.linspace(lo, hi, INTERP_POINTS)
    a = np.interp(grid, xa, ya)
    b = np.interp(grid, xb, yb)
    return grid, a, b, lo, hi


# ---------------------------------------------------------------------
# LOAD + QC ALL 50 TABLES
# ---------------------------------------------------------------------
curves: dict[str, dict[str, pd.DataFrame]] = {}
long_parts = []
qc_rows = []

for feature in FEATURES:
    curves[feature] = {}
    for model in MODELS:
        df = load_curve(model, feature)
        curves[feature][model] = df

        out = df.copy()
        out["analysis_model"] = model
        out["analysis_feature"] = feature
        out["ale_implementation"] = (
            "exact_skexplain_tella_protocol"
            if model != "TabPFN"
            else "cached_local_effect_bootstrap_validated"
        )
        long_parts.append(out)

        qc_rows.append({
            "model": model,
            "feature": feature,
            "n_bins": len(df),
            "x_min": float(df["bin_value"].min()),
            "x_max": float(df["bin_value"].max()),
            "ale_min": float(df["ale_mean"].min()),
            "ale_max": float(df["ale_mean"].max()),
        })

long_df = pd.concat(long_parts, ignore_index=True, sort=False)
long_df.to_csv(OUT_DIR / "ale_all5_long.csv", index=False)
pd.DataFrame(qc_rows).to_csv(OUT_DIR / "ale_all5_qc.csv", index=False)


# ---------------------------------------------------------------------
# LOCAL-MODEL BIN CONSISTENCY CHECK
# TabPFN is intentionally NOT required to have identical bin locations.
# ---------------------------------------------------------------------
bin_rows = []
local_models = ["LR", "RF", "XGBoost", "CatBoost"]

for feature in FEATURES:
    ref = curves[feature][local_models[0]]["bin_value"].to_numpy(float)
    for model in local_models[1:]:
        x = curves[feature][model]["bin_value"].to_numpy(float)
        same = len(x) == len(ref) and np.allclose(x, ref, rtol=0, atol=1e-10)
        bin_rows.append({
            "feature": feature,
            "reference_model": local_models[0],
            "comparison_model": model,
            "same_bin_count": len(x) == len(ref),
            "same_bin_values": bool(same),
            "reference_n_bins": len(ref),
            "comparison_n_bins": len(x),
        })

bin_check = pd.DataFrame(bin_rows)
bin_check.to_csv(OUT_DIR / "ale_local_bin_consistency.csv", index=False)

if not bin_check["same_bin_values"].all():
    bad = bin_check.loc[~bin_check["same_bin_values"]]
    raise RuntimeError(
        "Exact local-model ALE bin locations are unexpectedly inconsistent:\n"
        + bad.to_string(index=False)
    )


# ---------------------------------------------------------------------
# PAIRWISE SHAPE COMPARISON
# Metrics are computed after interpolation on each pair's common x-range.
# Spearman = rank/shape agreement, not effect magnitude agreement.
# ---------------------------------------------------------------------
pair_rows = []

for feature in FEATURES:
    for model_a, model_b in combinations(MODELS, 2):
        df_a = curves[feature][model_a]
        df_b = curves[feature][model_b]
        grid, a, b, lo, hi = interpolate_pair(df_a, df_b)

        rho = spearmanr(a, b).statistic
        pearson = np.corrcoef(a, b)[0, 1]
        sign_agreement = np.mean(np.sign(a) == np.sign(b))

        pair_rows.append({
            "feature": feature,
            "model_a": model_a,
            "model_b": model_b,
            "spearman_shape": float(rho),
            "pearson_shape": float(pearson),
            "sign_agreement_fraction": float(sign_agreement),
            "common_x_min": float(lo),
            "common_x_max": float(hi),
            "interpolation_points": INTERP_POINTS,
        })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT_DIR / "ale_pairwise_shape_metrics_all5.csv", index=False)


# ---------------------------------------------------------------------
# NONLINEAR-MODEL CONSENSUS SUMMARY
# RF + XGBoost + CatBoost + TabPFN
# No arbitrary pass/fail threshold is imposed.
# ---------------------------------------------------------------------
consensus_rows = []

for feature in FEATURES:
    lo = max(
        curves[feature][m]["bin_value"].min()
        for m in NONLINEAR_MODELS
    )
    hi = min(
        curves[feature][m]["bin_value"].max()
        for m in NONLINEAR_MODELS
    )

    grid = np.linspace(lo, hi, INTERP_POINTS)
    matrix = []

    for model in NONLINEAR_MODELS:
        df = curves[feature][model]
        x = df["bin_value"].to_numpy(float)
        y = df["ale_mean"].to_numpy(float)
        matrix.append(np.interp(grid, x, y))

    matrix = np.vstack(matrix)

    pair_rhos = []
    pair_signs = []
    for i, j in combinations(range(len(NONLINEAR_MODELS)), 2):
        pair_rhos.append(spearmanr(matrix[i], matrix[j]).statistic)
        pair_signs.append(np.mean(np.sign(matrix[i]) == np.sign(matrix[j])))

    signs = np.sign(matrix)
    all4_same_sign = np.all(signs == signs[0], axis=0)
    n_pos = np.sum(signs > 0, axis=0)
    n_neg = np.sum(signs < 0, axis=0)
    at_least_3_same_sign = (n_pos >= 3) | (n_neg >= 3)

    consensus_rows.append({
        "feature": feature,
        "mean_pairwise_spearman_nonlin": float(np.mean(pair_rhos)),
        "min_pairwise_spearman_nonlin": float(np.min(pair_rhos)),
        "mean_pairwise_sign_agreement_nonlin": float(np.mean(pair_signs)),
        "all4_same_sign_fraction": float(np.mean(all4_same_sign)),
        "at_least_3of4_same_sign_fraction": float(np.mean(at_least_3_same_sign)),
        "common_x_min": float(lo),
        "common_x_max": float(hi),
        "interpolation_points": INTERP_POINTS,
    })

consensus_df = pd.DataFrame(consensus_rows)
consensus_df.to_csv(OUT_DIR / "ale_nonlinear_consensus_summary.csv", index=False)


# ---------------------------------------------------------------------
# DESCRIPTIVE MEAN-ALE ZERO CROSSINGS
# These are model-response transitions, NOT causal/physical thresholds.
# All crossings are retained because some curves cross zero multiple times.
# ---------------------------------------------------------------------
zero_rows = []

for feature in FEATURES:
    for model in MODELS:
        df = curves[feature][model]
        x = df["bin_value"].to_numpy(float)
        y = df["ale_mean"].to_numpy(float)
        crossings = zero_crossings(x, y)

        zero_rows.append({
            "feature": feature,
            "model": model,
            "n_zero_crossings": len(crossings),
            "zero_crossings": ";".join(f"{v:.10g}" for v in crossings),
        })

zero_df = pd.DataFrame(zero_rows)
zero_df.to_csv(OUT_DIR / "ale_zero_crossings_descriptive_all5.csv", index=False)


# ---------------------------------------------------------------------
# MODEL/METHOD RECORD
# ---------------------------------------------------------------------
method_df = pd.DataFrame([
    {
        "model": "LR",
        "ale_method": "scikit-explain",
        "bootstrap": "150 model-evaluation bootstrap replicates",
        "status": "exact Tella-style protocol",
    },
    {
        "model": "RF",
        "ale_method": "scikit-explain",
        "bootstrap": "150 model-evaluation bootstrap replicates",
        "status": "exact Tella-style protocol",
    },
    {
        "model": "XGBoost",
        "ale_method": "scikit-explain",
        "bootstrap": "150 model-evaluation bootstrap replicates",
        "status": "exact Tella-style protocol",
    },
    {
        "model": "CatBoost",
        "ale_method": "scikit-explain",
        "bootstrap": "150 model-evaluation bootstrap replicates",
        "status": "exact Tella-style protocol",
    },
    {
        "model": "TabPFN",
        "ale_method": "cached local effects",
        "bootstrap": "150 within-bin cached-local-effect bootstrap replicates",
        "status": "same ALE design; optimized implementation",
    },
])
method_df.to_csv(OUT_DIR / "ale_method_by_model.csv", index=False)


# ---------------------------------------------------------------------
# COMBINED FIGURES
# Curves are plotted at their OWN physical bin locations; TabPFN is not
# forced onto the local-model bins. 95% bootstrap intervals are shown.
# ---------------------------------------------------------------------
for feature in FEATURES:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in MODELS:
        df = curves[feature][model]
        x = df["bin_value"].to_numpy(float)
        y = df["ale_mean"].to_numpy(float)
        lo = df["ale_lower_95"].to_numpy(float)
        hi = df["ale_upper_95"].to_numpy(float)

        line, = ax.plot(x, y, marker="o", markersize=3, linewidth=1.6, label=model)
        ax.fill_between(x, lo, hi, alpha=0.10, color=line.get_color())

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_xlabel(feature)
    ax.set_ylabel("Accumulated Local Effect on predicted flood probability")
    ax.set_title(f"ALE comparison across five models — {feature}")
    ax.legend(frameon=False, ncol=3)
    ax.grid(False)
    fig.tight_layout()

    fig.savefig(FIG_DIR / f"ale_compare_all5_{feature}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"ale_compare_all5_{feature}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# MACHINE-READABLE CONFIG / METHOD NOTE
# ---------------------------------------------------------------------
config = {
    "models": MODELS,
    "nonlinear_models": NONLINEAR_MODELS,
    "features": FEATURES,
    "central_curve_used_for_comparison": "ale_mean",
    "interpolation_points": INTERP_POINTS,
    "interpolation_rule": (
        "Linear interpolation only within the common observed x-range; "
        "no extrapolation. Used only for cross-model shape/sign metrics."
    ),
    "plot_rule": (
        "Each model is plotted at its original ALE bin_value coordinates."
    ),
    "zero_crossing_warning": (
        "Zero crossings are descriptive model-response transitions, not "
        "causal or physical thresholds. Multiple crossings are retained."
    ),
    "tabpfn_note": (
        "TabPFN uncertainty uses cached-local-effect bootstrap rather than "
        "repeated GPU model inference. Central slope ALE was separately "
        "validated against scikit-explain."
    ),
}

with open(OUT_DIR / "ale_all5_comparison_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)


# ---------------------------------------------------------------------
# FINAL CONSOLE SUMMARY
# ---------------------------------------------------------------------
print("=" * 80)
print("ALL-FIVE-MODEL ALE COMPARISON COMPLETE")
print("=" * 80)
print(f"Loaded: {len(MODELS) * len(FEATURES)} ALE tables")
print("Local LR/RF/XGBoost/CatBoost bin consistency: PASS")
print("TabPFN bin locations: handled independently (no forced matching)")
print(f"\nTables saved to:\n  {OUT_DIR}")
print(f"\nFigures saved to:\n  {FIG_DIR}")
print("\nNonlinear-model consensus summary:")
print(
    consensus_df[
        [
            "feature",
            "mean_pairwise_spearman_nonlin",
            "mean_pairwise_sign_agreement_nonlin",
            "all4_same_sign_fraction",
            "at_least_3of4_same_sign_fraction",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
)
print("\nIMPORTANT: zero crossings are descriptive ALE transitions, not causal thresholds.")
