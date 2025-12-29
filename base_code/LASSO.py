#!/usr/bin/env python3
"""
LASSO regression on PSD indices with DB_2/PSD merge,
flag filtering ("include"), and robust error checking.
Now includes operational variables:
  - A_Flow  (airflow, from DB_2/DB)
  - Diaphragm_on (binary, derived from Test_procedure: 1=STD, 0=No_pres/other)

Also fits:
  - Random Forest Regressor
  - Gradient Boosting Regressor
for non-linear comparison.

Also includes interaction features so LASSO can model
"diaphragm effect depends on PSD":
  - D10_Dia       = D10 * Diaphragm_on
  - Span_Dia      = (D80/D20) * Diaphragm_on
  - D10_Span_Dia  = D10 * (D80/D20) * Diaphragm_on   # diaphragm-specific curvature

Training is done on a subset of sheet 'DB_2'.
Internal validation uses a held-out subset of DB_2.
External validation / Predicted vs Actual uses sheet 'DB'
for selected sample codes (Si_M, Si_F, Si_Rep, Si_Rep_new, Si_BM).

Only R2 / RMSE on held-out data (internal + external)
are reported for the main LASSO model. No training R2 is printed.
"""

import argparse
import sys
import textwrap
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from typing import Optional
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_DB_SHEET = "DB_2"   # main experimental sheet (for training + internal val)
VAL_DB_SHEET = "DB"         # external validation sheet
DEFAULT_PSD_SHEET = "PSD"

DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"

DEFAULT_TARGET_COL = "Mc_%"

# PSD + operational + interaction features
DEFAULT_FEATURE_COLS = [
    # PSD indices
    "D10",
    # "D20",
    # "D50",
    # "D80",
    # "D90",
    # "D90/D50",
    # "D50/D10",
    "D80/D20",
    # operational variables from DB_2 / DB
    # "A_Flow",
    "Diaphragm_on",
    # interactions (PSD x Diaphragm)
    "D10_Dia",       # D10 * Diaphragm_on
    "Span_Dia",      # (D80/D20) * Diaphragm_on
    # quadratic / interaction terms
    "D10^2",
    "D10_Span_Dia",  # D10 * (D80/D20) * Diaphragm_on  <-- NEW TERM FOR CURVED ?FMC
]


SAMPLE_CODE_COL = "Sample Code"
FLAG_COL = "flag"
TEST_PROC_COL = "Test_procedure"
AFLOW_COL = "A_Flow"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def error(msg: str):
    sys.stderr.write(f"\nERROR: {msg}\n\n")
    sys.exit(1)


def load_sheets(xlsx_path: str, sheet_db: str, sheet_psd: str):
    """Load DB_x and PSD sheets with error checking."""
    try:
        xls = pd.ExcelFile(xlsx_path)
    except FileNotFoundError:
        error(f"Excel file not found: {xlsx_path}")
    except Exception as e:
        error(f"Failed to open Excel file '{xlsx_path}': {e}")

    if sheet_db not in xls.sheet_names:
        error(f"DB sheet '{sheet_db}' not found. Sheets: {xls.sheet_names}")

    if sheet_psd not in xls.sheet_names:
        error(f"PSD sheet '{sheet_psd}' not found. Sheets: {xls.sheet_names}")

    try:
        df_db = pd.read_excel(xls, sheet_name=sheet_db)
        df_psd = pd.read_excel(xls, sheet_name=sheet_psd)
    except Exception as e:
        error(f"Failed reading sheets: {e}")

    return df_db, df_psd


def check_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str):
    """Ensure required columns exist in a DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        error(
            f"Missing required columns in '{df_name}' sheet: {missing}\n"
            f"Found: {list(df.columns)}"
        )


def apply_flag_filter(df_db: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where flag contains 'include' (case-insensitive)."""
    if FLAG_COL not in df_db.columns:
        error(f"Flag column '{FLAG_COL}' missing in DB sheet.")

    mask = df_db[FLAG_COL].astype(str).str.contains("include", case=False, na=False)
    df = df_db.loc[mask].copy()

    if df.empty:
        error("No rows remain after flag filter ('include').")

    return df


def merge_db_psd(df_db, df_psd, features, target_col):
    """
    Merge DB_x and PSD on sample code, recompute ratio features,
    engineer operational & interaction variables, drop rows with
    missing data, and return df, X, y.
    """

    # Base PSD D-values required from PSD sheet
    base_needed = ["D10", "D20", "D50", "D80", "D90"]

    # Required DB columns (for target, flags and operational vars)
    db_required = [SAMPLE_CODE_COL, target_col, FLAG_COL, TEST_PROC_COL, AFLOW_COL]

    check_required_columns(df_db, db_required, "DB")
    check_required_columns(df_psd, [SAMPLE_CODE_COL] + base_needed, "PSD")

    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner")

    if df.empty:
        error("Merge between DB and PSD sheets is empty. Check sample codes.")

    # Convert base D-values to float
    for col in base_needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recompute ratios (override Excel values if present)
    df["D80/D20"] = df["D80"] / df["D20"]
    df["D90/D50"] = df["D90"] / df["D50"]
    df["D50/D10"] = df["D50"] / df["D10"]

    # Operational variables
    df[AFLOW_COL] = pd.to_numeric(df[AFLOW_COL], errors="coerce")

    # Diaphragm_on: 1 if Test_procedure contains "STD", else 0
    df["Diaphragm_on"] = df[TEST_PROC_COL].astype(str).str.contains(
        "STD", case=False, na=False
    ).astype(int)

    # Interaction terms with diaphragm
    df["D10_Dia"] = df["D10"] * df["Diaphragm_on"]
    df["Span_Dia"] = df["D80/D20"] * df["Diaphragm_on"]

    # Quadratic term
    df["D10^2"] = df["D10"] ** 2

    # NEW: diaphragm-specific bilinear term D10 * Span * Dia
    df["D10_Span_Dia"] = df["D10"] * df["D80/D20"] * df["Diaphragm_on"]

    # Select only needed columns
    cols_to_keep = [SAMPLE_CODE_COL, target_col] + features
    df = df[cols_to_keep].copy()

    # Identify rows with missing data
    missing_mask = df[[target_col] + features].isna().any(axis=1)
    dropped = df.loc[missing_mask]

    if not dropped.empty:
        print("\nWarning: dropped rows due to NaNs in DB/PSD merge:\n")
        print(dropped.to_string(index=False))
        print("\n--- End dropped row report ---\n")

    df = df.loc[~missing_mask].copy()

    if df.empty:
        error("All rows dropped due to missing values in target/features.")

    X = df[features].to_numpy(float)
    y = df[target_col].to_numpy(float)

    return df, X, y


# ---------------------------------------------------------------------
# LASSO Fit & Summary (coefficients only - no training R2)
# ---------------------------------------------------------------------

def fit_lasso(X, y) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=0, n_alphas=100, max_iter=10000)),
    ])
    pipe.fit(X, y)
    return pipe


def summarise_lasso(pipe: Pipeline, features: list[str]):
    """
    Print summary of fitted LASSO model:
    - alpha
    - coefficients and which features were kept/dropped
    No training R2 is reported here.
    """
    lasso: LassoCV = pipe.named_steps["lasso"]

    print("\n================ LASSO Regression Summary (coefficients only) ================")
    print(f"Chosen alpha (lambda): {lasso.alpha_:.6g}")

    coefs = lasso.coef_
    coef_table = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("\nCoefficients (sorted by |coefficient|):")
    print(coef_table[["feature", "coefficient"]].to_string(index=False))

    non_zero = coef_table[coef_table["coefficient"] != 0.0]
    zero = coef_table[coef_table["coefficient"] == 0.0]

    print("\nNon-zero coefficients (selected features):")
    if non_zero.empty:
        print("  None (all coefficients shrank to zero!)")
    else:
        print(non_zero.to_string(index=False))

    print("\nZero coefficients (dropped features):")
    if zero.empty:
        print("  None")
    else:
        print(zero["feature"].to_string(index=False))

    print("\nIntercept:", lasso.intercept_)
    print("==========================================================================\n")


# ---------------------------------------------------------------------
# Tree-based models (RF & GB)  -- definitions left but unused
# ---------------------------------------------------------------------

def fit_random_forest(X, y):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=0
    )
    rf.fit(X, y)
    return rf


def fit_gradient_boosting(X, y):
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=3,
        random_state=0
    )
    gb.fit(X, y)
    return gb


def report_held_out_metrics(model, X_val, y_val, label: str):
    """Print R2 and RMSE for any model on held-out data."""
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{label} R^2:  {r2:.4f}")
    print(f"{label} RMSE: {rmse:.4f}\n")
    return y_pred, r2, rmse


# ---------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_codes: Optional[np.ndarray] = None,
    title: str = "Predicted vs Actual",
    save_path: Optional[str] = None,
):
    """
    Predicted vs Actual FMC (in %) scatter plot.

    If sample_codes is provided, points are colour-coded by Sample Code
    and a legend is added.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Convert to percent for plotting
    y_true_pct = 100 * y_true
    y_pred_pct = 100 * y_pred

    min_val = 0.0
    max_val = max(np.max(y_true_pct), np.max(y_pred_pct)) * 1.05

    if sample_codes is not None:
        sample_codes = np.asarray(sample_codes)
        unique_codes = np.unique(sample_codes)

        # use a categorical colormap
        cmap = plt.get_cmap("tab20")
        n_colors = cmap.N

        for i, code in enumerate(unique_codes):
            mask = sample_codes == code
            ax.scatter(
                y_true_pct[mask],
                y_pred_pct[mask],
                alpha=0.8,
                s=40,
                label=str(code),
                color=cmap(i % n_colors),
                edgecolors="k",
                linewidths=0.3,
            )
    else:
        ax.scatter(y_true_pct, y_pred_pct, alpha=0.8)

    # 1:1 line
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", label="1:1 line", color="grey")

    # best-fit line (on all points)
    m, c = np.polyfit(y_true_pct, y_pred_pct, 1)
    ax.plot([min_val, max_val],
            [m * min_val + c, m * max_val + c],
            linestyle="-", label=f"Best fit (slope={m:.2f})")

    r2 = r2_score(y_true, y_pred)
    rmse_pct = 100 * np.sqrt(mean_squared_error(y_true, y_pred))

    ax.set_xlabel("Measured FMC (%)")
    ax.set_ylabel("Predicted FMC (%)")
    ax.set_title(title)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

    # put legend outside if there are many sample codes
    if sample_codes is not None:
        ax.legend(
            title="Sample Code",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=7
        )
    else:
        ax.legend()

    # Text box with metrics (in % units)
    txt = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse_pct:.3f}\\,\\%$"
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pdp_2d_d20_span(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    feature_x: str = "D10",
    feature_y: str = "D80/D20",
    nx: int = 40,
    ny: int = 40,
    title: str = "Predicted FMC as a function of D10 and D80/D20",
    save_path: Optional[str] = None,
    delta_threshold: float = 0.0,  # boundary where ?FMC = threshold (default = 0)
):
    """
    2D PDP of predicted FMC vs D10 and Span (D80/D20).

    Overlays a red curve where the diaphragm effect becomes significant:
        ?FMC = FMC(diaphragm on) - FMC(no diaphragm)

    By default, the red curve is the ?FMC = 0 contour
    (i.e., where the diaphragm stops helping / starts helping).
    """

    # Ensure both features exist
    for f in (feature_x, feature_y):
        if f not in features:
            raise ValueError(f"{f} is not in feature list {features}")

    # Work from the mean feature vector as baseline
    X = df[features].to_numpy(dtype=float)
    base = X.mean(axis=0)

    ix = features.index(feature_x)
    iy = features.index(feature_y)

    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), nx)
    y_vals = np.linspace(df[feature_y].min(), df[feature_y].max(), ny)

    XX, YY = np.meshgrid(x_vals, y_vals)

    # Start from the mean row and vary D10 and Span
    grid = np.tile(base, (nx * ny, 1))
    grid[:, ix] = XX.ravel()
    grid[:, iy] = YY.ravel()

    # Base grid DataFrame
    grid_df = pd.DataFrame(grid, columns=features)

    # ------------------------------------------------------------------
    # 1) Predicted FMC surface (for the current "mean" diaphragm state)
    # ------------------------------------------------------------------
    # Recompute engineered features consistent with grid_df values
    if "D10^2" in features:
        grid_df["D10^2"] = grid_df["D10"] ** 2
    if "D10_Span_Dia" in features and "D80/D20" in features and "Diaphragm_on" in features:
        grid_df["D10_Span_Dia"] = grid_df["D10"] * grid_df["D80/D20"] * grid_df["Diaphragm_on"]
    if "D10_Dia" in features and "Diaphragm_on" in features:
        grid_df["D10_Dia"] = grid_df["D10"] * grid_df["Diaphragm_on"]
    if "Span_Dia" in features and "D80/D20" in features and "Diaphragm_on" in features:
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]

    Z = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float)).reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(XX, YY, Z, levels=15)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted FMC (%)")

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(title)

    # ------------------------------------------------------------------
    # 2) Diaphragm effect ?FMC(D10, Span) from the same model
    # ------------------------------------------------------------------
    required_for_delta = {"Diaphragm_on", "D10_Dia", "Span_Dia"}
    if required_for_delta.issubset(set(features)):
        # Build two copies: dia=0 and dia=1, recomputing interactions
        grid_off = grid_df.copy()
        grid_on = grid_df.copy()

        # diaphragm OFF
        grid_off["Diaphragm_on"] = 0
        grid_off["D10_Dia"] = grid_off["D10"] * grid_off["Diaphragm_on"]
        grid_off["Span_Dia"] = grid_off["D80/D20"] * grid_off["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_off["D10_Span_Dia"] = (
                grid_off["D10"] * grid_off["D80/D20"] * grid_off["Diaphragm_on"]
            )

        # diaphragm ON
        grid_on["Diaphragm_on"] = 1
        grid_on["D10_Dia"] = grid_on["D10"] * grid_on["Diaphragm_on"]
        grid_on["Span_Dia"] = grid_on["D80/D20"] * grid_on["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_on["D10_Span_Dia"] = (
                grid_on["D10"] * grid_on["D80/D20"] * grid_on["Diaphragm_on"]
            )

        # Predict FMC (%) for both states
        Z_off = 100 * pipe.predict(grid_off[features].to_numpy(dtype=float)).reshape(ny, nx)
        Z_on  = 100 * pipe.predict(grid_on[features].to_numpy(dtype=float)).reshape(ny, nx)

        # Diaphragm effect
        # ?FMC = FMC(dia on) - FMC(no dia)
        Z_delta = Z_on - Z_off

        # Contour where ?FMC = delta_threshold (usually 0)
        cs = ax.contour(
            XX,
            YY,
            Z_delta,
            levels=[delta_threshold],
            colors="red",
            linewidths=2.0,
        )

        # Optional label on the curve
        label_str = f"?FMC = {delta_threshold:.1f}"
        ax.clabel(cs, fmt={delta_threshold: label_str}, inline=True, fontsize=8)

        # Small text box explaining sign convention
        ax.text(
            0.02,
            0.02,
            "?FMC = FMC(dia) - FMC(no dia)\n"
            "Below red curve: diaphragm reduces FMC",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    else:
        # If for some reason interaction features aren't in the model,
        # we silently skip the diaphragm boundary.
        print(
            "Warning: cannot compute diaphragm boundary - "
            "features do not include Diaphragm_on, D10_Dia, Span_Dia."
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pdp_d10_diaphragm(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    1D PDP for D10 with two curves:
      - Diaphragm_on = 0
      - Diaphragm_on = 1

    Interaction features (D10_Dia, Span_Dia, D10_Span_Dia) are recomputed so they stay
    consistent with the chosen Diaphragm_on and D10.
    """
    required = {"D10", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20"}
    if not required.issubset(features):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df["D10"].min(), df["D10"].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D10"] = x_vals
        grid_df["Diaphragm_on"] = dia
        # recompute interactions correctly using D10
        grid_df["D10_Dia"] = grid_df["D10"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_df["D10_Span_Dia"] = (
                grid_df["D10"] * grid_df["D80/D20"] * grid_df["Diaphragm_on"]
            )
        if "D10^2" in features:
            grid_df["D10^2"] = grid_df["D10"] ** 2

        y_pred = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, preds[0], label="No diaphragm", linewidth=2)
    ax.plot(x_vals, preds[1], label="Diaphragm on", linewidth=2)

    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("Predicted FMC (%)")
    ax.set_title("Effect of diaphragm across D10 (Model)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_delta_d10_diaphragm(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot ?FMC(D10) = FMC(diaphragm on) - FMC(no diaphragm) vs D10
    for the LASSO model.
    """
    required = {"D10", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20"}
    if not required.issubset(features):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df["D10"].min(), df["D10"].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D10"] = x_vals
        grid_df["Diaphragm_on"] = dia
        grid_df["D10_Dia"] = grid_df["D10"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_df["D10_Span_Dia"] = (
                grid_df["D10"] * grid_df["D80/D20"] * grid_df["Diaphragm_on"]
            )
        if "D10^2" in features:
            grid_df["D10^2"] = grid_df["D10"] ** 2

        y_pred = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    delta = preds[1] - preds[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, delta, linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title("Effect of diaphragm vs D10 (Model)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_delta_dia_tree(
    model,
    df: pd.DataFrame,
    features: list[str],
    feature_x: str = "D20",
    n_points: int = 60,
    model_name: str = "Tree model",
    save_path: Optional[str] = None,
):
    """
    Compute the diaphragm effect using a NON-LINEAR model (RF or GB):

        ?FMC(feature_x) = FMC_dia(feature_x) - FMC_no_dia(feature_x)

    If this curve is curved (not a straight line), the diaphragm
    effect is genuinely nonlinear in that variable.
    """
    required = {"D20", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20", "D10"}
    if not required.issubset(set(features)):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df[feature_x] = x_vals
        grid_df["Diaphragm_on"] = dia
        # D10 stays at mean; D10_Dia uses D10, not feature_x
        grid_df["D10_Dia"] = grid_df["D10"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_df["D10_Span_Dia"] = (
                grid_df["D10"] * grid_df["D80/D20"] * grid_df["Diaphragm_on"]
            )
        if "D10^2" in features:
            grid_df["D10^2"] = grid_df["D10"] ** 2

        y_pred = 100 * model.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    delta = preds[1] - preds[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, delta, linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel(f"{feature_x} (\u00B5m)" if "D" in feature_x else feature_x)
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title(f"Nonlinear diaphragm effect vs {feature_x} ({model_name})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# NEW: Empirical diaphragm plots from data (no ML model)
# ---------------------------------------------------------------------

def plot_empirical_d10_diaphragm(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    title: str = "FMC vs D10 for diaphragm on/off",
    save_path: Optional[str] = None,
):
    """
    X-axis = D10
    Labels = Span (D80/D20)
    Thick connection lines drawn between dia_off and dia_on for same D10 cluster.
    """
    required = {"D10", "D80/D20", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required}")

    df0 = df[df["Diaphragm_on"] == 0].copy()
    df1 = df[df["Diaphragm_on"] == 1].copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    # convert to %
    df0["Mc_pct"] = 100 * df0[target_col]
    df1["Mc_pct"] = 100 * df1[target_col]

    # scatter
    ax.scatter(df0["D10"], df0["Mc_pct"], label="No diaphragm", alpha=0.8, edgecolors="k")
    ax.scatter(df1["D10"], df1["Mc_pct"], label="Diaphragm on", alpha=0.8, edgecolors="k")

    # ---------- thick connection lines ----------
    unique_d10 = sorted(df["D10"].unique())
    for d in unique_d10:
        g0 = df0[df0["D10"] == d]["Mc_pct"]
        g1 = df1[df1["D10"] == d]["Mc_pct"]
        if len(g0) > 0 and len(g1) > 0:
            ax.plot(
                [d, d],
                [g0.mean(), g1.mean()],
                color="grey",
                linewidth=2.5,
                alpha=0.8,
            )

    # ---------- cluster labels (Span) ----------
    df_all = df.copy()
    df_all["Mc_pct"] = 100 * df_all[target_col]

    stats = (
        df_all
        .groupby("D10")
        .agg(
            Span_mean=("D80/D20", "mean"),
            Mc_max=("Mc_pct", "max"),
        )
        .reset_index()
    )

    y_range = df_all["Mc_pct"].max() - df_all["Mc_pct"].min()
    x_range = df_all["D10"].max() - df_all["D10"].min()
    # smaller vertical offset (closer to line)
    y_offset = 0.06 * y_range if y_range > 0 else 1.0
    # small horizontal offset to dodge labels left/right
    x_offset = 0.03 * x_range if x_range > 0 else 0.2

    for i, (_, row) in enumerate(stats.iterrows()):
        x_line = row["D10"]
        y_line_top = row["Mc_max"]
        # alternate left/right offsets
        direction = -1 if i % 2 == 0 else 1
        x_lab = x_line + direction * x_offset
        y_lab = y_line_top + y_offset

        # thin connector line from cluster to label
        ax.plot(
            [x_line, x_lab],
            [y_line_top, y_lab],
            color="grey",
            linewidth=0.8,
            alpha=0.8,
        )

        ax.text(
            x_lab,
            y_lab,
            f"{row['Span_mean']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("Measured FMC (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_empirical_span_diaphragm(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    title: str = "FMC vs Span (D80/D20) for diaphragm on/off",
    save_path: Optional[str] = None,
):
    required_cols = {"D80/D20", "D10", "Diaphragm_on", target_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required_cols}")

    df0 = df[df["Diaphragm_on"] == 0].copy()
    df1 = df[df["Diaphragm_on"] == 1].copy()

    df0["Mc_pct"] = 100 * df0[target_col]
    df1["Mc_pct"] = 100 * df1[target_col]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(df0["D80/D20"], df0["Mc_pct"], label="No diaphragm", alpha=0.8, edgecolors="k")
    ax.scatter(df1["D80/D20"], df1["Mc_pct"], label="Diaphragm on", alpha=0.8, edgecolors="k")

    # ---------- thick connection lines ----------
    unique_span = sorted(df["D80/D20"].unique())
    for s in unique_span:
        g0 = df0[df0["D80/D20"] == s]["Mc_pct"]
        g1 = df1[df1["D80/D20"] == s]["Mc_pct"]
        if len(g0) > 0 and len(g1) > 0:
            ax.plot(
                [s, s],
                [g0.mean(), g1.mean()],
                color="grey",
                linewidth=2.5,
                alpha=0.8,
            )

    # ---------- cluster labels (D10) ----------
    df_all = df.copy()
    df_all["Mc_pct"] = 100 * df_all[target_col]

    span_stats = (
        df_all
        .groupby("D80/D20")
        .agg(
            D10_mean=("D10", "mean"),
            Mc_max=("Mc_pct", "max"),
        )
        .reset_index()
    )

    y_range = df_all["Mc_pct"].max() - df_all["Mc_pct"].min()
    x_range = df_all["D80/D20"].max() - df_all["D80/D20"].min()
    y_offset = 0.06 * y_range if y_range > 0 else 1.0
    x_offset = 0.03 * x_range if x_range > 0 else 0.1

    for i, (_, row) in enumerate(span_stats.iterrows()):
        x_line = row["D80/D20"]
        y_line_top = row["Mc_max"]
        direction = -1 if i % 2 == 0 else 1
        x_lab = x_line + direction * x_offset
        y_lab = y_line_top + y_offset

        # thin connector line
        ax.plot(
            [x_line, x_lab],
            [y_line_top, y_lab],
            color="grey",
            linewidth=0.8,
            alpha=0.8,
        )

        ax.text(
            x_lab,
            y_lab,
            f"{row['D10_mean']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("Measured FMC (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_empirical_delta_span_diaphragm(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    n_points: int = 10,
    save_path: Optional[str] = None,
):
    """
    Empirical ?FMC(Span) curve using binned means of the raw data.

    Steps:
      - Bin Span (D80/D20) into n_points bins.
      - For each bin, compute mean Mc% for diaphragm_on = 0 and 1.
      - ?FMC(Span_bin) = mean(Mc%, dia=1) - mean(Mc%, dia=0).

    No linear regression / best-fit lines are used.
    """
    if "D80/D20" not in df.columns or "Diaphragm_on" not in df.columns:
        raise ValueError("Dataframe must contain 'D80/D20' and 'Diaphragm_on' columns.")

    data = df[["D80/D20", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    span_min = data["D80/D20"].min()
    span_max = data["D80/D20"].max()
    if span_min == span_max:
        print("Span (D80/D20) has no variation; cannot build empirical delta curve.")
        return

    bins = np.linspace(span_min, span_max, n_points + 1)
    data["Span_bin"] = pd.cut(data["D80/D20"], bins=bins, include_lowest=True)

    grouped = (
        data
        .groupby(["Span_bin", "Diaphragm_on"])
        .agg(
            Span_mean=("D80/D20", "mean"),
            Mc_mean=("Mc_pct", "mean"),
        )
        .reset_index()
    )

    pivot = grouped.pivot(
        index="Span_bin",
        columns="Diaphragm_on",
        values=["Span_mean", "Mc_mean"],
    )

    required_cols = [("Span_mean", 0), ("Span_mean", 1), ("Mc_mean", 0), ("Mc_mean", 1)]
    for col in required_cols:
        if col not in pivot.columns:
            print("Not enough overlapping Span bins with both diaphragm states to build empirical delta curve.")
            return

    pivot = pivot.dropna(subset=required_cols)
    if pivot.empty:
        print("No bins with both diaphragm_on=0 and 1; cannot build empirical delta curve.")
        return

    x_vals = 0.5 * (pivot[("Span_mean", 0)].to_numpy() + pivot[("Span_mean", 1)].to_numpy())
    delta = pivot[("Mc_mean", 1)].to_numpy() - pivot[("Mc_mean", 0)].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, delta, marker="o", linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title("Effect of diaphragm vs Span")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_empirical_delta_d10_diaphragm(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    n_points: int = 10,
    save_path: Optional[str] = None,
):
    """
    Empirical ?FMC(D10) curve using binned means of the raw data.

    Steps:
      - Bin D10 into n_points bins.
      - For each bin, compute mean Mc% for diaphragm_on = 0 and 1.
      - ?FMC(D10_bin) = mean(Mc%, dia=1) - mean(Mc%, dia=0).

    No linear regression / best-fit lines are used.
    """
    if "D10" not in df.columns or "Diaphragm_on" not in df.columns:
        raise ValueError("Dataframe must contain 'D10' and 'Diaphragm_on' columns.")

    # Work on a copy and convert Mc to %
    data = df[["D10", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    # Define D10 bins
    d10_min = data["D10"].min()
    d10_max = data["D10"].max()
    if d10_min == d10_max:
        print("D10 has no variation; cannot build empirical delta curve.")
        return

    # n_points = number of bins
    bins = np.linspace(d10_min, d10_max, n_points + 1)
    data["D10_bin"] = pd.cut(data["D10"], bins=bins, include_lowest=True)

    # Compute mean D10 and Mc% in each bin & diaphragm state
    grouped = (
        data
        .groupby(["D10_bin", "Diaphragm_on"])
        .agg(
            D10_mean=("D10", "mean"),
            Mc_mean=("Mc_pct", "mean"),
        )
        .reset_index()
    )

    # Pivot to get columns for dia=0 and dia=1
    pivot = grouped.pivot(
        index="D10_bin",
        columns="Diaphragm_on",
        values=["D10_mean", "Mc_mean"],
    )

    # We only keep bins that have both dia=0 and dia=1
    required_cols = [("D10_mean", 0), ("D10_mean", 1), ("Mc_mean", 0), ("Mc_mean", 1)]
    for col in required_cols:
        if col not in pivot.columns:
            print("Not enough overlapping D10 bins with both diaphragm states to build empirical delta curve.")
            return

    pivot = pivot.dropna(subset=required_cols)

    if pivot.empty:
        print("No bins with both diaphragm_on=0 and 1; cannot build empirical delta curve.")
        return

    # x = average D10 in the bin (we can take mean of the two means)
    x_vals = 0.5 * (pivot[("D10_mean", 0)].to_numpy() + pivot[("D10_mean", 1)].to_numpy())
    # ?FMC = Mc(dia=1) - Mc(dia=0)
    delta = pivot[("Mc_mean", 1)].to_numpy() - pivot[("Mc_mean", 0)].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, delta, marker="o", linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title("Effect of diaphragm vs D10")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_delta_span_diaphragm(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot ?FMC(Span) = FMC(diaphragm on) - FMC(no diaphragm) vs Span (D80/D20)
    for the LASSO model.

    D10 is held at its mean; Span (D80/D20) is varied and Span_Dia and D10_Span_Dia are updated.
    """
    required = {"D80/D20", "Diaphragm_on", "D10_Dia", "Span_Dia", "D10"}
    if not required.issubset(set(features)):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    span_min = df["D80/D20"].min()
    span_max = df["D80/D20"].max()
    span_vals = np.linspace(span_min, span_max, n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D80/D20"] = span_vals
        grid_df["Diaphragm_on"] = dia

        # Interactions: Span_Dia uses Span, D10_Dia uses D10 (held at mean)
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]
        grid_df["D10_Dia"] = grid_df["D10"] * grid_df["Diaphragm_on"]
        if "D10_Span_Dia" in features:
            grid_df["D10_Span_Dia"] = (
                grid_df["D10"] * grid_df["D80/D20"] * grid_df["Diaphragm_on"]
            )
        if "D10^2" in features:
            grid_df["D10^2"] = grid_df["D10"] ** 2

        y_pred = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    delta = preds[1] - preds[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(span_vals, delta, linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title("Effect of diaphragm vs Span (Model)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LASSO + tree-based models on PSD indices with DB_2 training (with hold-out) and DB external validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python %(prog)s
              python %(prog)s custom_file.xlsx
            """
        ),
    )

    parser.add_argument(
        "xlsx_path",
        nargs="?",
        default=DEFAULT_XLSX_PATH,
        help=f"Path to the Excel workbook (default: {DEFAULT_XLSX_PATH})",
    )
    parser.add_argument("--sheet_db_train", default=DEFAULT_DB_SHEET)
    parser.add_argument("--sheet_db_val", default=VAL_DB_SHEET)
    parser.add_argument("--sheet_psd", default=DEFAULT_PSD_SHEET)
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL)

    args = parser.parse_args()

    print(f"\nUsing Excel file:\n  {args.xlsx_path}\n")

    # ---------- DB_2: load and merge (for training + internal validation) ----------
    df_db_train_full, df_psd = load_sheets(args.xlsx_path, args.sheet_db_train, args.sheet_psd)
    # Use ALL rows from DB_2 (no flag filter)
    df_db_train_inc = df_db_train_full

    df_train_merged, X_all, y_all = merge_db_psd(
        df_db_train_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
    )

    print(f"Total usable rows from sheet '{args.sheet_db_train}' after merge & filtering: {len(df_train_merged)}")
    print(f"Features: {DEFAULT_FEATURE_COLS}")
    print(f"Target:   {args.target_col}\n")

    df_features_train = df_train_merged[DEFAULT_FEATURE_COLS].copy()
    sample_codes_all = df_train_merged[SAMPLE_CODE_COL].to_numpy()

    # ---------- INTERNAL HOLD-OUT SPLIT (DB_2) ----------
    X_train, X_val_int, y_train, y_val_int, codes_train, codes_val_int = train_test_split(
        X_all,
        y_all,
        sample_codes_all,
        test_size=0.20,
        random_state=2,
    )

    print(f"Internal training rows: {len(y_train)}")
    print(f"Internal validation rows: {len(y_val_int)}\n")

    # ---------------- LASSO ----------------
    pipe_lasso = fit_lasso(X_train, y_train)
    summarise_lasso(pipe_lasso, DEFAULT_FEATURE_COLS)  # coefficients only

    # Internal validation metrics & graph (held-out DB_2)
    y_val_int_pred, r2_int, rmse_int = report_held_out_metrics(
        pipe_lasso, X_val_int, y_val_int, "Model validation"
    )

    plot_predicted_vs_actual(
        y_val_int,
        y_val_int_pred,
        sample_codes=codes_val_int,
        title="Model validation (20% hold-out)",
    )

    # 2D surface + "any effect" boundary (?FMC = 0)
    plot_pdp_2d_d20_span(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
        title="Model Predicted FMC (?FMC = 0 boundary)",
        delta_threshold=0.0,   # diaphragm just starts/stops helping
    )

    # 2D surface + "? 3% reduction" boundary (?FMC = -3)
    plot_pdp_2d_d20_span(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
        title="Model Predicted FMC (? 3% diaphragm benefit)",
        delta_threshold=5.0,  # diaphragm gives 3% lower FMC
    )

    # Diaphragm effect plots for LASSO (linear model with nonlinear features)
    plot_pdp_d10_diaphragm(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
    )
    plot_delta_d10_diaphragm(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
    )
    # Model-based diaphragm effect vs Span (D80/D20)
    plot_delta_span_diaphragm(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
    )

    # Empirical diaphragm plots using raw DB_2 data
    plot_empirical_d10_diaphragm(
        df_train_merged,
        target_col=args.target_col,
        title="FMC vs D10 for diaphragm on/off",
    )

    plot_empirical_span_diaphragm(
        df_train_merged,
        target_col=args.target_col,
        title="FMC vs Span (D80/D20) for diaphragm on/off",
    )

    plot_empirical_delta_d10_diaphragm(
        df_train_merged,
        target_col=args.target_col,
    )

    plot_empirical_delta_span_diaphragm(
        df_train_merged,
        target_col=args.target_col,
    )

    # ---------- EXTERNAL VALIDATION DATA (DB) ----------
    df_db_val, _ = load_sheets(args.xlsx_path, args.sheet_db_val, args.sheet_psd)
    df_db_val_inc = apply_flag_filter(df_db_val)

    # External validation: use ALL DB samples except Si_BM and Si_Rep_new
    excluded_codes = {"Si_BM", "Si_Rep_new"}
    df_db_val_inc = df_db_val_inc[~df_db_val_inc[SAMPLE_CODE_COL].isin(excluded_codes)].copy()

    if df_db_val_inc.empty:
        error("No external validation rows left after filtering to specified sample codes in DB.")

    df_val_merged, X_val_ext, y_val_ext = merge_db_psd(
        df_db_val_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
    )

    print(f"External validation rows from sheet '{args.sheet_db_val}' (after filtering): {len(df_val_merged)}\n")

    # External validation metrics & graph (DB) for LASSO
    y_val_ext_pred, r2_ext, rmse_ext = report_held_out_metrics(
        pipe_lasso, X_val_ext, y_val_ext, "LASSO EXTERNAL validation (DB samples)"
    )

    plot_predicted_vs_actual(
        y_val_ext,
        y_val_ext_pred,
        sample_codes=df_val_merged[SAMPLE_CODE_COL].values,
        title="LASSO - External validation (DB samples)",
    )

    # --------- Random Forest / Gradient Boosting REMOVED FROM EXECUTION ---------
    # (functions remain defined above but are not called; only LASSO is used)


if __name__ == "__main__":
    main()
