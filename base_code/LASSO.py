#!/usr/bin/env python3
"""
LASSO regression on PSD indices with DB_2/PSD merge,
flag filtering ("include"), and robust error checking.
Now includes operational variables:
  - A_Flow  (airflow, from DB_2)
  - Diaphragm_on (binary, derived from Test_procedure: 1=STD, 0=No_pres/other)
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_DB_SHEET = "DB_2"
DEFAULT_PSD_SHEET = "PSD"

DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"

DEFAULT_TARGET_COL = "Mc_%"

# PSD + operational features
DEFAULT_FEATURE_COLS = [
    # PSD indices
    "D10",
    "D20",
    "D50",
    "D80",
    "D90",
    "D90/D50",
    "D50/D10",
    "D80/D20",
    # operational variables from DB_2
    "A_Flow",
    "Diaphragm_on",
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
    """Load DB_2 and PSD sheets with error checking."""
    try:
        xls = pd.ExcelFile(xlsx_path)
    except FileNotFoundError:
        error(f"Excel file not found: {xlsx_path}")
    except Exception as e:
        error(f"Failed to open Excel file '{xlsx_path}': {e}")

    if sheet_db not in xls.sheet_names:
        error(f"DB_2 sheet '{sheet_db}' not found. Sheets: {xls.sheet_names}")

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
        error(f"Flag column '{FLAG_COL}' missing in DB_2 sheet.")

    mask = df_db[FLAG_COL].astype(str).str.contains("include", case=False, na=False)
    df = df_db.loc[mask].copy()

    if df.empty:
        error("No rows remain after flag filter ('include').")

    return df


def merge_db_psd(df_db, df_psd, features, target_col):
    """
    Merge DB_2 and PSD on sample code, recompute ratio features,
    engineer operational variables, drop rows with missing data,
    and return df, X, y.
    """

    # Base PSD D-values required from PSD sheet
    base_needed = ["D10", "D20", "D50", "D80", "D90"]

    # Required DB_2 columns (for target, flags and operational vars)
    db_required = [SAMPLE_CODE_COL, target_col, FLAG_COL, TEST_PROC_COL, AFLOW_COL]

    check_required_columns(df_db, db_required, "DB_2")
    check_required_columns(df_psd, [SAMPLE_CODE_COL] + base_needed, "PSD")

    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner")

    if df.empty:
        error("Merge between DB_2 and PSD sheets is empty. Check sample codes.")

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

    # Select only needed columns
    cols_to_keep = [SAMPLE_CODE_COL, target_col] + features
    df = df[cols_to_keep].copy()

    # Identify rows with missing data
    missing_mask = df[[target_col] + features].isna().any(axis=1)
    dropped = df.loc[missing_mask]

    if not dropped.empty:
        print("\nWarning: dropped rows due to NaNs:\n")
        print(dropped.to_string(index=False))
        print("\n--- End dropped row report ---\n")

    df = df.loc[~missing_mask].copy()

    if df.empty:
        error("All rows dropped due to missing values in target/features.")

    X = df[features].to_numpy(float)
    y = df[target_col].to_numpy(float)

    return df, X, y


# ---------------------------------------------------------------------
# LASSO Fit & Summary
# ---------------------------------------------------------------------

def fit_lasso(X, y) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=0, n_alphas=100, max_iter=10000)),
    ])
    pipe.fit(X, y)
    return pipe


def summarise_lasso(pipe: Pipeline, features: list[str], X: np.ndarray, y: np.ndarray):
    """Print summary of fitted LASSO model."""
    lasso: LassoCV = pipe.named_steps["lasso"]
    y_pred = pipe.predict(X)

    print("\n================ LASSO Regression Summary ================")
    print(f"Chosen alpha (lambda): {lasso.alpha_:.6g}")
    print(f"R^2 (on full dataset): {r2_score(y, y_pred):.4f}")
    print(f"RMSE:                 {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print("==========================================================\n")

    coefs = lasso.coef_
    coef_table = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("Coefficients (sorted by |coefficient|):")
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


# ---------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 5))

    # scatter
    ax.scatter(y_true, y_pred, alpha=0.8)

    # 1:1 line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="1:1 line")

    # best-fit line
    m, c = np.polyfit(y_true, y_pred, 1)
    ax.plot([min_val, max_val],
            [m * min_val + c, m * max_val + c],
            linestyle="-", label=f"Best fit (slope={m:.2f})")

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax.set_xlabel("Measured Mc")
    ax.set_ylabel("Predicted Mc")
    ax.set_title("Predicted vs Actual Moisture")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Text box with metrics
    txt = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$"
    ax.text(0.05, 0.95, txt,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pdp_1d(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    feature_name: str,
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    Simple 1D partial dependence-style plot for a single feature.
    Other features are held at their mean values.
    """
    if feature_name not in features:
        raise ValueError(f"{feature_name} is not in feature list {features}")

    X = df[features].to_numpy(dtype=float)
    base = X.mean(axis=0)
    idx = features.index(feature_name)

    x_vals = np.linspace(df[feature_name].min(), df[feature_name].max(), n_points)

    X_grid = np.tile(base, (n_points, 1))
    X_grid[:, idx] = x_vals

    y_pred = pipe.predict(X_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, y_pred, linewidth=2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted Mc")
    ax.set_title(f"Partial dependence of Mc on {feature_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pdp_2d_d20_span(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    feature_x: str = "D20",
    feature_y: str = "D80/D20",
    nx: int = 40,
    ny: int = 40,
    save_path: Optional[str] = None,
):
    """
    2D partial dependence-style plot:
      x-axis: D20
      y-axis: D80/D20
      colour: predicted Mc
    Other features are held at their mean values.
    """
    for f in (feature_x, feature_y):
        if f not in features:
            raise ValueError(f"{f} is not in feature list {features}")

    X = df[features].to_numpy(dtype=float)
    base = X.mean(axis=0)
    ix = features.index(feature_x)
    iy = features.index(feature_y)

    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), nx)
    y_vals = np.linspace(df[feature_y].min(), df[feature_y].max(), ny)

    XX, YY = np.meshgrid(x_vals, y_vals)
    grid = np.tile(base, (nx * ny, 1))
    grid[:, ix] = XX.ravel()
    grid[:, iy] = YY.ravel()

    Z = pipe.predict(grid).reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(XX, YY, Z, levels=15)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Mc")

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Predicted moisture as a function of D20 and D80/D20")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LASSO regression on PSD indices with DB_2/PSD merge and flag filtering.",
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
    parser.add_argument("--sheet_db", default=DEFAULT_DB_SHEET)
    parser.add_argument("--sheet_psd", default=DEFAULT_PSD_SHEET)
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL)

    args = parser.parse_args()

    print(f"\nUsing Excel file:\n  {args.xlsx_path}\n")

    # Load sheets
    df_db, df_psd = load_sheets(args.xlsx_path, args.sheet_db, args.sheet_psd)

    # Apply flag filter ("include")
    df_db_inc = apply_flag_filter(df_db)

    # Merge and build X, y
    df_merged, X, y = merge_db_psd(
        df_db_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
    )

    print(f"Using {len(df_merged)} rows after merge, NaN removal, and flag filtering.")
    print(f"Features: {DEFAULT_FEATURE_COLS}")
    print(f"Target:   {args.target_col}\n")

    # Fit LASSO
    pipe = fit_lasso(X, y)

    # Summarise
    summarise_lasso(pipe, DEFAULT_FEATURE_COLS, X, y)

    # Diagnostic plots
    y_pred = pipe.predict(X)
    plot_predicted_vs_actual(y, y_pred)

    df_features = df_merged[DEFAULT_FEATURE_COLS].copy()
    plot_pdp_1d(pipe, df_features, DEFAULT_FEATURE_COLS, "D20")
    plot_pdp_1d(pipe, df_features, DEFAULT_FEATURE_COLS, "D80/D20")
    plot_pdp_2d_d20_span(pipe, df_features, DEFAULT_FEATURE_COLS)


if __name__ == "__main__":
    main()
