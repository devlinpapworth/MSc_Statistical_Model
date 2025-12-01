#!/usr/bin/env python3
"""
Final LASSO model on PSD indices using ALL data from DB_2 and DB,
with a 15% hold-out set for validation.

- Loads DB_2 and DB, applies flag filter ("include")
- Merges each with PSD
- Concatenates both datasets into a single dataframe
- Splits combined data: 85% train, 15% validation
- Fits a LASSO model on the training set
- Reports TRAINING R^2 / RMSE (on 85%) and VALIDATION R^2 / RMSE (on 15%)
- Produces a Predicted vs Actual plot for the held-out 15%,
  colour-coded by Sample Code.

Optional (commented at bottom):
- Random Forest and Gradient Boosting fits on the same training data
  and validation on the 15% held-out.
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

DB2_SHEET = "DB_2"
DB_SHEET = "DB"
PSD_SHEET = "PSD"

DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
DEFAULT_TARGET_COL = "Mc_%"

# PSD + operational + interaction features
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
    # operational variables from DB_2 / DB
    "A_Flow",
    "Diaphragm_on",
    # interactions (PSD x Diaphragm)
    "D20_Dia",   # D20 * Diaphragm_on
    "Span_Dia",  # (D80/D20) * Diaphragm_on
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


def merge_db_psd(df_db, df_psd, features, target_col, db_name="DB"):
    """
    Merge DB_x and PSD on sample code, recompute ratio features,
    engineer operational & interaction variables, drop rows with
    missing data, and return df, X, y.
    """

    # Base PSD D-values required from PSD sheet
    base_needed = ["D10", "D20", "D50", "D80", "D90"]

    # Required DB columns (for target, flags and operational vars)
    db_required = [SAMPLE_CODE_COL, target_col, FLAG_COL, TEST_PROC_COL, AFLOW_COL]

    check_required_columns(df_db, db_required, db_name)
    check_required_columns(df_psd, [SAMPLE_CODE_COL] + base_needed, "PSD")

    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner")

    if df.empty:
        error(f"Merge between {db_name} and PSD sheets is empty. Check sample codes.")

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

    # Interaction terms
    df["D20_Dia"] = df["D20"] * df["Diaphragm_on"]
    df["Span_Dia"] = df["D80/D20"] * df["Diaphragm_on"]

    # Select only needed columns
    cols_to_keep = [SAMPLE_CODE_COL, target_col] + features
    df = df[cols_to_keep].copy()

    # Identify rows with missing data
    missing_mask = df[[target_col] + features].isna().any(axis=1)
    dropped = df.loc[missing_mask]

    if not dropped.empty:
        print(f"\nWarning: dropped rows due to NaNs in {db_name}/PSD merge:\n")
        print(dropped.to_string(index=False))
        print("\n--- End dropped row report ---\n")

    df = df.loc[~missing_mask].copy()

    if df.empty:
        error(f"All rows in {db_name} dropped due to missing values in target/features.")

    X = df[features].to_numpy(float)
    y = df[target_col].to_numpy(float)

    return df, X, y


# ---------------------------------------------------------------------
# LASSO Fit & Summary (training metrics)
# ---------------------------------------------------------------------

def fit_lasso(X, y) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=0, n_alphas=100, max_iter=10000)),
    ])
    pipe.fit(X, y)
    return pipe


def summarise_lasso_with_fit(pipe: Pipeline, features: list[str], X: np.ndarray, y: np.ndarray):
    """
    Print summary of fitted LASSO model on the TRAINING set:
    - alpha
    - training R^2 / RMSE
    - coefficients and which features were kept/dropped
    """
    lasso = pipe.named_steps["lasso"]
    y_pred = pipe.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print("\n================ LASSO Regression Summary (TRAINING SET) ================")
    print(f"Chosen alpha (lambda): {lasso.alpha_:.6g}")
    print(f"Training R^2 (85% of data):  {r2:.4f}")
    print(f"Training RMSE (85% of data): {rmse:.4f}\n")

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
    print("======================================================================\n")


# ---------------------------------------------------------------------
# Optional tree-based models
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


def report_validation_metrics(model, X_val, y_val, label: str):
    """Print R2 and RMSE for any model on validation data."""
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{label} VALIDATION R^2 (15% hold-out):  {r2:.4f}")
    print(f"{label} VALIDATION RMSE (15% hold-out): {rmse:.4f}\n")
    return y_pred, r2, rmse


# ---------------------------------------------------------------------
# Plotting
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

    y_true_pct = 100.0 * y_true
    y_pred_pct = 100.0 * y_pred

    min_val = 0.0
    max_val = max(np.max(y_true_pct), np.max(y_pred_pct)) * 1.05

    if sample_codes is not None:
        sample_codes = np.asarray(sample_codes)
        unique_codes = np.unique(sample_codes)

        cmap = plt.get_cmap("tab20")
        n_colors = cmap.N

        for i, code in enumerate(unique_codes):
            mask = (sample_codes == code)
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

    # best-fit line
    m, c = np.polyfit(y_true_pct, y_pred_pct, 1)
    ax.plot(
        [min_val, max_val],
        [m * min_val + c, m * max_val + c],
        linestyle="-",
        label=f"Best fit (slope={m:.2f})",
    )

    r2 = r2_score(y_true, y_pred)
    rmse_pct = 100.0 * np.sqrt(mean_squared_error(y_true, y_pred))

    ax.set_xlabel("Measured FMC (%)")
    ax.set_ylabel("Predicted FMC (%)")
    ax.set_title(title)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

    if sample_codes is not None:
        ax.legend(
            title="Sample Code",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=7,
        )
    else:
        ax.legend()

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Final LASSO model on PSD indices using ALL data from DB_2 and DB with 15% hold-out validation.",
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
    parser.add_argument("--sheet_db2", default=DB2_SHEET)
    parser.add_argument("--sheet_db", default=DB_SHEET)
    parser.add_argument("--sheet_psd", default=PSD_SHEET)
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL)

    args = parser.parse_args()

    print(f"\nUsing Excel file:\n  {args.xlsx_path}\n")

    # ---------- Load and merge DB_2 ----------
    df_db2_raw, df_psd = load_sheets(args.xlsx_path, args.sheet_db2, args.sheet_psd)
    df_db2_inc = apply_flag_filter(df_db2_raw)

    df_db2_merged, X_db2, y_db2 = merge_db_psd(
        df_db2_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
        db_name=args.sheet_db2,
    )

    print(f"Rows from '{args.sheet_db2}' after merge and filtering: {len(df_db2_merged)}")

    # ---------- Load and merge DB ----------
    df_db_raw, _ = load_sheets(args.xlsx_path, args.sheet_db, args.sheet_psd)
    df_db_inc = apply_flag_filter(df_db_raw)

    df_db_merged, X_db, y_db = merge_db_psd(
        df_db_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
        db_name=args.sheet_db,
    )

    print(f"Rows from '{args.sheet_db}' after merge and filtering:  {len(df_db_merged)}")

    # ---------- Combine DB_2 + DB ----------
    df_all = pd.concat([df_db2_merged, df_db_merged], ignore_index=True)
    X_all = df_all[DEFAULT_FEATURE_COLS].to_numpy(float)
    y_all = df_all[args.target_col].to_numpy(float)
    sample_codes_all = df_all[SAMPLE_CODE_COL].to_numpy()

    print(f"\nTotal rows before split (DB_2 + DB): {len(df_all)}")
    print(f"Features: {DEFAULT_FEATURE_COLS}")
    print(f"Target:   {args.target_col}\n")

    # ---------- 85% training / 15% validation split ----------
    X_train, X_val, y_train, y_val, codes_train, codes_val = train_test_split(
        X_all,
        y_all,
        sample_codes_all,
        test_size=0.15,
        random_state=0,
    )

    print(f"Training rows (85%):   {len(y_train)}")
    print(f"Validation rows (15%): {len(y_val)}\n")

    # ---------- LASSO on training data ----------
    pipe_lasso = fit_lasso(X_train, y_train)
    summarise_lasso_with_fit(pipe_lasso, DEFAULT_FEATURE_COLS, X_train, y_train)

    # ---------- Validation metrics & plot ----------
    y_val_pred, r2_val, rmse_val = report_validation_metrics(
        pipe_lasso, X_val, y_val, "LASSO"
    )

    plot_predicted_vs_actual(
        y_val,
        y_val_pred,
        sample_codes=codes_val,
        title="LASSO - Validation on 15% held-out (DB_2 + DB)",
    )

    # ---------- Optional: non-linear models on same split ----------
    # print("\n--- Random Forest on same 85/15 split ---")
    # rf = fit_random_forest(X_train, y_train)
    # _, _, _ = report_validation_metrics(rf, X_val, y_val, "Random Forest")

    # print("\n--- Gradient Boosting on same 85/15 split ---")
    # gb = fit_gradient_boosting(X_train, y_train)
    # _, _, _ = report_validation_metrics(gb, X_val, y_val, "Gradient Boosting")


if __name__ == "__main__":
    main()
