#!/usr/bin/env python3
"""
LASSO regression on PSD indices with DB_2/PSD merge,
flag filtering ("include"), and robust error checking.
"""

import argparse
import sys
import textwrap

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------
# Configuration (you can tweak these to match your workbook)
# ---------------------------------------------------------------------

# Default sheets
DEFAULT_DB_SHEET = "DB_2"
DEFAULT_PSD_SHEET = "PSD"

# Automatic default Excel file path
DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"

# Default target column in DB_2 sheet
DEFAULT_TARGET_COL = "Mc_%"

# PSD feature columns (you can edit this list)
DEFAULT_FEATURE_COLS = [
    "D10",
    "D20",
    "D50",
    "D80",
    "D90",
    "D90/D50",
    "D50/D10",
    "D80/D20"
]

# Column name for sample code (must be present in both DB_2 and PSD sheets)
SAMPLE_CODE_COL = "Sample Code"

# Column name for flags in DB_2
FLAG_COL = "flag"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def error(msg: str):
    """Print an error and exit with non-zero status."""
    sys.stderr.write(f"\nERROR: {msg}\n\n")
    sys.exit(1)


def load_sheets(xlsx_path: str, sheet_db: str, sheet_psd: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DB_2 and PSD sheets with error checking."""
    try:
        xls = pd.ExcelFile(xlsx_path)
    except FileNotFoundError:
        error(f"Excel file not found: {xlsx_path}")
    except Exception as e:
        error(f"Failed to open Excel file '{xlsx_path}': {e}")

    if sheet_db not in xls.sheet_names:
        error(
            f"DB_2 sheet '{sheet_db}' not found in workbook.\n"
            f"Available sheets: {xls.sheet_names}"
        )
    if sheet_psd not in xls.sheet_names:
        error(
            f"PSD sheet '{sheet_psd}' not found in workbook.\n"
            f"Available sheets: {xls.sheet_names}"
        )

    try:
        df_db = pd.read_excel(xls, sheet_name=sheet_db)
        df_psd = pd.read_excel(xls, sheet_name=sheet_psd)
    except Exception as e:
        error(f"Failed to read sheets: {e}")

    return df_db, df_psd


def check_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str):
    """Ensure required columns exist in a DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        error(
            f"Missing required columns in '{df_name}' sheet: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def apply_flag_filter(df_db: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where flag contains 'include' (case-insensitive)."""
    if FLAG_COL not in df_db.columns:
        error(
            f"Flag column '{FLAG_COL}' not found in DB_2 sheet.\n"
            f"Columns available: {list(df_db.columns)}"
        )

    mask = df_db[FLAG_COL].astype(str).str.contains("include", case=False, na=False)
    df_filtered = df_db.loc[mask].copy()

    if df_filtered.empty:
        error(
            "No rows remaining after flag filter.\n"
            "Check that your 'flag' column contains 'include' for rows you want."
        )

    return df_filtered


def merge_db_psd(
    df_db: pd.DataFrame,
    df_psd: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Merge DB_2 and PSD on SAMPLE_CODE_COL, ensure required cols,
    recompute ratio features (D80/D20, D90/D50, D50/D10) from base D-values,
    and return X (features) and y (target) arrays after dropping NAs.
    Includes detailed debug printout showing which rows were dropped.
    """

    # --- Check columns exist ---
    base_needed = ["D10", "D20", "D50", "D80", "D90"]
    check_required_columns(df_db, [SAMPLE_CODE_COL, target_col, FLAG_COL], "DB_2")
    check_required_columns(df_psd, [SAMPLE_CODE_COL] + base_needed, "PSD")

    # --- Merge DB_2 + PSD ---
    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner", suffixes=("_db", "_psd"))
    if df.empty:
        error(
            "Merge between DB_2 and PSD is empty.\n"
            f"Check that '{SAMPLE_CODE_COL}' matches between sheets."
        )

    # --- Ensure base D-columns are float ---
    for col in base_needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Recompute ratio columns in Python (override any Excel values) ---
    if "D80/D20" in features:
        df["D80/D20"] = df["D80"] / df["D20"]
    if "D90/D50" in features:
        df["D90/D50"] = df["D90"] / df["D50"]
    if "D50/D10" in features:
        df["D50/D10"] = df["D50"] / df["D10"]

    # --- Select only relevant columns ---
    cols_to_keep = [SAMPLE_CODE_COL, target_col] + features
    df = df[cols_to_keep].copy()

    # --- Identify rows with missing values ---
    before = len(df)
    missing_mask = df[[target_col] + features].isna().any(axis=1)
    after = (~missing_mask).sum()

    # --- Print dropped rows for debugging ---
    if after < before:
        print(f"\nWarning: dropped {before - after} rows due to NaNs.\n")
        print("Rows that were dropped because of missing values:\n")
        print(
            df.loc[missing_mask, [SAMPLE_CODE_COL, target_col] + features]
            .to_string(index=False)
        )
        print("\n--- End of dropped row report ---\n")

    # --- Keep only complete rows ---
    df = df.loc[~missing_mask].copy()

    if df.empty:
        error("All rows dropped due to missing values in target/features.")

    # --- Build matrices for regression ---
    X = df[features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    return df, X, y




def fit_lasso(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Fit LASSO with cross-validation in a pipeline (StandardScaler + LassoCV).
    Returns the fitted Pipeline.
    """
    lasso = LassoCV(
        cv=5,
        random_state=0,
        n_alphas=100,
        max_iter=10000,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lasso", lasso),
        ]
    )

    pipe.fit(X, y)
    return pipe


def summarise_lasso(pipe: Pipeline, features: list[str], X: np.ndarray, y: np.ndarray):
    """Print summary of fitted LASSO model."""
    lasso: LassoCV = pipe.named_steps["lasso"]

    y_pred = pipe.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print("\n================ LASSO Regression Summary ================")
    print(f"Chosen alpha (lambda): {lasso.alpha_:.6g}")
    print(f"R^2 (on full dataset): {r2:.4f}")
    print(f"RMSE:                 {rmse:.4f}")
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

    non_zero = coef_table[np.isclose(coef_table["coefficient"], 0.0) == False]
    zero = coef_table[np.isclose(coef_table["coefficient"], 0.0)]

    print("\nNon-zero coefficients (selected features):")
    if non_zero.empty:
        print("  None (all coefficients shrank to zero!)")
    else:
        for _, row in non_zero.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.6g}")

    print("\nZero coefficients (dropped features):")
    if zero.empty:
        print("  None")
    else:
        for _, row in zero.iterrows():
            print(f"  {row['feature']}")


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
    parser.add_argument("--sheet_db", default=DEFAULT_DB_SHEET, help=f"Name of DB_2 sheet (default: {DEFAULT_DB_SHEET})")
    parser.add_argument("--sheet_psd", default=DEFAULT_PSD_SHEET, help=f"Name of PSD sheet (default: {DEFAULT_PSD_SHEET})")
    parser.add_argument("--target_col", default=DEFAULT_TARGET_COL, help=f"Target column in DB_2 sheet (default: {DEFAULT_TARGET_COL})")

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


if __name__ == "__main__":
    main()
