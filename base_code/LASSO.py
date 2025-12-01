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
  - D10_Dia  = D20 * Diaphragm_on
  - Span_Dia = (D80/D20) * Diaphragm_on

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
    "D10_Dia",   # D20 * Diaphragm_on
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

    # Interaction terms
    df["D10_Dia"] = df["D10"] * df["Diaphragm_on"]
    df["Span_Dia"] = df["D80/D20"] * df["Diaphragm_on"]

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
# Tree-based models (RF & GB)
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
    feature_x: str = "D20",
    feature_y: str = "D80/D20",
    nx: int = 40,
    ny: int = 40,
    title: str = "Predicted FMC as a function of D20 and D80/D20",
    save_path: Optional[str] = None,
):
    """
    2D partial dependence-style plot:
      x-axis: D20
      y-axis: D80/D20
      colour: predicted FMC (%).
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

    Z = 100 * pipe.predict(grid).reshape(ny, nx)  # convert to %

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(XX, YY, Z, levels=15)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted FMC (%)")

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pdp_d20_diaphragm(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    1D PDP for D20 with two curves:
      - Diaphragm_on = 0
      - Diaphragm_on = 1

    Interaction features (D10_Dia, Span_Dia) are recomputed so they stay
    consistent with the chosen Diaphragm_on.
    """
    required = {"D20", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20"}
    if not required.issubset(features):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df["D20"].min(), df["D20"].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D20"] = x_vals
        grid_df["Diaphragm_on"] = dia
        # recompute interactions
        grid_df["D10_Dia"] = grid_df["D20"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]

        y_pred = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, preds[0], label="No diaphragm", linewidth=2)
    ax.plot(x_vals, preds[1], label="Diaphragm on", linewidth=2)

    ax.set_xlabel("D20 (\u00B5m)")
    ax.set_ylabel("Predicted FMC (%)")
    ax.set_title("Effect of diaphragm across D20 (LASSO)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_delta_d20_diaphragm(
    pipe: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot ?FMC(D20) = FMC(diaphragm on) - FMC(no diaphragm) vs D20
    for the LASSO model. This will always be a straight line
    because LASSO is linear (even with interactions).
    """
    required = {"D20", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20"}
    if not required.issubset(features):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df["D20"].min(), df["D20"].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D20"] = x_vals
        grid_df["Diaphragm_on"] = dia
        grid_df["D10_Dia"] = grid_df["D20"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]

        y_pred = 100 * pipe.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    delta = preds[1] - preds[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, delta, linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("D20 (\u00B5m)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title("Additional effect of diaphragm vs D20 (LASSO)")
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

        ?FMC(D20) = FMC_dia(D20) - FMC_no_dia(D20)

    If this curve is curved (not a straight line), the diaphragm
    effect is genuinely nonlinear in D20.
    """
    required = {"D20", "Diaphragm_on", "D10_Dia", "Span_Dia", "D80/D20"}
    if not required.issubset(features):
        raise ValueError(f"Features must include {required} for this plot.")

    base_row = df[features].mean()
    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), n_points)

    preds = {}
    for dia in [0, 1]:
        grid_df = pd.DataFrame([base_row] * n_points)
        grid_df["D20"] = x_vals
        grid_df["Diaphragm_on"] = dia
        grid_df["D10_Dia"] = grid_df["D20"] * grid_df["Diaphragm_on"]
        grid_df["Span_Dia"] = grid_df["D80/D20"] * grid_df["Diaphragm_on"]

        y_pred = 100 * model.predict(grid_df[features].to_numpy(dtype=float))
        preds[dia] = y_pred

    delta = preds[1] - preds[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, delta, linewidth=2)

    ax.axhline(0.0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("D20 (\u00B5m)")
    ax.set_ylabel("\u0394FMC (Dia - No Dia) (%)")
    ax.set_title(f"Nonlinear diaphragm effect vs D20 ({model_name})")
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
    df_db_train_inc = apply_flag_filter(df_db_train_full)

    df_train_merged, X_all, y_all = merge_db_psd(
        df_db_train_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
    )

    print(f"Total usable rows from sheet '{args.sheet_db_train}' after merge & filtering: {len(df_train_merged)}")
    print(f"Features: {DEFAULT_FEATURE_COLS}")
    print(f"Target:   {args.target_col}\n")

    sample_codes_all = df_train_merged[SAMPLE_CODE_COL].to_numpy()

    # ---------- INTERNAL HOLD-OUT SPLIT (DB_2) ----------
    X_train, X_val_int, y_train, y_val_int, codes_train, codes_val_int = train_test_split(
        X_all,
        y_all,
        sample_codes_all,
        test_size=0.25,
        random_state=2,
    )

    print(f"Internal training rows: {len(y_train)}")
    print(f"Internal validation rows: {len(y_val_int)}\n")

    # ---------------- LASSO ----------------
    pipe_lasso = fit_lasso(X_train, y_train)
    summarise_lasso(pipe_lasso, DEFAULT_FEATURE_COLS)  # coefficients only

    # Internal validation metrics & graph (held-out DB_2)
    y_val_int_pred, r2_int, rmse_int = report_held_out_metrics(
        pipe_lasso, X_val_int, y_val_int, "LASSO INTERNAL validation"
    )

    plot_predicted_vs_actual(
        y_val_int,
        y_val_int_pred,
        sample_codes=codes_val_int,
        title="LASSO - Internal validation (DB_2 hold-out)",
    )

    # 2D contour PDP using ALL DB_2 rows (for smoother surface)
    df_features_train = df_train_merged[DEFAULT_FEATURE_COLS].copy()
    plot_pdp_2d_d20_span(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
        title="LASSO - Predicted FMC as a function of D20 and D80/D20",
    )

    # Diaphragm effect plots for LASSO (linear)
    plot_pdp_d20_diaphragm(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
    )
    plot_delta_d20_diaphragm(
        pipe_lasso,
        df_features_train,
        DEFAULT_FEATURE_COLS,
    )

    # ---------- EXTERNAL VALIDATION DATA (DB) ----------
    df_db_val, _ = load_sheets(args.xlsx_path, args.sheet_db_val, args.sheet_psd)
    df_db_val_inc = apply_flag_filter(df_db_val)

    # Restrict external validation to specific sample codes
    allowed_codes = {"Si_M", "Si_F", "Si_Rep", "Si_Rep_new", "Si_BM"}
    df_db_val_inc = df_db_val_inc[df_db_val_inc[SAMPLE_CODE_COL].isin(allowed_codes)].copy()

    if df_db_val_inc.empty:
        error("No external validation rows left after filtering to specified sample codes in DB.")

    df_val_merged, X_val_ext, y_val_ext = merge_db_psd(
        df_db_val_inc,
        df_psd,
        features=DEFAULT_FEATURE_COLS,
        target_col=args.target_col,
    )

    print(f"External validation rows from sheet '{args.sheet_db_val}' (Si_M/Si_F/Si_Rep/Si_Rep_new/Si_BM): {len(df_val_merged)}\n")

    # External validation metrics & graph (DB) for LASSO
    y_val_ext_pred, r2_ext, rmse_ext = report_held_out_metrics(
        pipe_lasso, X_val_ext, y_val_ext, "LASSO EXTERNAL validation (DB samples)"
    )

    plot_predicted_vs_actual(
        y_val_ext,
        y_val_ext_pred,
        sample_codes=df_val_merged[SAMPLE_CODE_COL].values,
        title="LASSO - External validation (DB: Si_M, Si_F, Si_Rep, Si_Rep_new, Si_BM)",
    )

    # ---------------- Random Forest (non-linear) ----------------
    rf = fit_random_forest(X_train, y_train)
    print("\n--- Random Forest held-out metrics ---")
    _, r2_rf_int, rmse_rf_int = report_held_out_metrics(
        rf, X_val_int, y_val_int, "RF INTERNAL validation"
    )
    _, r2_rf_ext, rmse_rf_ext = report_held_out_metrics(
        rf, X_val_ext, y_val_ext, "RF EXTERNAL validation (DB samples)"
    )

    # Non-linear diaphragm effect vs D20 (Random Forest)
    plot_delta_dia_tree(
        rf,
        df_features_train,
        DEFAULT_FEATURE_COLS,
        model_name="Random Forest",
    )

    # ---------------- Gradient Boosting (non-linear) ----------------
    gb = fit_gradient_boosting(X_train, y_train)
    print("\n--- Gradient Boosting held-out metrics ---")
    _, r2_gb_int, rmse_gb_int = report_held_out_metrics(
        gb, X_val_int, y_val_int, "GB INTERNAL validation"
    )
    _, r2_gb_ext, rmse_gb_ext = report_held_out_metrics(
        gb, X_val_ext, y_val_ext, "GB EXTERNAL validation (DB samples)"
    )

    # Non-linear diaphragm effect vs D20 (Gradient Boosting)
    plot_delta_dia_tree(
        gb,
        df_features_train,
        DEFAULT_FEATURE_COLS,
        model_name="Gradient Boosting",
    )


if __name__ == "__main__":
    main()
