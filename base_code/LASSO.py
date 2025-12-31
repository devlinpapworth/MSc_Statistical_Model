"""
regression_db_only.py
---------------------
Runs a regression (LASSO) on 100% of the DB sheet (no train/val split, no external validation).

- Reads FP_db_all.xlsx
- Merges DB with PSD sheet to bring in PSD indices
- Fits LassoCV on ALL rows (cross-validation is only for alpha selection; model is fit on full data)
- Prints coefficients + TRAINING-ONLY metrics (no validation)

Edit FILEPATH / SHEET NAMES / TARGET / FEATURE SET as needed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error


# -----------------------------
# USER SETTINGS
# -----------------------------
FILEPATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
DB_SHEET = "DB"
PSD_SHEET = "PSD"

KEY = "Sample Code"

# Target in DB (you said FMC comes in as decimal sometimes; if yours is already %, set FMC_MULT = 1.0)
TARGET = "Mc_%"
FMC_MULT = 1.0   # set to 100.0 if Mc_% is a decimal fraction and you want %

# PSD columns (in PSD sheet) you want included (ONLY these)
PSD_COLS = [
    "D10", "D20", "D50", "D80", "D90",
    "D90/D50", "D50/D10", "D80/D20"
]

# LASSO settings
RANDOM_STATE = 2
N_FOLDS = 5


# -----------------------------
# Helpers
# -----------------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_and_merge() -> pd.DataFrame:
    db = pd.read_excel(FILEPATH, sheet_name=DB_SHEET)
    psd = pd.read_excel(FILEPATH, sheet_name=PSD_SHEET)

    db.columns = [c.strip() for c in db.columns]
    psd.columns = [c.strip() for c in psd.columns]

    # numeric target
    if TARGET in db.columns:
        db[TARGET] = to_num(db[TARGET]) * FMC_MULT

    # ensure PSD cols numeric if present
    psd_keep = [KEY]
    for c in PSD_COLS:
        if c in psd.columns:
            psd[c] = to_num(psd[c])
            psd_keep.append(c)

    # merge
    df = db.merge(psd[psd_keep], on=KEY, how="left")

    return df


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X and y (no validation split).

    PSD features included (ONLY):
      D10, D20, D50, D80, D90, D90/D50, D50/D10, D80/D20
    """
    required = [TARGET] + PSD_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d = d.dropna(subset=required)

    # target
    y = to_num(d[TARGET])

    # features (ONLY the PSD cols)
    X = pd.DataFrame(index=d.index)
    for c in PSD_COLS:
        X[c] = to_num(d[c])

    # final drop of any weird NaNs
    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep].copy()
    y = y.loc[keep].copy()

    return X, y


def fit_lasso_full(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=N_FOLDS, random_state=RANDOM_STATE, max_iter=200000)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def print_coefficients(pipe: Pipeline, feature_names: list[str]) -> None:
    lasso = pipe.named_steps["lasso"]
    coefs = pd.Series(lasso.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)

    print("\n===== LASSO (fit on 100% of DB) =====")
    print(f"Chosen alpha: {lasso.alpha_:.6g}\n")
    print("Coefficients (sorted by |coef|):")
    print(coefs.to_string())

    nonzero = coefs[coefs != 0.0]
    print("\nNon-zero terms:")
    if len(nonzero):
        print(nonzero.to_string())
    else:
        print("(none)")

    print("\nDropped terms (coef = 0):")
    dropped = coefs[coefs == 0.0]
    if len(dropped):
        print(dropped.to_string())
    else:
        print("(none)")


def print_training_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    yhat = pipe.predict(X)
    r2 = r2_score(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    print("\n===== TRAINING-ONLY metrics (no validation) =====")
    print(f"n = {len(y)}")
    print(f"R^2 (train): {r2:.4f}")
    print(f"RMSE (train): {rmse:.4f} (same units as y)")


def main() -> None:
    df = load_and_merge()
    print("Loaded rows:", len(df))

    X, y = build_design_matrix(df)
    print("Rows used after dropping NaNs:", len(y))
    print("Feature columns:", list(X.columns))

    pipe = fit_lasso_full(X, y)

    print_coefficients(pipe, list(X.columns))
    print_training_metrics(pipe, X, y)


if __name__ == "__main__":
    main()
