
# Models/stepwise_psd_models.py
# -*- coding: utf-8 -*-
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ---------------------------
# Utilities
# ---------------------------

def _flag_include_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows where 'flag' contains 'include' (case-insensitive).
    If 'flag' not present, keep all rows (and warn).
    """
    if "flag" not in df.columns:
        warnings.warn("Column 'flag' not found in DB; using all rows.")
        return df.copy()
    mask = df["flag"].astype(str).str.lower().str.contains("include", na=False)
    return df.loc[mask].copy()


def _safe_ratio(a: pd.Series, b: pd.Series, name: str) -> pd.Series:
    """Compute a/b, guarding against zero/NaN denominators."""
    out = pd.Series(np.nan, index=a.index, name=name, dtype=float)
    valid = b.replace(0, np.nan).notna() & a.notna()
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def _build_psd_features(df_psd: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns at least: 'Sample Code' and some of D10,D20,D50,D80,D90.
    Creates ratio features too. Returns a tidy feature frame keyed by 'Sample Code'.
    """
    needed_any = ["D10", "D20", "D50", "D80", "D90"]
    missing = [c for c in needed_any if c not in df_psd.columns]
    if missing:
        warnings.warn(f"Missing PSD columns {missing}. Continue with what is available.")

    feats = df_psd[["Sample Code"] + [c for c in needed_any if c in df_psd.columns]].copy()

    # Ratios (compute only if both components exist)
    if {"D90", "D50"}.issubset(feats.columns):
        feats["D90_over_D50"] = _safe_ratio(feats["D90"], feats["D50"], "D90_over_D50")
    if {"D50", "D10"}.issubset(feats.columns):
        feats["D50_over_D10"] = _safe_ratio(feats["D50"], feats["D10"], "D50_over_D10")
    if {"D80", "D20"}.issubset(feats.columns):
        feats["D80_over_D20"] = _safe_ratio(feats["D80"], feats["D20"], "D80_over_D20")

    return feats


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Compute a quick VIF table (informational)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_ = X.dropna().copy()
    X_ = sm.add_constant(X_, has_constant='add')
    names = ["const"] + list(X_.columns.drop("const"))
    vifs = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
    return pd.DataFrame({"feature": names, "VIF": vifs})


# ---------------------------
# Stepwise (AIC) selection
# ---------------------------

def stepwise_aic(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial: Op
