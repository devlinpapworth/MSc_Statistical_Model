# Models/stepwise_psd_models_nosplit.py
# -*- coding: utf-8 -*-

import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Utilities
# ---------------------------

def _flag_include_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows where 'flag' contains 'include' (case-insensitive).
    If 'flag' not present, keep all rows (and warn).
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    flag_col = next((c for c in df.columns if c.lower() == "flag"), None)
    if flag_col is None:
        warnings.warn("Column 'flag' not found in ign; using all rows.")
        return df

    mask = df[flag_col].astype(str).str.lower().str.contains("include", na=False)
    return df.loc[mask].copy()


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=a.index, dtype=float)
    valid = a.notna() & b.notna() & (b != 0)
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def _build_psd_features(df_psd: pd.DataFrame) -> pd.DataFrame:
    df = df_psd.copy()
    df.columns = df.columns.astype(str).str.strip()

    needed = ["D10", "D20", "D50", "D80", "D90"]
    keep = [c for c in needed if c in df.columns]

    if not keep:
        raise ValueError("No PSD size columns (D10.D90) found in PSD sheet.")

    feats = df[["Sample Code"] + keep].copy()

    if {"D90", "D50"}.issubset(feats.columns):
        feats["D90_over_D50"] = _safe_ratio(feats["D90"], feats["D50"])
    if {"D50", "D10"}.issubset(feats.columns):
        feats["D50_over_D10"] = _safe_ratio(feats["D50"], feats["D10"])
    if {"D80", "D20"}.issubset(feats.columns):
        feats["D80_over_D20"] = _safe_ratio(feats["D80"], feats["D20"])

    return feats


# ---------------------------
# Stepwise AIC
# ---------------------------

def stepwise_aic(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    candidate_features: Optional[List[str]] = None,
    max_steps: int = 50,
    verbose: bool = True,
) -> Tuple[List[str], sm.regression.linear_model.RegressionResultsWrapper]:

    if candidate_features is None:
        candidate_features = list(X.columns)

    selected: List[str] = []

    def fit_aic(cols):
        if not cols:
            return np.inf, None
        X_ = sm.add_constant(X[cols], has_constant="add")
        model = sm.OLS(y, X_, missing="drop").fit()
        return model.aic, model

    current_aic, current_model = fit_aic(selected)

    if verbose:
        print(f"Start AIC: {current_aic:.3f}")

    for step in range(max_steps):
        improved = False

        # forward
        remaining = list(set(candidate_features) - set(selected))
        forward = [(fit_aic(selected + [c])[0], c) for c in remaining]

        # backward
        backward = [(fit_aic([c for c in selected if c != r])[0], r)
                    for r in selected]

        best_fwd = min(forward, default=(np.inf, None))
        best_bwd = min(backward, default=(np.inf, None))

        if best_fwd[0] < current_aic and best_fwd[0] <= best_bwd[0]:
            selected.append(best_fwd[1])
            current_aic, current_model = fit_aic(selected)
            improved = True
            if verbose:
                print(f"ADD {best_fwd[1]} arrow AIC {current_aic:.3f}")

        elif best_bwd[0] < current_aic:
            selected.remove(best_bwd[1])
            current_aic, current_model = fit_aic(selected)
            improved = True
            if verbose:
                print(f"REMOVE {best_bwd[1]} arrow AIC {current_aic:.3f}")

        if not improved:
            break

    return selected, current_model


# ---------------------------
# Main (NO SPLIT)
# ---------------------------

def fit_stepwise_models(
    xlsx_path: str,
    sheet_ign: str = "ign",
    sheet_psd: str = "PSD",
    target_moisture: str = "Mc_%",
    target_porosity: Optional[str] = "Cake_por",
    verbose: bool = True,
):

    # --- Read ---
    df_ign = pd.read_excel(xlsx_path, sheet_name=sheet_ign, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    df_ign.columns = df_ign.columns.astype(str).str.strip()
    df_psd.columns = df_psd.columns.astype(str).str.strip()

    # --- Filter ---
    df_ign = _flag_include_only(df_ign)

    # --- PSD features ---
    feats = _build_psd_features(df_psd)

    # --- Merge ---
    if "Sample Code" not in df_ign.columns:
        raise ValueError("'Sample Code' not found in ign sheet.")

    df = pd.merge(df_ign, feats, on="Sample Code", how="inner")

    # --- Feature set ---
    base_feats = [c for c in feats.columns if c != "Sample Code"]

    # --- Moisture model ---
    def run_target(target, label):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")

        df_m = df[[target] + base_feats].dropna()

        X = df_m[base_feats]
        y = df_m[target].astype(float)

        scaler = StandardScaler()
        Xs = pd.DataFrame(
            scaler.fit_transform(X),
            columns=base_feats,
            index=X.index,
        )

        if verbose:
            print("\n" + "=" * 80)
            print(f"STEPWISE AIC - {label}")
            print("=" * 80)

        selected, model = stepwise_aic(
            Xs, y, candidate_features=base_feats, verbose=verbose
        )

        print(model.summary())
        print(f"\nSelected features ({label}): {selected}")
        coef_table = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        })

       # coef_table = coef_table.drop("const", errors="ignore")

        print("\nCoefficient significance table:")
        print(coef_table.to_string(float_format="{:.4g}".format))

        return {"selected": selected, "model": model}

    res_moist = run_target(target_moisture, "Final Moisture")

    res_poro = None
    if target_porosity is not None:
        res_poro = run_target(target_porosity, "Cake Porosity")

    return {
        "features_used": base_feats,
        "moisture": res_moist,
        "porosity": res_poro,
    }


# ---------------------------
# Run directly
# ---------------------------

if __name__ == "__main__":
    fit_stepwise_models(
        xlsx_path=r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_DB_all.xlsx",
        sheet_ign="ign",
        sheet_psd="PSD",
        target_moisture="Mc_%",
        target_porosity="Cake_por",  # set to None if not present
        verbose=True,
    )
