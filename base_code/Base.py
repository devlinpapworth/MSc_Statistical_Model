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
    initial: Optional[List[str]] = None,
    candidate_features: Optional[List[str]] = None,
    max_steps: int = 50,
    verbose: bool = True,
) -> Tuple[List[str], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Forward-backward stepwise selection using AIC on an OLS model.

    Returns (selected_feature_names, final_model).
    """
    X = X.copy()
    y = y.copy()

    if candidate_features is None:
        candidate_features = list(X.columns)

    if initial is None:
        selected: List[str] = []
    else:
        selected = [f for f in initial if f in candidate_features]

    def fit_aic(cols: List[str]) -> Tuple[float, Optional[sm.regression.linear_model.RegressionResultsWrapper]]:
        if len(cols) == 0:
            return np.inf, None
        X_ = sm.add_constant(X[cols], has_constant='add')
        model = sm.OLS(y, X_, missing='drop').fit()
        return model.aic, model

    # Start
    current_aic, current_model = fit_aic(selected)
    if verbose:
        print(f"Start AIC: {current_aic:.3f} with features={selected}")

    steps = 0
    improved = True

    while improved and steps < max_steps:
        steps += 1
        improved = False

        # Try forward adds
        remaining = list(set(candidate_features) - set(selected))
        aic_with_candidates = []
        for c in remaining:
            aic_c, _ = fit_aic(selected + [c])
            aic_with_candidates.append((aic_c, c))

        if aic_with_candidates:
            best_add_aic, best_add = min(aic_with_candidates, key=lambda t: t[0])
        else:
            best_add_aic, best_add = np.inf, None

        # Try backward removes
        aic_with_removals = []
        for c in list(selected):
            cols = [f for f in selected if f != c]
            aic_c, _ = fit_aic(cols)
            aic_with_removals.append((aic_c, c))  # AIC if we remove c

        if aic_with_removals:
            best_remove_aic, best_remove = min(aic_with_removals, key=lambda t: t[0])
        else:
            best_remove_aic, best_remove = np.inf, None

        # Decide best move
        best_move = min(best_add_aic, best_remove_aic, current_aic)

        if best_add is not None and best_add_aic < current_aic and best_add_aic <= best_remove_aic:
            selected.append(best_add)
            current_aic, current_model = fit_aic(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: ADD {best_add}  -> AIC {current_aic:.3f}")

        elif best_remove is not None and best_remove_aic < current_aic and best_remove_aic < best_add_aic:
            selected.remove(best_remove)
            current_aic, current_model = fit_aic(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: REMOVE {best_remove}  -> AIC {current_aic:.3f}")

        else:
            if verbose:
                print(f"Step {steps}: no improvement (AIC {current_aic:.3f})")

    return selected, current_model


# ---------------------------
# Main modeling function
# ---------------------------

def fit_stepwise_models(
    xlsx_path: str,
    sheet_db: str = "DB",
    sheet_psd: str = "PSD",
    target_moisture: str = "Mc_%",
    target_porosity: str = "Cake_por",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    1) Read DB & PSD, keep rows with flag == include
    2) Engineer PSD features & merge
    3) Train/test split, standardize X
    4) Stepwise-AIC OLS models for two targets
    5) Print summaries and save predictions CSV
    """

    # --- Read sheets ---
    df_db  = pd.read_excel(xlsx_path, sheet_name=sheet_db,  engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # --- Keep only 'include' rows in DB ---
    df_db = _flag_include_only(df_db)

    # --- Build features from PSD ---
    feats = _build_psd_features(df_psd)

    # --- Merge ---
    # Expect 'Sample Code' in both
    for col in ["Sample Code", target_moisture, target_porosity]:
        if col not in df_db.columns and col != "Sample Code":
            raise ValueError(f"Required column '{col}' not found in DB.")
    if "Sample Code" not in df_db.columns or "Sample Code" not in feats.columns:
        raise ValueError("'Sample Code' must exist in both DB and PSD sheets.")

    df = pd.merge(df_db, feats, on="Sample Code", how="inner")

    # --- Define X and y's ---
    base_feats = [c for c in ["D10","D20","D50","D80","D90",
                              "D90_over_D50","D50_over_D10","D80_over_D20"]
                  if c in df.columns]

    if len(base_feats) == 0:
        raise ValueError("No PSD features available after merging. Check your PSD sheet columns.")

    # Drop rows missing targets or features
    df_model = df[["Sample Code", target_moisture, target_porosity] + base_feats].dropna()

    # Train/test split indices so both targets share the same split
    train_idx, test_idx = train_test_split(
        df_model.index, test_size=test_size, random_state=random_state
    )
    df_train = df_model.loc[train_idx].copy()
    df_test  = df_model.loc[test_idx].copy()

    X_train = df_train[base_feats].copy()
    X_test  = df_test[base_feats].copy()

    # Standardize X (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=base_feats, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=base_feats, index=X_test.index)

    # Helper to fit one target
    def fit_one_target(target_col: str, label: str):
        y_train = df_train[target_col].astype(float)
        y_test  = df_test[target_col].astype(float)

        if verbose:
            print("\n" + "="*80)
            print(f"Stepwise model for {label} ({target_col})")
            print("="*80)

        selected, model = stepwise_aic(X_train_s, y_train, candidate_features=base_feats, verbose=verbose)

        # Print model summary
        if model is not None:
            print(model.summary())

        # Predictions on test
        if selected:
            X_test_sel = sm.add_constant(X_test_s[selected], has_constant='add')
            y_pred = model.predict(X_test_sel)
        else:
            # No features selected ? predict mean
            y_pred = pd.Series(y_train.mean(), index=y_test.index)

        # Metrics
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
        r2 = float(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        if verbose:
            print(f"Selected features ({label}): {selected}")
            print(f"Test MAE={mae:.3f}, RMSE={rmse:.3f}, R^2={r2:.3f}")

        # VIF (informational; use original unscaled X to check raw collinearity if you prefer)
        try:
            if selected:
                vif = _vif_table(X_train[selected])
                print("\nVIF (train, selected features):")
                print(vif.to_string(index=False))
        except Exception as e:
            warnings.warn(f"VIF calculation failed: {e}")

        return {
            "target": target_col,
            "label": label,
            "selected_features": selected,
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    # Fit both targets
    res_moist = fit_one_target(target_moisture, "Final moisture")
    res_poro  = fit_one_target(target_porosity, "Cake porosity")

    # Save predictions
    out = pd.DataFrame({
        "Sample Code": df_test["Sample Code"],
        f"{target_moisture}_actual": res_moist["y_test"],
        f"{target_moisture}_pred":   res_moist["y_pred"],
        f"{target_porosity}_actual": res_poro["y_test"],
        f"{target_porosity}_pred":   res_poro["y_pred"],
    })
    out_path = "stepwise_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved test-set predictions to: {out_path}")

    return {
        "features_used": base_feats,
        "res_moisture": res_moist,
        "res_porosity": res_poro,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # Example usage:
    # python -m Models.stepwise_psd_models
    print("Run fit_stepwise_models(xlsx_path='path/to/your.xlsx') from a Python REPL or another script.")
