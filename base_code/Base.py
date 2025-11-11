# Models/stepwise_psd_models.py
# -*- coding: utf-8 -*-
import sys
import warnings
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# error checking
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

# error checking
def _safe_ratio(a: pd.Series, b: pd.Series, name: str) -> pd.Series:
    """Compute a/b, guarding against zero/NaN denominators."""
    out = pd.Series(np.nan, index=a.index, name=name, dtype=float)
    valid = b.replace(0, np.nan).notna() & a.notna()
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out

# taking in PSD data plus do calcs
def _build_psd_features(df_psd: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: 'Sample Code' and any of D10,D20,D50,D80,D90.
    Creates ratio features too. Returns feature frame keyed by 'Sample Code'.
    """
    needed_any = ["D10", "D20", "D50", "D80", "D90"]
    missing = [c for c in needed_any if c not in df_psd.columns]
    if missing:
        warnings.warn(f"Missing PSD columns {missing}. Continuing with available columns.")

    cols = ["Sample Code"] + [c for c in needed_any if c in df_psd.columns]
    feats = df_psd[cols].copy()

    # Ratios (compute only if ingredients exist)
    if {"D90", "D50"}.issubset(feats.columns):
        feats["D90_over_D50"] = _safe_ratio(feats["D90"], feats["D50"], "D90_over_D50")
    if {"D50", "D10"}.issubset(feats.columns):
        feats["D50_over_D10"] = _safe_ratio(feats["D50"], feats["D10"], "D50_over_D10")
    if {"D80", "D20"}.issubset(feats.columns):
        feats["D80_over_D20"] = _safe_ratio(feats["D80"], feats["D20"], "D80_over_D20")
    if {"D90", "D10"}.issubset(feats.columns):
        feats["D90_over_D10"] = _safe_ratio(feats["D90"], feats["D10"], "D90_over_D10")
    if {"D80", "D50"}.issubset(feats.columns):
        feats["D80_over_D50"] = _safe_ratio(feats["D80"], feats["D50"], "D80_over_D50")
    if {"D80", "D50"}.issubset(feats.columns):
        feats["D50_over_D20"] = _safe_ratio(feats["D50"], feats["D20"], "D50_over_D20")
    return feats

# VIF table for chsoen PSD characteristisc = Variance Inflation Factors (VIF)
def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_ = X.dropna().copy()
    X_ = sm.add_constant(X_, has_constant='add')
    names = ["const"] + list(X_.columns.drop("const"))
    vifs = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
    return pd.DataFrame({"feature": names, "VIF": vifs})


# def _sig_stars(p: float) -> str:
#     if p < 1e-3: return "***"
#     if p < 1e-2: return "**"
#     if p < 5e-2: return "*"
#     if p < 1e-1: return "."
#     return ""


# def _print_effects(model: sm.regression.linear_model.RegressionResultsWrapper,
#                    selected: List[str],
#                    label: str,
#                    save_csv_path: Optional[str] = None) -> None:
#     """
#     Print a tidy 'effects' report for selected features (standardized X):
#       - coefficient (beta), p-value, 95% CI
#       - statement: 'as X increases by 1 SD, y increases/decreases by beta units'
#     Optionally saves a CSV.
#     """
#     if model is None or not selected:
#         print("\nNo effects to report (no features selected).")
#         return

#     params = model.params.copy()
#     pvals = model.pvalues.copy()
#     conf = model.conf_int()
#     rows: List[Dict[str, Any]] = []

#     print("\nEffect report (standardized predictors):")
#     print("  ? Interpret B as change in target units per +1 SD of the predictor.\n")

#     for f in selected:
#         if f not in params.index:
#             continue
#         beta = float(params[f])
#         p = float(pvals.get(f, np.nan))
#         ci_lo = float(conf.loc[f, 0]) if f in conf.index else np.nan
#         ci_hi = float(conf.loc[f, 1]) if f in conf.index else np.nan
#         star = _sig_stars(p)
#         direction = "decreases" if beta < 0 else "increases"
#         txt = (f"As {f} increases by 1 SD, {label} {direction} by {abs(beta):.3f} "
#                f"(?={beta:.3f}{star}, p={p:.3g}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]).")
#         print(" - " + txt)
#         rows.append({
#             "feature": f, "beta": beta, "p_value": p, "sig": star,
#             "ci_low": ci_lo, "ci_high": ci_hi, "direction": direction,
#             "interpretation": txt
#         })

    # if save_csv_path:
    #     try:
    #         pd.DataFrame(rows).to_csv(save_csv_path, index=False)
    #         print(f"\nSaved effect table to: {save_csv_path}")
    #     except Exception as e:
    #         warnings.warn(f"Could not save effect CSV: {e}")

    #AIC func
def stepwise_aic(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial: Optional[List[str]] = None,
    candidate_features: Optional[List[str]] = None,
    max_steps: int = 50,
    min_delta_aic: float = 0.0,
    verbose: bool = True,
    # Optional p-value gates (still AIC-driven; p-gates restrict candidates)
    pvalue_gate_in: Optional[float] = None,     # e.g. 0.05 to allow adds only if  < 0.05
    pvalue_gate_out: Optional[float] = None,    # e.g. 0.10 to allow remove if p 0.10
) -> Tuple[List[str], Optional[sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    Forward-backward stepwise selection using AIC on an OLS model.    
    """
    X = X.copy()
    y = y.copy()

    if candidate_features is None:
        candidate_features = list(X.columns)

    selected: List[str] = []
    if initial is not None:
        selected = [f for f in initial if f in candidate_features]

    def fit(cols: List[str]) -> Tuple[float, Optional[sm.regression.linear_model.RegressionResultsWrapper]]:
        if len(cols) == 0:
            return np.inf, None
        X_ = sm.add_constant(X[cols], has_constant='add')
        model_ = sm.OLS(y, X_, missing='drop').fit()
        return model_.aic, model_

    current_aic, current_model = fit(selected)
    if verbose:
        print(f"Start AIC: {current_aic:.3f} with features={selected}")

    steps = 0
    improved = True

    while improved and steps < max_steps:
        steps += 1
        improved = False

        # ----- Try forward adds -----
        remaining = list(set(candidate_features) - set(selected))
        add_candidates: List[Tuple[float, str, Optional[sm.regression.linear_model.RegressionResultsWrapper]]] = []
        for c in remaining:
            aic_c, m_c = fit(selected + [c])
            # Optional p-value gate for entry
            if pvalue_gate_in is not None and m_c is not None:
                # p-value for 'c' must be below gate to consider
                if c in m_c.pvalues.index and not np.isnan(m_c.pvalues[c]):
                    if m_c.pvalues[c] > pvalue_gate_in:
                        continue
                else:
                    continue
            add_candidates.append((aic_c, c, m_c))

        best_add_aic, best_add, _ = (min(add_candidates, key=lambda t: t[0]) if add_candidates
                                     else (np.inf, None, None))

        # ----- Try backward removes -----
        remove_candidates: List[Tuple[float, str, Optional[sm.regression.linear_model.RegressionResultsWrapper]]] = []
        for c in list(selected):
            cols = [f for f in selected if f != c]
            aic_c, m_c = fit(cols)
            # Optional p-value gate for removal (use p from current model if available)
            if pvalue_gate_out is not None and current_model is not None:
                if c in current_model.pvalues.index and not np.isnan(current_model.pvalues[c]):
                    # Only allow removal if current p-value is > gate (i.e., not significant)
                    if current_model.pvalues[c] <= pvalue_gate_out:
                        continue
            remove_candidates.append((aic_c, c, m_c))

        best_remove_aic, best_remove, _ = (min(remove_candidates, key=lambda t: t[0]) if remove_candidates
                                           else (np.inf, None, None))

        # ----- Decide move -----
        best_current = current_aic
        best_move = min(best_add_aic, best_remove_aic, best_current)

        if best_add is not None and (best_add_aic + min_delta_aic) < current_aic and best_add_aic <= best_remove_aic:
            selected.append(best_add)
            current_aic, current_model = fit(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: ADD {best_add}  -> AIC {current_aic:.3f}")

        elif best_remove is not None and (best_remove_aic + min_delta_aic) < current_aic and best_remove_aic < best_add_aic:
            selected.remove(best_remove)
            current_aic, current_model = fit(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: REMOVE {best_remove}  -> AIC {current_aic:.3f}")

        else:
            if verbose:
                print(f"Step {steps}: no improvement (AIC {current_aic:.3f})")

    return selected, current_model


#Main func

def fit_stepwise_models(
    xlsx_path: str,
    sheet_db: str = "DB",
    sheet_psd: str = "PSD",
    target_moisture: str = "Mc_%",
    target_porosity: str = "Cake_por",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    # Optional: add p-value gates and VIF warnings
    pvalue_gate_in: Optional[float] = None,
    pvalue_gate_out: Optional[float] = None,
    warn_vif_gt: Optional[float] = 10.0,   # warn if any selected VIF exceeds this
) -> Dict[str, Any]:
    """
    Steps:
      1) Read DB & PSD, keep rows with flag == include
      2) PSD features & merge
      3) Train/test split, standardize X
      4) Stepwise-AIC OLS models for two targets 
      5) Print summaries, effects, VIFs; save predictions CSV
    Returns dict with per-target results and scaler.
    """
    # --- Read sheets ---
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    # --- Keep only 'include' rows in DB ---
    df_db = _flag_include_only(df_db)

    # --- Build features from PSD ---
    feats = _build_psd_features(df_psd)

    # --- Merge ---
    for col in ["Sample Code", target_moisture, target_porosity]:
        if col not in df_db.columns and col != "Sample Code":
            raise ValueError(f"Required column '{col}' not found in DB.")
    if "Sample Code" not in df_db.columns or "Sample Code" not in feats.columns:
        raise ValueError("'Sample Code' must exist in both DB and PSD sheets.")

    df = pd.merge(df_db, feats, on="Sample Code", how="inner")

    # --- Define X and y's ---
    base_feats = [c for c in [
        "D10", "D20", "D50", "D80", "D90",
        "D90_over_D50", "D50_over_D10", "D80_over_D20", "D90_over_D10", "D50_over_D20", "D80_over_50"
    ] if c in df.columns]

    if len(base_feats) == 0:
        raise ValueError("No PSD features available after merging. Check your PSD sheet columns.")

    # Drop rows missing targets or features
    df_model = df[["Sample Code", target_moisture, target_porosity] + base_feats].dropna()

    # Train/test split indices so both targets share the same split
    train_idx, test_idx = train_test_split(
        df_model.index, test_size=test_size, random_state=random_state
    )
    df_train = df_model.loc[train_idx].copy()
    df_test = df_model.loc[test_idx].copy()

    X_train = df_train[base_feats].copy()
    X_test = df_test[base_feats].copy()

    # Standardize X (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=base_feats, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=base_feats, index=X_test.index)

    # Helper: fit one target
    def fit_one_target(target_col: str, label: str) -> Dict[str, Any]:
        y_train = df_train[target_col].astype(float)
        y_test_ = df_test[target_col].astype(float)

        if verbose:
            print("\n" + "=" * 80)
            print(f"Stepwise model for {label} ({target_col})")
            print("=" * 80)

        selected, model = stepwise_aic(
            X_train_s, y_train,
            candidate_features=base_feats,
            verbose=verbose,
            pvalue_gate_in=pvalue_gate_in,
            pvalue_gate_out=pvalue_gate_out
        )

        # Print model summary
        if model is not None:
            print(model.summary())

        # Human-readable effects report + CSV
        # _print_effects(
        #     model,
        #     selected,
        #     label,
        #     save_csv_path=f"effects_{target_col.replace('%','pct').replace(' ','_')}.csv"
        # )

        # Predictions on test
        if selected and model is not None:
            X_test_sel = sm.add_constant(X_test_s[selected], has_constant='add')
            y_pred = model.predict(X_test_sel)
        else:
            # No features selected: predict train mean
            y_pred = pd.Series(y_train.mean(), index=y_test_.index)

        # Metrics
        mae = float(np.mean(np.abs(y_test_ - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test_ - y_pred) ** 2)))
        r2 = float(1 - np.sum((y_test_ - y_pred) ** 2) / np.sum((y_test_ - np.mean(y_test_)) ** 2))
        if verbose:
            print(f"Selected features ({label}): {selected}")
            print(f"Test MAE={mae:.3f}, RMSE={rmse:.3f}, R^2={r2:.3f}")

        # VIF (informational; use original unscaled X to check raw collinearity)
        try:
            if selected:
                vif = _vif_table(X_train[selected])
                print("\nVIF (train, selected features):")
                print(vif.to_string(index=False))
                if warn_vif_gt is not None:
                    high = vif.loc[vif["feature"] != "const"].query("VIF > @warn_vif_gt")
                    if not high.empty:
                        warnings.warn(
                            f"High VIF detected > {warn_vif_gt}: "
                            + ", ".join(f"{r.feature}={r.VIF:.1f}" for _, r in high.iterrows())
                        )
        except Exception as e:
            warnings.warn(f"VIF calculation failed: {e}")

        return {
            "target": target_col,
            "label": label,
            "selected_features": selected,
            "model": model,
            "y_test": y_test_,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    # Fit both targets
    res_moist = fit_one_target(target_moisture, "Final moisture")
    res_poro = fit_one_target(target_porosity, "Cake porosity")

    # Save predictions CSV
    out = pd.DataFrame({
        "Sample Code": df_test["Sample Code"],
        f"{target_moisture}_actual": res_moist["y_test"],
        f"{target_moisture}_pred": res_moist["y_pred"],
        f"{target_porosity}_actual": res_poro["y_test"],
        f"{target_porosity}_pred": res_poro["y_pred"],
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
    if len(sys.argv) == 2:
        path = sys.argv[1]
        fit_stepwise_models(path)
    else:
        print("Usage:\n  python -m Models.stepwise_psd_models <path/to/data.xlsx>\n"
              "Or import and call fit_stepwise_models(...) from Python.")
