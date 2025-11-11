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
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# Utilities
# =========================

def _flag_include_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where 'flag' contains 'include' (case-insensitive)."""
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


def build_psd_features(df_psd: pd.DataFrame) -> pd.DataFrame:
    """
    Expect: 'Sample Code' and any of D10,D20,D50,D80,D90. Create ratio features.
    """
    need_any = ["D10", "D20", "D50", "D80", "D90"]
    missing = [c for c in need_any if c not in df_psd.columns]
    if missing:
        warnings.warn(f"Missing PSD columns {missing}. Continuing with available columns.")

    cols = ["Sample Code"] + [c for c in need_any if c in df_psd.columns]
    feats = df_psd[cols].copy()

    # Ratios (only if ingredients exist)
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
    if {"D50", "D20"}.issubset(feats.columns):
        feats["D50_over_D20"] = _safe_ratio(feats["D50"], feats["D20"], "D50_over_D20")

    return feats


def _design_diagnostics(X: pd.DataFrame, tag: str):
    """Print quick condition numbers for diagnostics."""
    Xn = X.dropna().to_numpy()
    if Xn.size == 0:
        return
    try:
        # plain cond on X (without constant)
        cn = np.linalg.cond(Xn)
        print(f"[Design diagnostics] Condition number ({tag}): {cn:.2e}")
    except Exception:
        pass


def _orthogonalize_ratios(X: pd.DataFrame, raw_cols: List[str], ratio_cols: List[str]) -> pd.DataFrame:
    """
    Residualize ratio_cols on raw_cols to remove linear dependence.
    New columns <ratio>_orth replace the original ratios in returned frame.
    """
    X = X.copy()
    have_raw = [c for c in raw_cols if c in X.columns]
    have_rat = [c for c in ratio_cols if c in X.columns]
    if not have_raw or not have_rat:
        return X

    # add constant for the small regressions
    X_raw = sm.add_constant(X[have_raw], has_constant='add')
    for rc in have_rat:
        y = X[rc]
        # small OLS to get residuals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X_raw, missing='drop').fit()
        resid = y - res.predict(X_raw)
        X[f"{rc}_orth"] = resid
        # optionally drop the original ratio to avoid duplication
        X.drop(columns=[rc], inplace=True)

    return X


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF on matrix with constant added."""
    X_ = X.dropna().copy()
    Xc = sm.add_constant(X_, has_constant='add').values
    names = ["const"] + list(X_.columns)
    vifs = [variance_inflation_factor(Xc, i) for i in range(Xc.shape[1])]
    return pd.DataFrame({"feature": names, "VIF": vifs})


# =========================
# Stepwise selection
# =========================

def _criterion_value(model, criterion: str) -> float:
    criterion = criterion.lower()
    if criterion == "aic":
        return model.aic
    elif criterion == "bic":
        return model.bic
    else:
        raise ValueError("criterion must be 'aic' or 'bic'")


def stepwise_select(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    mode: str = "both",               # "forward", "backward", "both"
    criterion: str = "bic",           # "aic" or "bic"
    initial: Optional[List[str]] = None,
    candidate_features: Optional[List[str]] = None,
    max_steps: int = 100,
    min_delta_ic: float = 0.0,
    max_features: Optional[int] = None,
    pvalue_gate_in: Optional[float] = None,
    pvalue_gate_out: Optional[float] = None,
    emergency_drop_p: Optional[float] = 0.20,  # repeat-until-clean
    verbose: bool = True,
) -> Tuple[List[str], Optional[sm.regression.linear_model.RegressionResultsWrapper], float]:
    """
    Bidirectional/forward/backward stepwise using AIC/BIC.
    - mode controls allowed moves.
    - emergency drop repeatedly removes worst p if > emergency_drop_p.
    Returns: (selected_features, model, criterion_value)
    """
    X = X.copy()
    y = y.copy()

    if candidate_features is None:
        candidate_features = list(X.columns)

    # Start set by mode
    mode = mode.lower()
    if mode == "forward":
        selected: List[str] = [] if initial is None else [f for f in initial if f in candidate_features]
    elif mode == "backward":
        selected = list(candidate_features) if initial is None else [f for f in initial if f in candidate_features]
    else:
        selected = [] if initial is None else [f for f in initial if f in candidate_features]

    def fit(cols: List[str]):
        if len(cols) == 0:
            return np.inf, None
        X_ = sm.add_constant(X[cols], has_constant='add')
        m_ = sm.OLS(y, X_, missing='drop').fit()
        return _criterion_value(m_, criterion), m_

    current_ic, current_model = fit(selected)
    if verbose:
        print(f"Start {criterion.upper()}: {current_ic:.3f} with features={selected}")

    steps = 0
    improved = True

    while improved and steps < max_steps:
        steps += 1
        improved = False

        # ---------- Try forward additions ----------
        best_add_ic, best_add, _m_add = np.inf, None, None
        if mode in ("forward", "both"):
            remaining = [c for c in candidate_features if c not in selected]
            add_candidates = []
            for c in remaining:
                # respect max_features
                if max_features is not None and (len(selected) + 1) > max_features:
                    continue
                ic_c, m_c = fit(selected + [c])
                # p-gate for entry
                if pvalue_gate_in is not None and m_c is not None and c in m_c.pvalues.index:
                    if np.isnan(m_c.pvalues[c]) or (m_c.pvalues[c] > pvalue_gate_in):
                        continue
                add_candidates.append((ic_c, c, m_c))
            if add_candidates:
                best_add_ic, best_add, _m_add = min(add_candidates, key=lambda t: t[0])

        # ---------- Try backward removals ----------
        best_rem_ic, best_rem, _m_rem = np.inf, None, None
        if mode in ("backward", "both") and selected:
            rem_candidates = []
            for c in list(selected):
                # do not remove if doing forward-only
                ic_c, m_c = fit([f for f in selected if f != c])
                # p-gate for removal: only remove if term looks weak in current model
                if pvalue_gate_out is not None and current_model is not None and c in current_model.pvalues.index:
                    pv = current_model.pvalues[c]
                    if not np.isnan(pv) and pv <= pvalue_gate_out:
                        # keep significant terms
                        continue
                rem_candidates.append((ic_c, c, m_c))
            if rem_candidates:
                best_rem_ic, best_rem, _m_rem = min(rem_candidates, key=lambda t: t[0])

        # ---------- Decide move ----------
        best_current = current_ic
        move_ic = min(best_add_ic, best_rem_ic, best_current)

        if best_add is not None and (best_add_ic + min_delta_ic) < current_ic and best_add_ic <= best_rem_ic:
            selected.append(best_add)
            current_ic, current_model = fit(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: ADD {best_add}  -> {criterion.upper()} {current_ic:.3f}")

        elif best_rem is not None and (best_rem_ic + min_delta_ic) < current_ic and best_rem_ic < best_add_ic:
            selected.remove(best_rem)
            current_ic, current_model = fit(selected)
            improved = True
            if verbose:
                print(f"Step {steps}: REMOVE {best_rem}  -> {criterion.upper()} {current_ic:.3f}")

        else:
            # ---------- Emergency drop: repeat-until-clean ----------
            dropped_any = False
            if emergency_drop_p is not None and current_model is not None and len(selected) > 1:
                while True:
                    pvals = current_model.pvalues.drop(labels=["const"], errors="ignore")
                    if pvals.empty:
                        break
                    worst_feat = pvals.idxmax()
                    worst_p = float(pvals.max())
                    if worst_p <= emergency_drop_p:
                        break
                    selected.remove(worst_feat)
                    current_ic, current_model = fit(selected)
                    dropped_any = True
                    if verbose:
                        print(f"Step {steps}: EMERGENCY DROP '{worst_feat}' (p={worst_p:.3f}) "
                              f"-> {criterion.upper()} {current_ic:.3f}")
                    # respect max_features (always true after drop)

            if dropped_any:
                improved = True
            else:
                if verbose:
                    print(f"Step {steps}: no improvement ({criterion.upper()} {current_ic:.3f})")

    return selected, current_model, current_ic


# =========================
# Main modeling
# =========================

def fit_stepwise_models(
    xlsx_path: str,
    sheet_db: str = "DB",
    sheet_psd: str = "PSD",
    target_moisture: str = "Mc_%",
    target_porosity: str = "Cake_por",
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    mode: str = "both",               # "forward" | "backward" | "both"
    criterion: str = "aic",           # "aic" | "bic"
    max_features: Optional[int] = None,
    pvalue_gate_in: Optional[float] = None,
    pvalue_gate_out: Optional[float] = None,
    emergency_drop_p: Optional[float] = 0.20,
    warn_vif_gt: float = 10.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    1) Read DB & PSD, keep 'include'
    2) Build raw + ratio features
    3) Orthogonalize ratios on raw sizes
    4) Train/test split, standardize X
    5) Stepwise with chosen mode/criterion + gates
    6) Print summaries, VIF, diagnostics; save predictions CSV
    """

    # --- Load ---
    df_db = pd.read_excel(xlsx_path, sheet_name=sheet_db, engine="openpyxl")
    df_psd = pd.read_excel(xlsx_path, sheet_name=sheet_psd, engine="openpyxl")

    df_db = _flag_include_only(df_db)
    feats = build_psd_features(df_psd)

    # --- Merge ---
    needed_in_db = ["Sample Code", target_moisture, target_porosity]
    for col in needed_in_db:
        if col not in df_db.columns:
            raise ValueError(f"Required column '{col}' not found in DB.")
    if "Sample Code" not in feats.columns:
        raise ValueError("'Sample Code' must exist in PSD sheet.")
    df = pd.merge(df_db, feats, on="Sample Code", how="inner")

    # --- Features (raw + ratios) ---
    raw_cols = [c for c in ["D10", "D20", "D50", "D80", "D90"] if c in df.columns]
    ratio_cols_all = [c for c in [
        "D90_over_D50", "D50_over_D10", "D80_over_D20", "D90_over_D10", "D50_over_D20", "D80_over_D50"
    ] if c in df.columns]

    all_feats = raw_cols + ratio_cols_all
    if not all_feats:
        raise ValueError("No PSD features available after merging.")

    df_model = df[["Sample Code", target_moisture, target_porosity] + all_feats].dropna()

    # --- Train/test split ---
    train_idx, test_idx = train_test_split(df_model.index, test_size=test_size, random_state=random_state)
    df_train = df_model.loc[train_idx].copy()
    df_test = df_model.loc[test_idx].copy()

    # --- Build X (orthogonalize ratios) ---
    X_train_raw = df_train[all_feats].copy()
    X_test_raw = df_test[all_feats].copy()

    X_train_ortho = _orthogonalize_ratios(X_train_raw, raw_cols, ratio_cols_all)
    X_test_ortho = _orthogonalize_ratios(X_test_raw, raw_cols, ratio_cols_all)

    # after orthogonalization, rename ratio set to *_orth present
    ratio_ortho = [c for c in X_train_ortho.columns if c.endswith("_orth")]
    feat_cols_final = raw_cols + ratio_ortho

    # --- Standardize ---
    scaler_X = StandardScaler()
    X_train_s = pd.DataFrame(scaler_X.fit_transform(X_train_ortho[feat_cols_final]),
                             columns=feat_cols_final, index=X_train_ortho.index)
    X_test_s = pd.DataFrame(scaler_X.transform(X_test_ortho[feat_cols_final]),
                            columns=feat_cols_final, index=X_test_ortho.index)

    # Diagnostics
    _design_diagnostics(X_train_s, "std+orthog")

    def fit_one(target_col: str, label: str) -> Dict[str, Any]:
        y_tr = df_train[target_col].astype(float)
        y_te = df_test[target_col].astype(float)

        if verbose:
            print("\n" + "=" * 80)
            print(f"Stepwise model for {label} ({target_col})")
            print("=" * 80)

        selected, model, ic_val = stepwise_select(
            X_train_s, y_tr,
            mode=mode,
            criterion=criterion,
            candidate_features=feat_cols_final,
            max_steps=100,
            min_delta_ic=0.0,
            max_features=max_features,
            pvalue_gate_in=pvalue_gate_in,
            pvalue_gate_out=pvalue_gate_out,
            emergency_drop_p=emergency_drop_p,
            verbose=verbose,
        )

        # Summary
        if model is not None:
            print(model.summary())

        # Predict
        if selected and model is not None:
            Xte_sel = sm.add_constant(X_test_s[selected], has_constant='add')
            y_pred = model.predict(Xte_sel)
        else:
            y_pred = pd.Series(y_tr.mean(), index=y_te.index)

        # Metrics
        mae = float(np.mean(np.abs(y_te - y_pred)))
        rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
        r2 = float(1 - np.sum((y_te - y_pred) ** 2) / np.sum((y_te - np.mean(y_te)) ** 2))
        if verbose:
            print(f"Selected features ({label}): {selected}")
            print(f"Test MAE={mae:.3f}, RMSE={rmse:.3f}, R^2={r2:.3f}")

        # VIF on transformed selected features
        try:
            if selected:
                vif = _vif_table(X_train_s[selected])
                print("\nVIF (train, transformed selected features):")
                print(vif.to_string(index=False))
                high = vif.loc[vif["feature"] != "const"].query("VIF > @warn_vif_gt")
                if not high.empty:
                    warnings.warn(
                        "High VIF detected > "
                        + f"{warn_vif_gt}: "
                        + ", ".join(f"{r.feature}={r.VIF:.1f}" for _, r in high.iterrows())
                    )
        except Exception as e:
            warnings.warn(f"VIF calculation failed: {e}")

        return {
            "target": target_col,
            "label": label,
            "selected_features": selected,
            "model": model,
            "ic": ic_val,
            "y_test": y_te,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        # end fit_one

    # Fit both targets
    res_moist = fit_one(target_moisture, "Final moisture")
    res_poro = fit_one(target_porosity, "Cake porosity")

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

    # Also return a predictor that reproduces the exact pipeline
    def make_predictor(result: Dict[str, Any]):
        selected = result["selected_features"]
        model = result["model"]

        def _predict(raw_df: pd.DataFrame) -> np.ndarray:
            Xraw = raw_df[feat_cols_final].copy()
            # NOTE: caller must supply same columns; if not, align/raise.
            Xs = pd.DataFrame(scaler_X.transform(Xraw), columns=feat_cols_final, index=raw_df.index)
            Xsel = sm.add_constant(Xs[selected], has_constant='add')
            return model.predict(Xsel)
        return _predict

    return {
        "features_used_raw": raw_cols,
        "features_used_ratio": ratio_cols_all,
        "features_after_orthog": feat_cols_final,
        "scaler_X": scaler_X,
        "res_moisture": res_moist,
        "res_porosity": res_poro,
        "predict_moisture": make_predictor(res_moist),
        "predict_porosity": make_predictor(res_poro),
        "config": {
            "mode": mode,
            "criterion": criterion,
            "max_features": max_features,
            "pvalue_gate_in": pvalue_gate_in,
            "pvalue_gate_out": pvalue_gate_out,
            "emergency_drop_p": emergency_drop_p,
        }
    }


# =========================
# CLI entry
# =========================

if __name__ == "__main__":
    # Example:
    # python -m Models.stepwise_psd_models <path/to/data.xlsx> [mode] [criterion]
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) >= 3 else "both"
        crit = sys.argv[3] if len(sys.argv) >= 4 else "bic"
        fit_stepwise_models(
            path,
            mode=mode,
            criterion=crit,
            # sensible defaults for your small-n setting:
            max_features=5,
            pvalue_gate_in=0.15,
            pvalue_gate_out=0.10,
            emergency_drop_p=0.20,
            verbose=True,
        )
    else:
        print("Usage:\n  python -m Models.stepwise_psd_models <path/to/data.xlsx> [mode] [criterion]\n"
              "Examples:\n  python -m Models.stepwise_psd_models mydata.xlsx backward bic\n"
              "  python -m Models.stepwise_psd_models mydata.xlsx forward aic")
