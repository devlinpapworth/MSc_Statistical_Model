# Base/export_formula.py
import math
import warnings
import pandas as pd
import numpy as np


def _require(keys, dct, ctx="results"):
    missing = [k for k in keys if k not in dct]
    if missing:
        raise KeyError(f"Missing keys in {ctx}: {missing}")


def _collect_transform_pieces(results):
    
    _require(["scaler_raw", "scaler_transformed", "orthog_map", "raw_cols", "ratio_cols"], results)

    scaler1 = results["scaler_raw"]
    scaler2 = results["scaler_transformed"]
    orth = results["orthog_map"]
    raw_cols = list(results["raw_cols"])
    ratio_cols = list(results["ratio_cols"])

    # All features (order not critical here; we’ll index by name)
    all_feats = list(set(raw_cols + ratio_cols))

    # First scaler (on raw feature space)
    m1 = pd.Series(scaler1.mean_, index=scaler1.feature_names_in_)
    s1 = pd.Series(scaler1.scale_, index=scaler1.feature_names_in_)

    # Second scaler (after orthogonalization)
    m2 = pd.Series(scaler2.mean_, index=scaler2.feature_names_in_)
    s2 = pd.Series(scaler2.scale_, index=scaler2.feature_names_in_)

    # Ensure we have entries for all features we might touch
    for name in all_feats:
        if name not in m1.index or name not in s1.index:
            raise KeyError(f"Feature '{name}' missing from scaler_raw stats.")
        if name not in m2.index or name not in s2.index:
            # This can happen if a feature was dropped before scaler2.
            # Safer to fill zeros/ones to avoid crashes, but warn.
            warnings.warn(f"Feature '{name}' missing from scaler_transformed stats; using 0 mean / 1 scale.")
            m2.loc[name] = 0.0
            s2.loc[name] = 1.0

    return dict(m1=m1, s1=s1, m2=m2, s2=s2,
                orthog_map=orth, raw_cols=raw_cols, ratio_cols=ratio_cols)


def _to_raw_formula_from_transformed(model, selected, tr):
    
    params = model.params.copy()
    c = float(params["const"])
    betas = {f: float(params[f]) for f in selected if f in params.index}

    m1, s1, m2, s2 = tr["m1"], tr["s1"], tr["m2"], tr["s2"]
    raw_cols = tr["raw_cols"]
    ratio_cols = tr["ratio_cols"]
    orth = tr["orthog_map"]

    # Work first in z-space (after scaler1), then convert to x (raw)
    coef_z = {f: 0.0 for f in (raw_cols + ratio_cols)}
    intercept_z = c

    # Contribution from base (raw) features:
    for b in raw_cols:
        if b in betas:
            intercept_z += - betas[b] * (m2[b] / s2[b])
            coef_z[b] += betas[b] / s2[b]

    # Contribution from ratio features (remember they were residualized)
    for r in ratio_cols:
        if r not in betas:
            continue
        beta_r = betas[r]
        # u_r = z_r - (a_r + B_r z_b)
        a_r = orth.get(r, {}).get("intercept", 0.0)
        B_r = orth.get(r, {}).get("coef", {})  # dict base->coef
        intercept_z += - beta_r * ((a_r + m2[r]) / s2[r])
        # coefficient on original z_r
        coef_z[r] += beta_r / s2[r]
        # subtract the projections onto base z_b
        for b, B_rb in B_r.items():
            coef_z[b] += - beta_r * (B_rb / s2[r])

    # Now convert z back to raw x: z_j = (x_j - m1_j)/s1_j
    coef_x = {}
    for f, g in coef_z.items():
        if f not in s1.index:
            # Shouldn't happen if scalers are aligned; skip defensively.
            continue
        coef_x[f] = g / s1[f]

    intercept_x = intercept_z
    for f, g in coef_z.items():
        if f in s1.index:
            intercept_x += - g * (m1[f] / s1[f])

    return float(intercept_x), {k: float(v) for k, v in coef_x.items() if k in selected}


def _export_raw_formula_single(results, which_key):
    
    _require([which_key], results, ctx="results")
    block = results[which_key]
    _require(["model", "selected_features"], block, ctx=f"results['{which_key}']")

    # Transform stacks and specs
    tr = _collect_transform_pieces(results)

    model = block["model"]
    feats_selected = block["selected_features"]
    b0, coefs = _to_raw_formula_from_transformed(model, feats_selected, tr)
    return b0, coefs


def print_formulas(results):
    """
    Pretty-print raw-input formulas for moisture and porosity.
    Falls back to transformed-space printing if required pieces are missing.
    """
    try:
        b0_m, b_m = _export_raw_formula_single(results, "res_moisture")
        print("=== Final Moisture (raw-input) formula ===")
        rhs = " + ".join([f"{b_m[k]:.6f}*{k}" for k in b_m])
        print(f"Mc_%  = {b0_m:.6f}" + ((" + " + rhs) if rhs else ""))

        b0_p, b_p = _export_raw_formula_single(results, "res_porosity")
        print("\n=== Cake Porosity (raw-input) formula ===")
        rhs = " + ".join([f"{b_p[k]:.6f}*{k}" for k in b_p])
        print(f"Cake_por = {b0_p:.6f}" + ((" + " + rhs) if rhs else ""))

    except KeyError as e:
        warnings.warn(
            f"{e}. Falling back to transformed-space coefficients "
            "(effects per +1 SD of transformed features)."
        )
        for which, label in [("res_moisture", "Final Moisture"),
                             ("res_porosity", "Cake Porosity")]:
            block = results[which]
            model = block["model"]
            feats = block["selected_features"]
            params = model.params
            print(f"\n=== {label} (transformed-space) ===")
            print(f"{label} = {params['const']:.6f} + " +
                  " + ".join([f"{params[f]:.6f}*{f}" for f in feats]))
