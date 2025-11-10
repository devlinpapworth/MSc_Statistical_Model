
# Base/export_formula.py
import pandas as pd

def export_raw_formula(results, which):
    """
    which: 'res_moisture' or 'res_porosity'
    Returns (intercept, coef_dict) with coefficients for RAW features.
    """
    model = results[which]["model"]
    feats = results[which]["selected_features"]
    scaler = results["scaler"]
    all_feats = results["features_used"]

    means = pd.Series(scaler.mean_, index=all_feats)
    scales = pd.Series(scaler.scale_, index=all_feats)
    params = model.params.copy()

    raw_coefs = {}
    raw_intercept = float(params["const"])
    for f in feats:
        beta_std = float(params[f])
        raw_coefs[f] = beta_std / scales[f]
        raw_intercept -= beta_std * means[f] / scales[f]

    return raw_intercept, raw_coefs


def print_formulas(results):
    b0_m, b_m = export_raw_formula(results, "res_moisture")
    print("=== Final Moisture (raw-input) formula ===")
    print("Mc_%  = {:.6f} + ".format(b0_m) +
          " + ".join([f"{b_m[k]:.6f}*{k}" for k in b_m]))

    b0_p, b_p = export_raw_formula(results, "res_porosity")
    print("\n=== Cake Porosity (raw-input) formula ===")
    print("Cake_por = {:.6f} + ".format(b0_p) +
          " + ".join([f"{b_p[k]:.6f}*{k}" for k in b_p]))
