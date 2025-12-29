import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from typing import Any, Optional

# ============================================================
# 1. CONFIG - EDIT THIS PART
# ============================================================

FILE_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
SHEET_NAME = "PSD_Full"

COL_SAMPLE = "Sample Name"  # column with sample codes/names

SAMPLE_CODES = [
    "Si_BM",
    "Si_Rep_new",
    "Si_45_2",
    "Si_45_5",
    "Si_25_5",
    "Si_25_2",
    "Si_5_2",
    "Si_5_5",
]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def _get_size_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify the columns that are the PSD size classes.
    Assumes all columns that can be converted to float are size columns.
    """
    size_cols: list[str] = []
    for col in df.columns:
        try:
            _ = float(str(col))
            size_cols.append(col)
        except ValueError:
            continue
    if not size_cols:
        raise ValueError("Could not detect any numeric size-class columns.")
    size_cols = sorted(size_cols, key=lambda c: float(c))
    return size_cols


def _discrete_pdf_from_row(row: pd.Series, size_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    From a PSD_Full row, return:
      d : diameters (\u00B5m)
      p : normalised probabilities (volume fraction per size class)
    """
    d = np.array([float(c) for c in size_cols], dtype=float)
    y = row[size_cols].astype(float).to_numpy()
    y = np.nan_to_num(y, nan=0.0)

    total = y.sum()
    if total <= 0:
        raise ValueError("PSD row has zero or negative total - cannot normalise.")

    p = y / total
    return d, p


def _fit_gmm_logspace(
    d: np.ndarray,
    p: np.ndarray,
    max_components: int = 3,
    min_effective_weight: float = 0.05,
) -> dict[str, Any]:
    """
    Fit Gaussian Mixture Model(s) in log10(d) space WITHOUT sample_weight
    (for older sklearn). We emulate weights by repeating each point according
    to its probability p.

    Also computes an 'effective' number of modes (K_eff) by ignoring very
    low-weight components, and Ashman D between the two dominant modes if
    K_eff >= 2.

    Returns keys:
      K           : BIC-selected component count
      means_log10 : all component means in log10(d)
      stds_log10  : all component std devs
      weights     : all component weights
      K_eff       : effective number of modes (after weight threshold)
      eff_means_log10 : means of effective modes
      eff_weights     : weights of effective modes
      ashman_D_main   : Ashman D between two dominant effective modes (log10 space), or NaN
    """
    x_base = np.log10(d).reshape(-1, 1)
    w = p / p.sum()

    # Expand x according to weights so total number of points is reasonable
    total_points = 2000
    counts = np.maximum(1, np.round(w * total_points).astype(int))
    x_expanded = np.repeat(x_base, counts, axis=0)

    best_bic = np.inf
    best_k = 1
    best_model: Optional[GaussianMixture] = None

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
        )
        gmm.fit(x_expanded)
        bic = gmm.bic(x_expanded)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm

    if best_model is None:
        raise RuntimeError("GMM fitting failed for all component counts.")

    means_log10 = best_model.means_.flatten()
    stds_log10 = np.sqrt(best_model.covariances_.flatten())
    weights = best_model.weights_

    # --- Effective modes: ignore tiny weights ---
    # Sort by weight descending
    idx_sort = np.argsort(weights)[::-1]
    means_sorted = means_log10[idx_sort]
    stds_sorted = stds_log10[idx_sort]
    weights_sorted = weights[idx_sort]

    mask_eff = weights_sorted >= min_effective_weight
    eff_means = means_sorted[mask_eff]
    eff_stds = stds_sorted[mask_eff]
    eff_weights = weights_sorted[mask_eff]

    K_eff = eff_means.size

    # --- Ashman D between two dominant effective modes (if at least 2) ---
    if K_eff >= 2:
        mu1, mu2 = eff_means[:2]
        s1, s2 = eff_stds[:2]
        ashman_D_main = np.abs(mu1 - mu2) / np.sqrt(s1 ** 2 + s2 ** 2)
    else:
        ashman_D_main = np.nan

    return {
        "K": best_k,
        "means_log10": means_log10,
        "stds_log10": stds_log10,
        "weights": weights,
        "K_eff": K_eff,
        "eff_means_log10": eff_means,
        "eff_weights": eff_weights,
        "ashman_D_main": ashman_D_main,
    }


def _lognormal_pdf_mixture(d: np.ndarray,
                           means_log10: np.ndarray,
                           stds_log10: np.ndarray,
                           weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the lognormal mixture PDF on a diameter grid d.
    We treat the GMM as Gaussian in log10(d), convert to ln(d), and
    use the lognormal PDF in d-space.
    """
    d = np.asarray(d, dtype=float)
    # Avoid zero or negative values
    d[d <= 0] = np.min(d[d > 0])

    # Convert log10 parameters to natural log parameters
    ln10 = np.log(10.0)
    mu_ln = means_log10 * ln10
    sigma_ln = stds_log10 * ln10

    pdf_components = []
    for mu, sigma, w in zip(mu_ln, sigma_ln, weights):
        # Lognormal pdf
        comp = (w / (d * sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            - (np.log(d) - mu) ** 2 / (2.0 * sigma ** 2)
        )
        pdf_components.append(comp)

    pdf_components = np.array(pdf_components)   # shape (K, len(d))
    mixture_pdf = pdf_components.sum(axis=0)    # sum over components

    return mixture_pdf, pdf_components


def _plot_psd_and_gmm(sample_name: str,
                      d: np.ndarray,
                      p: np.ndarray,
                      gmm_info: dict[str, Any]) -> None:
    """
    Plot discrete PSD and fitted GMM (mixture + components) for a sample.
    """
    # Grid for smooth curve
    d_grid = np.logspace(np.log10(d.min()), np.log10(d.max()), 400)

    mix_pdf, comp_pdfs = _lognormal_pdf_mixture(
        d_grid,
        gmm_info["means_log10"],
        gmm_info["stds_log10"],
        gmm_info["weights"],
    )

    # Scale mixture to roughly match the discrete PDF peak for visual comparison
    scale = p.max() / mix_pdf.max() if mix_pdf.max() > 0 else 1.0
    mix_pdf_scaled = mix_pdf * scale
    comp_pdfs_scaled = comp_pdfs * scale

    fig, ax = plt.subplots(figsize=(7, 4))

    # Discrete PSD points
    ax.scatter(d, p, color="k", s=25, label="Discrete PSD (per bin)")

    # Mixture curve
    ax.plot(d_grid, mix_pdf_scaled, label=f"GMM mixture (K={gmm_info['K']})")

    # Individual components
    for i, comp in enumerate(comp_pdfs_scaled):
        ax.plot(d_grid, comp, linestyle="--", alpha=0.7, label=f"Mode {i+1}")

    ax.set_xscale("log")
    ax.set_xlabel("Particle size d (\u00B5m)")
    ax.set_ylabel("Relative density (scaled)")
    ax.grid(True, which="both", alpha=0.3)

    title = (
        f"{sample_name} | K_raw={gmm_info['K']} "
        f"| K_eff={gmm_info.get('K_eff', gmm_info['K'])} "
        f"| Ashman D (log10)={gmm_info.get('ashman_D_main', float('nan')):.2f}"
    )
    ax.set_title(title)

    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show()



# ============================================================
# 3. MAIN DRIVER
# ============================================================

def run_gmm_psd_analysis(
    file_path: str = FILE_PATH,
    sheet_name: str = SHEET_NAME,
    sample_codes: list[str] = SAMPLE_CODES,
) -> pd.DataFrame:
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    size_cols = _get_size_columns(df)

    print("Detected size columns (first 10 shown):", size_cols[:10])
    print("Number of size columns:", len(size_cols))

    results: list[dict[str, Any]] = []

    for s in sample_codes:
        sub = df[df[COL_SAMPLE] == s]
        if sub.empty:
            print(f"[WARN] Sample '{s}' not found in {sheet_name}. Skipping.")
            continue

        row = sub.iloc[0]

        # Skip samples with zero PSD total
        total_psd = row[size_cols].astype(float).sum()
        if total_psd <= 0:
            print(f"[WARN] Sample '{s}' has zero/empty PSD data. Skipping.")
            continue

        # Build discrete PDF
        d, p = _discrete_pdf_from_row(row, size_cols)

        gmm_info = _fit_gmm_logspace(d, p, max_components=3)

        # Plot PSD + GMM (full mixture)
        _plot_psd_and_gmm(s, d, p, gmm_info)

        # Full-component stats
        means_um = 10 ** gmm_info["means_log10"]
        weights = gmm_info["weights"]

        # Effective-mode stats
        eff_means_um = 10 ** gmm_info["eff_means_log10"]
        eff_weights = gmm_info["eff_weights"]

        print(f"\nSample: {s}")
        print(f"  BIC-chosen K (raw)       = {gmm_info['K']}")
        print(f"  Raw centres (\u00B5m)         = {np.round(means_um, 3)}")
        print(f"  Raw weights (?_k)        = {np.round(weights, 3)}")
        print(f"  Effective K (K_eff)      = {gmm_info['K_eff']}")
        print(f"  Effective centres (\u00B5m)   = {np.round(eff_means_um, 3)}")
        print(f"  Effective weights (?_k)  = {np.round(eff_weights, 3)}")
        print(f"  Ashman D (dominant modes, log10) = {gmm_info['ashman_D_main']:.3f}")

        results.append({
            "sample": s,
            "K_raw": gmm_info["K"],
            "K_eff": gmm_info["K_eff"],
            "means_um_raw": means_um,
            "weights_raw": weights,
            "means_um_eff": eff_means_um,
            "weights_eff": eff_weights,
            "ashman_D_log10_main": gmm_info["ashman_D_main"],
        })



    if not results:
        raise RuntimeError("No results produced - check sample codes and data.")

    return pd.DataFrame(results)


if __name__ == "__main__":
    metrics_df = run_gmm_psd_analysis()
    print("\n=== Summary metrics ===")
    print(metrics_df)
