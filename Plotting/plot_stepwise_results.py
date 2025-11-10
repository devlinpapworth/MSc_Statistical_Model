# Plotting/prediction_plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _plot_actual_vs_pred(df, actual_col, pred_col, title, ax):
    ax.scatter(df[actual_col], df[pred_col], s=60, edgecolor="k", alpha=0.85)
    lo = min(df[actual_col].min(), df[pred_col].min())
    hi = max(df[actual_col].max(), df[pred_col].max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="1:1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(f"Actual {actual_col}")
    ax.set_ylabel(f"Predicted {pred_col}")
    r2 = np.corrcoef(df[actual_col], df[pred_col])[0, 1] ** 2
    ax.text(0.04, 0.92, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.grid(alpha=0.3); ax.legend()

def _plot_residuals(df, actual_col, pred_col, title, ax):
    resid = df[actual_col] - df[pred_col]
    ax.scatter(df[actual_col], resid, s=50, edgecolor="k", alpha=0.85)
    ax.axhline(0.0, linestyle="--", linewidth=2)
    ax.set_title(f"{title} residuals")
    ax.set_xlabel(f"Actual {actual_col}")
    ax.set_ylabel("Residual (Actual - Pred)")
    ax.grid(alpha=0.3)

def plots_main(csv_path="stepwise_predictions.csv", save_path=None):
    """
    Reads predictions CSV and shows 1:1 + residual plots for:
      - Mc_% (Final moisture)
      - Cake_por (Cake porosity)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)

    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
    _plot_actual_vs_pred(df, "Mc_%_actual", "Mc_%_pred", "Final Moisture (Mc%)", axes1[0])
    _plot_actual_vs_pred(df, "Cake_por_actual", "Cake_por_pred", "Cake Porosity", axes1[1])
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    _plot_residuals(df, "Mc_%_actual", "Mc_%_pred", "Moisture", axes2[0])
    _plot_residuals(df, "Cake_por_actual", "Cake_por_pred", "Porosity", axes2[1])
    fig2.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        fig1.savefig(f"{base}_scatter{ext or '.png'}", dpi=300, bbox_inches="tight")
        fig2.savefig(f"{base}_residuals{ext or '.png'}", dpi=300, bbox_inches="tight")

    plt.show()
