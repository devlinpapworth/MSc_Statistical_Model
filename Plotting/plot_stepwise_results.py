import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_actual_vs_pred(df, actual_col, pred_col, title, ax, x_label, y_label):
    """Predicted vs measured scatter with 1:1 red line and R^2 annotation."""
    ax.scatter(df[actual_col], df[pred_col], s=60, edgecolor="k", alpha=0.85)

    # --- Fixed axes 0 to 0.5 ---
    lo, hi = 0.0, 0.5
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, color="r", label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    r2 = np.corrcoef(df[actual_col], df[pred_col])[0, 1] ** 2
    ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.grid(alpha=0.3)
    ax.legend()


def _plot_residuals(df, actual_col, pred_col, title, ax, x_label):
    """Residuals vs measured scatter with red zero line."""
    resid = df[actual_col] - df[pred_col]
    ax.scatter(df[actual_col], resid, s=50, edgecolor="k", alpha=0.85)
    ax.axhline(0.0, linestyle="--", linewidth=2, color="r")

    # --- Fixed x-axis from 0 to 0.5 ---
    ax.set_xlim(0.0, 0.5)

    ax.set_title(f"{title} Residuals")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Residual (Actual - Pred)")
    ax.grid(alpha=0.3)


def plots_main(csv_path="stepwise_predictions.csv", save_path=None):
    """
    Reads predictions CSV and shows:
      - Predicted vs Measured scatter (with 1:1 red line)
      - Residuals vs Measured
    for:
      - Final Water Content (%)
      - Cake Porosity
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Scatter (Predicted vs Measured) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
    _plot_actual_vs_pred(
        df,
        "Mc_%_actual",
        "Mc_%_pred",
        "Final Water Content",
        axes1[0],
        x_label="Measured Final Water Content (%)",
        y_label="Predicted Final Water Content (%)",
    )
    _plot_actual_vs_pred(
        df,
        "Cake_por_actual",
        "Cake_por_pred",
        "Cake Porosity",
        axes1[1],
        x_label="Measured Cake Porosity",
        y_label="Predicted Cake Porosity",
    )
    fig1.tight_layout()

    # --- Residuals ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    _plot_residuals(
        df,
        "Mc_%_actual",
        "Mc_%_pred",
        "Final Water Content",
        axes2[0],
        x_label="Measured Final Water Content (%)",
    )
    _plot_residuals(
        df,
        "Cake_por_actual",
        "Cake_por_pred",
        "Cake Porosity",
        axes2[1],
        x_label="Measured Cake Porosity",
    )
    fig2.tight_layout()

    # --- Optional save ---
    if save_path:
        base, ext = os.path.splitext(save_path)
        fig1.savefig(f"{base}_scatter{ext or '.png'}", dpi=300, bbox_inches="tight")
        fig2.savefig(f"{base}_residuals{ext or '.png'}", dpi=300, bbox_inches="tight")

    plt.show()
