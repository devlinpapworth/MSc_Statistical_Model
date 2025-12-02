#!/usr/bin/env python3


import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
DB_SHEET = "DB_2"
PSD_SHEET = "PSD"

TARGET_COL = "Mc_%"
SAMPLE_CODE_COL = "Sample Code"
FLAG_COL = "flag"
TEST_PROC_COL = "Test_procedure"

# PSD cols we need from PSD sheet
PSD_BASE_COLS = ["D10", "D20", "D50", "D80", "D90"]


# ---------------------------------------------------------------------
# Helpers: load / merge / engineer columns
# ---------------------------------------------------------------------

def error(msg: str):
    raise SystemExit(f"\nERROR: {msg}\n")


def load_sheets(xlsx_path: str, sheet_db: str = DB_SHEET, sheet_psd: str = PSD_SHEET):
    try:
        xls = pd.ExcelFile(xlsx_path)
    except FileNotFoundError:
        error(f"Excel file not found: {xlsx_path}")
    except Exception as e:
        error(f"Failed to open Excel file '{xlsx_path}': {e}")

    if sheet_db not in xls.sheet_names:
        error(f"DB sheet '{sheet_db}' not found. Sheets: {xls.sheet_names}")
    if sheet_psd not in xls.sheet_names:
        error(f"PSD sheet '{sheet_psd}' not found. Sheets: {xls.sheet_names}")

    try:
        df_db = pd.read_excel(xls, sheet_name=sheet_db)
        df_psd = pd.read_excel(xls, sheet_name=sheet_psd)
    except Exception as e:
        error(f"Failed reading sheets: {e}")

    return df_db, df_psd


def apply_flag_filter(df_db: pd.DataFrame) -> pd.DataFrame:
    if FLAG_COL not in df_db.columns:
        error(f"Flag column '{FLAG_COL}' missing in DB sheet.")

    mask = df_db[FLAG_COL].astype(str).str.contains("include", case=False, na=False)
    df = df_db.loc[mask].copy()
    if df.empty:
        error("No rows remain after flag filter ('include').")
    return df


def merge_db_psd_empirical(
    df_db: pd.DataFrame,
    df_psd: pd.DataFrame,
) -> pd.DataFrame:
    """Merge DB_2 and PSD and compute only what we need for empirical plots."""
    # basic checks
    needed_db = {SAMPLE_CODE_COL, TARGET_COL, FLAG_COL, TEST_PROC_COL}
    if not needed_db.issubset(df_db.columns):
        error(f"DB sheet missing required columns: {needed_db - set(df_db.columns)}")

    if SAMPLE_CODE_COL not in df_psd.columns:
        error(f"PSD sheet missing '{SAMPLE_CODE_COL}' column.")
    for c in PSD_BASE_COLS:
        if c not in df_psd.columns:
            error(f"PSD sheet missing required PSD column '{c}'.")

    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner")
    if df.empty:
        error("Merge between DB and PSD sheets is empty. Check sample codes.")

    # convert base D-values to numeric
    for col in PSD_BASE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # span
    df["D80/D20"] = df["D80"] / df["D20"]

    # diaphragm flag from Test_procedure
    df["Diaphragm_on"] = df[TEST_PROC_COL].astype(str).str.contains(
        "STD", case=False, na=False
    ).astype(int)

    # keep only rows with all needed columns
    needed_cols = ["D10", "D80/D20", "Diaphragm_on", TARGET_COL]
    missing_mask = df[needed_cols].isna().any(axis=1)

    if missing_mask.any():
        print("\nWarning: dropped rows due to NaNs in empirical merge:")
        print(df.loc[missing_mask, [SAMPLE_CODE_COL] + needed_cols])
        print("--- end dropped rows ---\n")

    df_clean = df.loc[~missing_mask].copy()
    if df_clean.empty:
        error("All rows dropped due to missing values in empirical fields.")

    return df_clean


# ---------------------------------------------------------------------
# Plot 1: Interaction D10 × diaphragm_on (means)
# ---------------------------------------------------------------------

def plot_interaction_d10_diaphragm(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_bins: int = 3,
    title: str = "Interaction: D10 vs diaphragm (empirical means)",
    save_path: Optional[str] = None,
):
    required = {"D10", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        error(f"DataFrame must contain: {required}")

    data = df[["D10", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    d10_min, d10_max = data["D10"].min(), data["D10"].max()
    bins = np.linspace(d10_min, d10_max, n_bins + 1)
    data["D10_bin"] = pd.cut(data["D10"], bins=bins, include_lowest=True)

    grouped = (
        data
        .groupby(["D10_bin", "Diaphragm_on"])
        .agg(
            D10_mean=("D10", "mean"),
            Mc_mean=("Mc_pct", "mean"),
            n=("Mc_pct", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for dia_val, label in [(0, "No diaphragm"), (1, "Diaphragm on")]:
        subset = grouped[grouped["Diaphragm_on"] == dia_val]
        if subset.empty:
            continue
        ax.plot(
            subset["D10_mean"],
            subset["Mc_mean"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("Mean FMC (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Plot 2: Interaction Span × diaphragm_on (means)
# ---------------------------------------------------------------------

def plot_interaction_span_diaphragm(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_bins: int = 3,
    title: str = "Interaction: Span vs diaphragm (empirical means)",
    save_path: Optional[str] = None,
):
    required = {"D80/D20", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        error(f"DataFrame must contain: {required}")

    data = df[["D80/D20", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    span_min, span_max = data["D80/D20"].min(), data["D80/D20"].max()
    bins = np.linspace(span_min, span_max, n_bins + 1)
    data["Span_bin"] = pd.cut(data["D80/D20"], bins=bins, include_lowest=True)

    grouped = (
        data
        .groupby(["Span_bin", "Diaphragm_on"])
        .agg(
            Span_mean=("D80/D20", "mean"),
            Mc_mean=("Mc_pct", "mean"),
            n=("Mc_pct", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for dia_val, label in [(0, "No diaphragm"), (1, "Diaphragm on")]:
        subset = grouped[grouped["Diaphragm_on"] == dia_val]
        if subset.empty:
            continue
        ax.plot(
            subset["Span_mean"],
            subset["Mc_mean"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("Mean FMC (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Plot 3: Boxplots - FMC by D10 class & diaphragm
# ---------------------------------------------------------------------

def plot_boxplot_d10_classes(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_bins: int = 3,
    title: str = "FMC by D10 class and diaphragm state",
    save_path: Optional[str] = None,
):
    required = {"D10", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        error(f"DataFrame must contain: {required}")

    data = df[["D10", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    d10_min, d10_max = data["D10"].min(), data["D10"].max()
    bins = np.linspace(d10_min, d10_max, n_bins + 1)
    labels = [f"{b1:.1f}-{b2:.1f}" for b1, b2 in zip(bins[:-1], bins[1:])]
    data["D10_class"] = pd.cut(data["D10"], bins=bins, labels=labels, include_lowest=True)

    box_data = []
    box_labels = []

    for cls in labels:
        for dia_val, dia_label in [(0, "Off"), (1, "On")]:
            subset = data[(data["D10_class"] == cls) & (data["Diaphragm_on"] == dia_val)]
            if not subset.empty:
                box_data.append(subset["Mc_pct"])
                box_labels.append(f"{cls}\nDia {dia_label}")

    if not box_data:
        print("No data available for boxplot after classing; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(box_data, labels=box_labels, showfliers=True)
    ax.set_ylabel("FMC (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Plot 4: Scatter D10 vs Span coloured by FMC
# ---------------------------------------------------------------------

def plot_d10_span_scatter(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    title: str = "D10 vs Span coloured by FMC (empirical)",
    save_path: Optional[str] = None,
):
    required = {"D10", "D80/D20", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        error(f"DataFrame must contain: {required}")

    data = df[["D10", "D80/D20", "Diaphragm_on", target_col]].dropna().copy()
    data["Mc_pct"] = 100 * data[target_col]

    fig, ax = plt.subplots(figsize=(6, 5))

    for dia_val, marker, label in [(0, "o", "No diaphragm"), (1, "s", "Diaphragm on")]:
        subset = data[data["Diaphragm_on"] == dia_val]
        if subset.empty:
            continue
        sc = ax.scatter(
            subset["D10"],
            subset["D80/D20"],
            c=subset["Mc_pct"],
            marker=marker,
            edgecolors="k",
            cmap="viridis",
            alpha=0.85,
            label=label,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("FMC (%)")

    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("Span (D80/D20)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# Plot 4b: Scatter D10 vs Span coloured by FMC – all points
# ---------------------------------------------------------------------
import numpy as np

def plot_allpoints_d10_vs_span(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    title: str = "D10 vs Span coloured by FMC (all data)",
    save_path: Optional[str] = None,
):
    """
    Plots ALL experimental datapoints:
        - X = D10 (with jitter)
        - Y = Span (D80/D20) (with jitter)
        - Colour = FMC (%)
        - Marker shape = diaphragm on/off
    """

    required = {"D10", "D80/D20", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataframe must contain columns: {required}")

    df_plot = df.copy()
    df_plot["Mc_pct"] = 100 * df_plot[target_col]

    # -------------------------
    # Add small jitter so that
    # repeated measurements (same D10/span) become visible
    # -------------------------
    jitter_scale_x = 0.15     # microns of jitter for D10
    jitter_scale_y = 0.05     # jitter for Span

    df_plot["D10_j"] = df_plot["D10"] + np.random.normal(0, jitter_scale_x, len(df_plot))
    df_plot["Span_j"] = df_plot["D80/D20"] + np.random.normal(0, jitter_scale_y, len(df_plot))

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- diaphragm OFF ---
    df0 = df_plot[df_plot["Diaphragm_on"] == 0]
    sc0 = ax.scatter(
        df0["D10_j"],
        df0["Span_j"],
        c=df0["Mc_pct"],
        cmap="viridis",
        marker="o",
        edgecolors="k",
        s=70,
        label="No diaphragm",
    )

    # --- diaphragm ON ---
    df1 = df_plot[df_plot["Diaphragm_on"] == 1]
    sc1 = ax.scatter(
        df1["D10_j"],
        df1["Span_j"],
        c=df1["Mc_pct"],
        cmap="viridis",
        marker="s",
        edgecolors="k",
        s=70,
        label="Diaphragm on",
    )

    # colourbar
    cbar = plt.colorbar(sc1, ax=ax)
    cbar.set_label("FMC (%)")

    ax.set_xlabel("D10 (\u00B5m)")
    ax.set_ylabel("Span (D80/D20)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_allpoints_d10_vs_span_separated(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    title_base: str = "D10 vs Span coloured by FMC",
    save_path_base: Optional[str] = None,
):
    """
    Creates TWO separate scatter plots:
        1) Only diaphragm OFF
        2) Only diaphragm ON
    Each point represents one experimental row.

    - X = D10 (with jitter)
    - Y = Span (D80/D20) (with jitter)
    - Colour = FMC (%)
    """

    required = {"D10", "D80/D20", "Diaphragm_on", target_col}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    df_plot = df.copy()
    df_plot["Mc_pct"] = 100 * df_plot[target_col]

    # --- Jitter so duplicates don't overlap ---
    jitter_x = 0.15
    jitter_y = 0.05
    df_plot["D10_j"] = df_plot["D10"] + np.random.normal(0, jitter_x, len(df_plot))
    df_plot["Span_j"] = df_plot["D80/D20"] + np.random.normal(0, jitter_y, len(df_plot))

    # --- Split ---
    df0 = df_plot[df_plot["Diaphragm_on"] == 0]   # Off
    df1 = df_plot[df_plot["Diaphragm_on"] == 1]   # On

    # Same axis limits for visual comparison
    d10_min, d10_max = df_plot["D10"].min(), df_plot["D10"].max()
    span_min, span_max = df_plot["D80/D20"].min(), df_plot["D80/D20"].max()

    # -----------------------------
    # FIGURE 1 — NO DIAPHRAGM
    # -----------------------------
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    sc0 = ax0.scatter(
        df0["D10_j"],
        df0["Span_j"],
        c=df0["Mc_pct"],
        cmap="viridis",
        marker="o",
        edgecolors="k",
        s=70,
    )
    cbar0 = plt.colorbar(sc0, ax=ax0)
    cbar0.set_label("FMC (%)")

    ax0.set_xlabel("D10 (\u00B5m)")
    ax0.set_ylabel("Span (D80/D20)")
    ax0.set_title(f"{title_base} - No Diaphragm")
    ax0.set_xlim(d10_min - 1, d10_max + 1)
    ax0.set_ylim(span_min - 0.5, span_max + 0.5)
    ax0.grid(alpha=0.3)

    plt.tight_layout()
    if save_path_base:
        fig0.savefig(save_path_base + "_no_diaphragm.png", dpi=300, bbox_inches="tight")

    plt.show()

    # -----------------------------
    # FIGURE 2 — DIAPHRAGM ON
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sc1 = ax1.scatter(
        df1["D10_j"],
        df1["Span_j"],
        c=df1["Mc_pct"],
        cmap="viridis",
        marker="s",
        edgecolors="k",
        s=70,
    )
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("FMC (%)")

    ax1.set_xlabel("D10 (\u00B5m)")
    ax1.set_ylabel("Span (D80/D20)")
    ax1.set_title(f"{title_base} - Diaphragm On")
    ax1.set_xlim(d10_min - 1, d10_max + 1)
    ax1.set_ylim(span_min - 0.5, span_max + 0.5)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    if save_path_base:
        fig1.savefig(save_path_base + "_diaphragm_on.png", dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Empirical FMC figures vs D10, Span and diaphragm (no models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python empirical_figs.py
              python empirical_figs.py path_to_file.xlsx
            """
        ),
    )
    parser.add_argument(
        "xlsx_path",
        nargs="?",
        default=DEFAULT_XLSX_PATH,
        help=f"Path to Excel workbook (default: {DEFAULT_XLSX_PATH})",
    )
    parser.add_argument("--sheet_db", default=DB_SHEET)
    parser.add_argument("--sheet_psd", default=PSD_SHEET)

    args = parser.parse_args()

    print(f"\nUsing Excel file:\n  {args.xlsx_path}\n")

    df_db_raw, df_psd = load_sheets(args.xlsx_path, args.sheet_db, args.sheet_psd)
    df_db_inc = apply_flag_filter(df_db_raw)

    df = merge_db_psd_empirical(df_db_inc, df_psd)
    print(f"Usable empirical rows after merge & filtering: {len(df)}\n")

    # ---- Make figures (all using the same df) ----
    plot_interaction_d10_diaphragm(df)
    plot_interaction_span_diaphragm(df)
    plot_boxplot_d10_classes(df)
    plot_d10_span_scatter(df)
    plot_allpoints_d10_vs_span(
        df,
        target_col=TARGET_COL,
        title="D10 vs Span coloured by FMC (all data)",
        save_path=None,  # or r"C:\path\to\save\d10_span_allpoints.png"
    )
    plot_allpoints_d10_vs_span_separated(
        df,
        target_col=TARGET_COL,
        title_base="D10 vs Span coloured by FMC",
        save_path_base=None,   # Or provide base filename
)


if __name__ == "__main__":
    main()

