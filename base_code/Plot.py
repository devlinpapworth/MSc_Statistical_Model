#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import argparse
import textwrap

# ---------------------------------------------------------------------
# Marker map: sample identity
# ---------------------------------------------------------------------
SAMPLE_DISPLAY_ORDER = [
    "Si_5_2",
    "Si_5_5",
    "Si_Rep_new",
    "Si_BM",
    "Si_25_2",
    "Si_25_5",
    "Si_45_2",
    "Si_45_5",
]

SAMPLE_DISPLAY_NAME = {
    "Si_5_2": "Fine narrow",
    "Si_5_5": "Fine middle",
    "Si_Rep_new": "Fine wide",
    "Si_BM": "Fine Bi-Modal",
    "Si_25_2": "Medium narrow",
    "Si_25_5": "Medium middle",
    "Si_45_2": "Coarse narrow",
    "Si_45_5": "Coarse middle",
}

SAMPLE_MARKER_MAP = {
    "Si_5_2": "o",
    "Si_5_5": "s",
    "Si_25_2": "^",
    "Si_25_5": "v",
    "Si_45_2": "D",
    "Si_45_5": "P",
    "Si_Rep_new": "X",
    "Si_BM": "*",
}

DEFAULT_MARKER = "o"
SAMPLE_CODE_COL = "Sample Code"

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_XLSX_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
DB_SHEET = "DB_2"
PSD_SHEET = "PSD"

TARGET_COL = "Mc_%"
FLAG_COL = "flag"
TEST_PROC_COL = "Test_procedure"

PSD_BASE_COLS = ["D10", "D20", "D50", "D80", "D90"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def error(msg: str):
    raise SystemExit(f"\nERROR: {msg}\n")


def _get_numeric_1d(col):
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")


def load_sheets(xlsx_path, sheet_db=DB_SHEET, sheet_psd=PSD_SHEET):
    xls = pd.ExcelFile(xlsx_path)
    return (
        pd.read_excel(xls, sheet_name=sheet_db),
        pd.read_excel(xls, sheet_name=sheet_psd),
    )


def apply_flag_filter(df):
    mask = df[FLAG_COL].astype(str).str.contains("include", case=False, na=False)
    return df.loc[mask].copy()


def merge_db_psd_empirical(df_db, df_psd):
    df = pd.merge(df_db, df_psd, on=SAMPLE_CODE_COL, how="inner")

    for c in PSD_BASE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["D80/D20"] = df["D80"] / df["D20"]
    df["Diaphragm_on"] = (
        df[TEST_PROC_COL].astype(str)
        .str.contains("STD", case=False, na=False)
        .astype(int)
    )

    needed = ["D10", "D80/D20", "Diaphragm_on", TARGET_COL]
    return df.dropna(subset=needed).copy()

# ---------------------------------------------------------------------
# Core plotting helpers
# ---------------------------------------------------------------------

def scatter_bw(ax, x, y, sample, dia):
    ax.scatter(
        x,
        y,
        marker=SAMPLE_MARKER_MAP.get(sample, DEFAULT_MARKER),
        s=80,
        facecolors="black" if dia == 1 else "none",
        edgecolors="black",
        linewidths=1.3,
        zorder=3,
    )


def errorbar_range_bw(ax, x_mean, x_min, x_max, y_mean, y_min, y_max):
    # distances from mean (must be >= 0)
    xerr_low  = max(0.0, x_mean - x_min)
    xerr_high = max(0.0, x_max - x_mean)
    yerr_low  = max(0.0, y_mean - y_min)
    yerr_high = max(0.0, y_max - y_mean)

    ax.errorbar(
        x_mean, y_mean,
        xerr=[[xerr_low], [xerr_high]],
        yerr=[[yerr_low], [yerr_high]],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=3,
        capthick=1.1,
        zorder=1,
    )


# ---------------------------------------------------------------------
# Legend helper
# ---------------------------------------------------------------------

def add_bw_legend(ax, samples_in_plot):
    ordered_samples = [s for s in SAMPLE_DISPLAY_ORDER if s in samples_in_plot]

    sample_handles = []
    for s in ordered_samples:
        sample_handles.append(
            Line2D(
                [0], [0],
                marker=SAMPLE_MARKER_MAP.get(s, DEFAULT_MARKER),
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.3,
                markersize=8,
                label=SAMPLE_DISPLAY_NAME.get(s, s),
            )
        )

    dia_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="black", markeredgecolor="black",
               markersize=8, label="Diaphragm on"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="black",
               markersize=8, label="Diaphragm off"),
    ]

    ax.legend(
        handles=sample_handles + dia_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        title="Key",
    )

# ---------------------------------------------------------------------
# Air-blow time vs D10
# ---------------------------------------------------------------------

def plot_AT_vs_D10(df, title="Air-blow time vs P10", save_path=None):

    stats = (
        df.groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
          .agg(
              D10_mean=("D10", "mean"),
              D10_min=("D10", "min"),
              D10_max=("D10", "max"),
              AT_mean=("A_T", "mean"),
              AT_min=("A_T", "min"),
              AT_max=("A_T", "max"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        errorbar_range_bw(
            ax,
            r["D10_mean"], r["D10_min"], r["D10_max"],
            r["AT_mean"], r["AT_min"], r["AT_max"],
        )
        scatter_bw(ax, r["D10_mean"], r["AT_mean"],
                   r[SAMPLE_CODE_COL], r["Diaphragm_on"])

    ax.set_xlabel("P10 (\u00B5m)")
    ax.set_ylabel("Air-blow time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# Air-blow time vs Span
# ---------------------------------------------------------------------

def plot_AT_vs_Span(df, title="Air-blow time vs Span", save_path=None):

    stats = (
        df.groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
          .agg(
              Span_mean=("D80/D20", "mean"),
              Span_min=("D80/D20", "min"),
              Span_max=("D80/D20", "max"),
              AT_mean=("A_T", "mean"),
              AT_min=("A_T", "min"),
              AT_max=("A_T", "max"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        errorbar_range_bw(
            ax,
            r["Span_mean"], r["Span_min"], r["Span_max"],
            r["AT_mean"], r["AT_min"], r["AT_max"],
        )
        scatter_bw(ax, r["Span_mean"], r["AT_mean"],
                   r[SAMPLE_CODE_COL], r["Diaphragm_on"])

    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("Air-blow time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# FMC vs D10
# ---------------------------------------------------------------------

def plot_FMC_vs_D10(df, title="Final moisture vs P10", save_path=None):

    stats = (
        df.groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
          .agg(
              D10_mean=("D10", "mean"),
              D10_min=("D10", "min"),
              D10_max=("D10", "max"),
              FMC_mean=(TARGET_COL, "mean"),
              FMC_min=(TARGET_COL, "min"),
              FMC_max=(TARGET_COL, "max"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        y_mean = 100 * r["FMC_mean"]
        y_min  = 100 * r["FMC_min"]
        y_max  = 100 * r["FMC_max"]

        errorbar_range_bw(
            ax,
            r["D10_mean"], r["D10_min"], r["D10_max"],
            y_mean, y_min, y_max,
        )
        scatter_bw(ax, r["D10_mean"], y_mean,
                   r[SAMPLE_CODE_COL], r["Diaphragm_on"])

    ax.set_xlabel("P10 (\u00B5m)")
    ax.set_ylabel("Final moisture content (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------
# FMC vs Span
# ---------------------------------------------------------------------

def plot_FMC_vs_Span(df, title="Final moisture vs Span", save_path=None):

    stats = (
        df.groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
          .agg(
              Span_mean=("D80/D20", "mean"),
              Span_min=("D80/D20", "min"),
              Span_max=("D80/D20", "max"),
              FMC_mean=(TARGET_COL, "mean"),
              FMC_min=(TARGET_COL, "min"),
              FMC_max=(TARGET_COL, "max"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        y_mean = 100 * r["FMC_mean"]
        y_min  = 100 * r["FMC_min"]
        y_max  = 100 * r["FMC_max"]

        errorbar_range_bw(
            ax,
            r["Span_mean"], r["Span_min"], r["Span_max"],
            y_mean, y_min, y_max,
        )
        scatter_bw(ax, r["Span_mean"], y_mean,
                   r[SAMPLE_CODE_COL], r["Diaphragm_on"])

    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("Final moisture content (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def plot_PT_vs_D10(df: pd.DataFrame,
                   title: str = "Pumping time vs D10",
                   save_path: Optional[str] = None):
    """
    Pumping time F_T vs D10.
    Same styling as air-blow time plots:
      - marker shape = sample (PSD class)
      - filled black = diaphragm ON
      - hollow = diaphragm OFF
      - no colours, no connector lines, no trend lines
    """
    required = {"D10", "F_T", SAMPLE_CODE_COL, "Diaphragm_on"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required}")

    df_plot = df[list(required)].copy()

    df_plot["D10"] = pd.to_numeric(df_plot["D10"], errors="coerce")
    df_plot["F_T"] = _get_numeric_1d(df_plot["F_T"])
    df_plot = df_plot.dropna(subset=["D10", "F_T"])

    stats = (
        df_plot
        .groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
        .agg(D10_mean=("D10", "mean"),
             PT_mean=("F_T", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        scatter_bw(ax,
                   r["D10_mean"],
                   r["PT_mean"],
                   r[SAMPLE_CODE_COL],
                   r["Diaphragm_on"])

    ax.set_xlabel("D10 (\u03bcm)")
    ax.set_ylabel("Pumping time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 190)


    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_PT_vs_Span(df: pd.DataFrame,
                    title: str = "Pumping time vs Span",
                    save_path: Optional[str] = None):
    """
    Pumping time F_T vs Span (D80/D20).
    Same styling as air-blow time plots:
      - marker shape = sample (PSD class)
      - filled black = diaphragm ON
      - hollow = diaphragm OFF
      - no colours, no connector lines, no trend lines
    """
    required = {"D80/D20", "F_T", SAMPLE_CODE_COL, "Diaphragm_on"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required}")

    df_plot = df[list(required)].copy()

    df_plot["D80/D20"] = pd.to_numeric(df_plot["D80/D20"], errors="coerce")
    df_plot["F_T"] = _get_numeric_1d(df_plot["F_T"])
    df_plot = df_plot.dropna(subset=["D80/D20", "F_T"])

    stats = (
        df_plot
        .groupby([SAMPLE_CODE_COL, "Diaphragm_on"])
        .agg(Span_mean=("D80/D20", "mean"),
             PT_mean=("F_T", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    for _, r in stats.iterrows():
        scatter_bw(ax,
                   r["Span_mean"],
                   r["PT_mean"],
                   r[SAMPLE_CODE_COL],
                   r["Diaphragm_on"])

    ax.set_xlabel("Span (D80/D20)")
    ax.set_ylabel("Pumping time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 190)

    add_bw_legend(ax, stats[SAMPLE_CODE_COL].unique())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def main():

    parser = argparse.ArgumentParser(
        description="Empirical filtration figures (monochrome, no trends)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python empirical_figs.py
            """
        ),
    )
    parser.add_argument("xlsx_path", nargs="?", default=DEFAULT_XLSX_PATH)
    args = parser.parse_args()

    df_db, df_psd = load_sheets(args.xlsx_path)
    df_db = apply_flag_filter(df_db)
    df = merge_db_psd_empirical(df_db, df_psd)

    #plot_PT_vs_Span(df)
    #plot_PT_vs_D10(df)

    plot_AT_vs_D10(df)
    plot_AT_vs_Span(df)
    plot_FMC_vs_D10(df)
    plot_FMC_vs_Span(df)


if __name__ == "__main__":
    main()
