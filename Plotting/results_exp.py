from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# USER SETTINGS
# -----------------------------
FILEPATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"
DB_SHEET = "DB"
PSD_SHEET = "PSD"

OUTDIR = "custom_plots"
DPI = 300

KEY = "Sample Code"
FMC = "Mc_%"
CLOTH = "Cloth"
FAIL_COL = "flag"
TEST_PROC = "Test_procedure"

P_PRESS = "F_P"
P_TIME  = "F_T"
P10 = "D10"

# If your fail column is numeric or uses different tokens, update this:
FAIL_REGEX = r"(?:fail|failed|no\s*cake|cake\s*fail|break|crack)"
PUMP_PLOT_CODE = "Pump_plot_code"


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def savefig(fig, path: str) -> None:
    fig.tight_layout()
    plt.show()


def classify_sample_type(sample_code: str) -> str:
    sc = str(sample_code).strip()

    # normalise common variants
    sc_norm = sc.replace("_", " ").replace("-", " ").strip()
    sc_low = sc_norm.lower()

    if sc_low.startswith("fine wide"):
        return "Fine wide"
    if sc_low.startswith("fine bi-modal") or sc_low.startswith("fine bi modal"):
        return "Fine Bi-Modal"
    if sc_low.startswith("fines"):
        return "Fines"
    if sc_low.startswith("middlings"):
        return "Middlings"
    if sc_low.startswith("coarse"):
        return "Coarse"
    if sc_low.startswith("mix"):
        return "Mix"

    return sc_norm



def marker_map(values: List[str]) -> Dict[str, str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    return {v: markers[i % len(markers)] for i, v in enumerate(values)}

def infer_fail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if FAIL_COL not in df.columns:
        df["IsFail"] = False
        return df
    s = df[FAIL_COL].astype(str).str.lower().str.strip()
    # if it's numeric 0/1, this still works because "1" won't match regex
    df["IsFail"] = s.str.contains(FAIL_REGEX, regex=True, na=False)
    return df

def infer_diaphragm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TEST_PROC not in df.columns:
        df["Diaphragm"] = pd.Series([None] * len(df), dtype="object")
        return df

    s = df[TEST_PROC].astype(str).str.strip().str.upper()

    # make column object dtype from the start (avoids FutureWarning)
    df["Diaphragm"] = pd.Series([None] * len(df), dtype="object")
    df.loc[s == "STD", "Diaphragm"] = "On"
    df.loc[s == "NO_PRES", "Diaphragm"] = "Off"

    return df


def load_data() -> pd.DataFrame:
    db = pd.read_excel(FILEPATH, sheet_name=DB_SHEET)
    psd = pd.read_excel(FILEPATH, sheet_name=PSD_SHEET)

    db.columns = [c.strip() for c in db.columns]
    psd.columns = [c.strip() for c in psd.columns]

    # numeric cols
    for c in [FMC, CLOTH, P_PRESS, P_TIME]:
        if c in db.columns:
            db[c] = to_num(db[c])
    if P10 in psd.columns:
        psd[P10] = to_num(psd[P10])

    df = db.merge(psd[[KEY, P10]], on=KEY, how="left")

    df["SampleType"] = df[KEY].apply(classify_sample_type)

    df = infer_fail(df)
    df = infer_diaphragm(df)

    return df


# -----------------------------
# Plot 1: Dual-axis, readable
# left y = P_T (open markers), right y = P_P (filled markers)
# one label per sample code
# -----------------------------
def plot_dual_axis(df: pd.DataFrame) -> None:
    # NO diaphragm separation
    d = df.dropna(subset=[P_TIME, P_PRESS, KEY]).copy()
    d[KEY] = d[KEY].astype(str)

    # underfilled flag (if column missing -> all False)
    if PUMP_PLOT_CODE in d.columns:
        underfilled = (
            d[PUMP_PLOT_CODE]
            .astype(str)
            .str.lower()
            .str.contains("underfilled", na=False)
        )
        d["IsUnderfilled"] = underfilled
    else:
        d["IsUnderfilled"] = False

    sample_codes = sorted(d[KEY].unique())
    n = len(sample_codes)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(7.5, 2.2 * n),
        sharex=True
    )

    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, sample_codes):
        dd = d[d[KEY] == sc].copy()

        # trend line EXCLUDES underfilled
        dd_line = dd[~dd["IsUnderfilled"]].sort_values(P_PRESS)

        if len(dd_line) >= 2:
            ax.plot(
                dd_line[P_PRESS],
                dd_line[P_TIME],
                marker="o",
                linestyle="-",
                markerfacecolor="none",
                markeredgecolor="black",
                color="black",
                linewidth=1.4
            )
        elif len(dd_line) == 1:
            ax.scatter(
                dd_line[P_PRESS],
                dd_line[P_TIME],
                marker="o",
                facecolors="none",
                edgecolors="black"
            )

        # UNDERFILLED points (not connected)
        dd_under = dd[dd["IsUnderfilled"]]
        if len(dd_under):
            ax.scatter(
                dd_under[P_PRESS],
                dd_under[P_TIME],
                marker="x",
                color="black",
                s=60,
                linewidths=1.6
            )

        ax.set_ylabel(r"$P_T$ (s)")
        ax.set_title(sc, loc="left", fontsize=10)

        # ---- Y-axis extension + more intervals ----
        y0, y1 = ax.get_ylim()
        pad = 0.08 * (y1 - y0) if (y1 > y0) else 1.0
        ax.set_ylim(y0 - pad, y1 + pad)

        ax.minorticks_on()
        ax.tick_params(axis="y", which="major", length=5)
        ax.tick_params(axis="y", which="minor", length=3)

        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)

    axes[-1].set_xlabel(r"Pumping pressure $P_P$ (bar)")
    fig.suptitle(
        "Pumping Tine vs Pumping Pressure",
        y=0.98
    )

    savefig(
        fig,
        os.path.join(OUTDIR, "01_pumping_PT_vs_PP_subplots_all.png")
    )







# -----------------------------
# Plot 2: Cloth vs FMC
# marker shape by sample type
# colour = FAIL (black) / PASS (light grey)
def plot_cloth_vs_fmc(df: pd.DataFrame) -> None:
    # single plot, fixed cloth values with equal spacing
    CLOTH_LEVELS = [130, 150, 230, 250, 260, 350]
    x_map = {v: i for i, v in enumerate(CLOTH_LEVELS)}

    d = df.dropna(subset=[CLOTH, FMC, KEY]).copy()
    d = d[d[CLOTH].isin(CLOTH_LEVELS)]
    d[KEY] = d[KEY].astype(str)

    # ---- FMC as percentage ----
    d[FMC] = d[FMC] * 100.0

    sample_codes = sorted(d[KEY].unique())
    n = len(sample_codes)

    # ---- 3 x 2 grid ----
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(9.0, 8.5),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for ax, sc in zip(axes, sample_codes):
        dd = d[d[KEY] == sc].copy()

        # map cloth sizes to evenly spaced categorical positions
        x = dd[CLOTH].map(x_map).values
        x = x + rng.normal(0, 0.04, size=len(x))  # tiny jitter

        # diaphragm ON = black
        on = dd["Diaphragm"] == "On"
        if on.any():
            ax.scatter(
                x[on],
                dd.loc[on, FMC],
                s=60,
                facecolors="black",
                edgecolors="black",
                zorder=3
            )

        # diaphragm OFF = hollow grey
        off = dd["Diaphragm"] == "Off"
        if off.any():
            ax.scatter(
                x[off],
                dd.loc[off, FMC],
                s=60,
                facecolors="none",
                edgecolors="0.6",
                linewidths=1.2,
                zorder=2
            )

        ax.set_title(sc, loc="left", fontsize=10)

        # ---- y-axis: more intervals ----
        ax.minorticks_on()
        ax.tick_params(axis="y", which="major", length=5)
        ax.tick_params(axis="y", which="minor", length=3)

        ax.grid(True, axis="y", which="major", alpha=0.35)
        ax.grid(True, axis="y", which="minor", alpha=0.18)

        # ---- x-axis: NO minor ticks (categorical) ----
        ax.tick_params(axis="x", which="minor", bottom=False)

    # turn off unused axes if fewer than 6 samples
    for ax in axes[len(sample_codes):]:
        ax.axis("off")

    # shared axis labels
    for ax in axes[::ncols]:
        ax.set_ylabel("FMC (%)")

    for ax in axes[-ncols:]:
        ax.set_xlabel("Filter cloth number")

    for ax in axes:
        ax.set_xticks(range(len(CLOTH_LEVELS)))
        ax.set_xticklabels(CLOTH_LEVELS)

    fig.suptitle("FMC vs Cloth number", y=0.98)

    savefig(
        fig,
        os.path.join(OUTDIR, "02_cloth_vs_FMC_subplots_fixedCloth.png")
    )








# -----------------------------
# Plot 3: FMC vs Sample Code
# filled = diaphragm ON, hollow = OFF
# -----------------------------
def plot_fmc_by_samplecode(df: pd.DataFrame) -> None:
    d = df.dropna(subset=[KEY, FMC]).copy()
    d[KEY] = d[KEY].astype(str)

    # ---- FMC as percentage ----
    d[FMC] = d[FMC] * 100.0

    order = sorted(d[KEY].unique().tolist())
    x_map = {sc: i for i, sc in enumerate(order)}

    types = sorted(d["SampleType"].unique().tolist())
    mm = marker_map(types)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rng = np.random.default_rng(1)

    for _, r in d.iterrows():
        t = r["SampleType"]
        m = mm[t]
        x = x_map[r[KEY]] + rng.normal(0, 0.07)
        dia = r.get("Diaphragm", np.nan)

        if dia == "On":
            ax.scatter(x, r[FMC], s=75, marker=m, facecolors="black", edgecolors="black", alpha=0.9)
        elif dia == "Off":
            ax.scatter(x, r[FMC], s=75, marker=m, facecolors="none", edgecolors="black", linewidths=1.2, alpha=0.9)
        else:
            ax.scatter(x, r[FMC], s=75, marker=m, facecolors="none", edgecolors="0.6", linewidths=1.2, alpha=0.9)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_xlabel("Sample Code")
    ax.set_ylabel("FMC (%)")
    ax.set_title("FMC by sample code")

    # ---- y-axis starts at 0 ----
    ax.set_ylim(bottom=0)

    ax.grid(True, axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker=mm[t], linestyle="",
                   markerfacecolor="none", markeredgecolor="black", markersize=8)
        for t in types
    ]
    ax.legend(handles, types, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, title="Sample name")

    savefig(fig, os.path.join(OUTDIR, "03_FMC_vs_SampleCode_diaphragmFill.png"))


def plot_dual_axis_singlepanel(df: pd.DataFrame) -> None:
    """
    Single shared plot:
      - x = Pumping pressure (P_PRESS)
      - y = Pumping time (P_TIME)
      - marker shape = SampleType (same mapping as plot_fmc_by_samplecode)
      - trend lines exclude underfilled points
      - underfilled points = black filled markers (not connected)
    """
    d = df.dropna(subset=[P_TIME, P_PRESS, KEY]).copy()
    d[KEY] = d[KEY].astype(str)

    # underfilled flag
    if PUMP_PLOT_CODE in d.columns:
        d["IsUnderfilled"] = (
            d[PUMP_PLOT_CODE]
            .astype(str)
            .str.lower()
            .str.contains("underfilled", na=False)
        )
    else:
        d["IsUnderfilled"] = False

    # same marker mapping as FMC plot
    types = sorted(d["SampleType"].unique().tolist())
    mm = marker_map(types)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    for t in types:
        dd = d[d["SampleType"] == t].copy()
        m = mm[t]

        # ---- trend line (EXCLUDES underfilled) ----
        dd_line = dd[~dd["IsUnderfilled"]].sort_values(P_PRESS)
        if len(dd_line) >= 2:
            ax.plot(
                dd_line[P_PRESS],
                dd_line[P_TIME],
                linestyle="-",
                linewidth=1.4,
                color="black",
                alpha=0.8
            )

            ax.scatter(
                dd_line[P_PRESS],
                dd_line[P_TIME],
                s=70,
                marker=m,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.9
            )

        elif len(dd_line) == 1:
            ax.scatter(
                dd_line[P_PRESS],
                dd_line[P_TIME],
                s=70,
                marker=m,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2
            )

        # ---- underfilled points (black filled, NOT connected) ----
        dd_under = dd[dd["IsUnderfilled"]]
        if len(dd_under):
            ax.scatter(
                dd_under[P_PRESS],
                dd_under[P_TIME],
                s=70,
                marker=m,
                facecolors="black",
                edgecolors="black",
                alpha=0.95
            )

    ax.set_xlabel(r"Pumping pressure (bar)")
    ax.set_ylabel(r"Pumping Time (s)")
    ax.set_title("Pumping Time vs Pumping Pressure")
    ax.grid(True, alpha=0.3)

    # ---- y-axis extension + more intervals ----
    y0, y1 = ax.get_ylim()
    pad = 0.08 * (y1 - y0) if (y1 > y0) else 1.0
    ax.set_ylim(y0 - pad, y1 + pad)

    ax.minorticks_on()
    ax.tick_params(axis="y", which="major", length=5)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18)

    # legend (same logic as FMC plot)
    handles = [
        plt.Line2D(
            [0], [0],
            marker=mm[t], linestyle="",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8
        )
        for t in types
    ]
    ax.legend(
        handles, types,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Sample name"
    )

    savefig(fig, os.path.join(OUTDIR, "01_pumping_PT_vs_PP_singlepanel_trendlines.png"))



def main():
    ensure_outdir(OUTDIR)
    df = load_data()

    # sanity prints (important)
    print("Rows:", len(df))
    print("Unique SampleType:", df["SampleType"].value_counts(dropna=False))
    print("Fail counts:", df["IsFail"].value_counts(dropna=False))
    print("Diaphragm counts:", df["Diaphragm"].value_counts(dropna=False))

    plot_dual_axis_singlepanel(df)
    plot_dual_axis(df)
    plot_cloth_vs_fmc(df)
    plot_fmc_by_samplecode(df)

    print("Saved plots to:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
