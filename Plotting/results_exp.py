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


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def savefig(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

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
    def _plot_one(d: pd.DataFrame, tag: str) -> None:
        d = d.dropna(subset=[P_TIME, P_PRESS, KEY]).copy()
        d[KEY] = d[KEY].astype(str)

        sample_codes = sorted(d[KEY].unique().tolist())

        line_styles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

        fig, ax = plt.subplots(figsize=(8.2, 5.0))

        for i, sc in enumerate(sample_codes):
            dd = d[d[KEY] == sc].sort_values(P_PRESS)

            ls = line_styles[i % len(line_styles)]
            mk = markers[i % len(markers)]

            ax.plot(
                dd[P_PRESS].values, dd[P_TIME].values,
                linestyle=ls, linewidth=1.4,
                marker=mk, markersize=6,
                markerfacecolor="none", markeredgecolor="black",
                color="black", alpha=0.85,
                label=sc
            )

        ax.set_xlabel(r"Pumping pressure $P_P$ (bar)")
        ax.set_ylabel(r"Pumping time $P_T$ (s)")
        ax.set_title(f"Pumping screening ({tag}): $P_T$ vs $P_P$")
        ax.grid(True, alpha=0.25)

        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Sample (line style)")
        savefig(fig, os.path.join(OUTDIR, f"01_pumping_PT_vs_PP_line_{tag}.png"))

    # Dia ON = STD
    d_on = df[df["Diaphragm"] == "On"].copy()
    if len(d_on):
        _plot_one(d_on, "DiaOn")

    # Dia OFF = NO_PRES
    d_off = df[df["Diaphragm"] == "Off"].copy()
    if len(d_off):
        _plot_one(d_off, "DiaOff")




# -----------------------------
# Plot 2: Cloth vs FMC
# marker shape by sample type
# colour = FAIL (black) / PASS (light grey)
# -----------------------------
def plot_cloth_vs_fmc(df: pd.DataFrame) -> None:
    def _plot_one(d: pd.DataFrame, tag: str) -> None:
        d = d.dropna(subset=[CLOTH, FMC, KEY]).copy()
        d[KEY] = d[KEY].astype(str)

        sample_codes = sorted(d[KEY].unique().tolist())

        line_styles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        rng = np.random.default_rng(0)

        for i, sc in enumerate(sample_codes):
            dd = d[d[KEY] == sc].sort_values(CLOTH)

            ls = line_styles[i % len(line_styles)]
            mk = markers[i % len(markers)]

            # jitter cloth slightly so overlapping cloth values are visible
            x = dd[CLOTH].values + rng.normal(0, 1.0, size=len(dd))

            # line (neutral)
            ax.plot(
                x, dd[FMC].values,
                linestyle=ls, linewidth=1.2,
                color="0.3", alpha=0.75
            )

            # points (colour-coded by fail)
            pass_mask = ~dd["IsFail"]
            if pass_mask.any():
                ax.scatter(
                    x[pass_mask], dd.loc[pass_mask, FMC].values,
                    s=65, marker=mk,
                    facecolors="none", edgecolors="0.6", linewidths=1.2,
                    alpha=0.95
                )

            fail_mask = dd["IsFail"]
            if fail_mask.any():
                ax.scatter(
                    x[fail_mask], dd.loc[fail_mask, FMC].values,
                    s=65, marker=mk,
                    facecolors="black", edgecolors="black",
                    alpha=0.95
                )

        ax.set_xlabel(r"Filter cloth ($\mu$m)")
        ax.set_ylabel("Final moisture content, FMC (%)")
        ax.set_title(f"Cloth screening ({tag}): FMC vs cloth (black points = FAIL)")
        ax.grid(True, axis="y", alpha=0.25)

        savefig(fig, os.path.join(OUTDIR, f"02_cloth_vs_FMC_line_{tag}.png"))

    # Dia ON
    d_on = df[df["Diaphragm"] == "On"].copy()
    if len(d_on):
        _plot_one(d_on, "DiaOn")

    # Dia OFF
    d_off = df[df["Diaphragm"] == "Off"].copy()
    if len(d_off):
        _plot_one(d_off, "DiaOff")




# -----------------------------
# Plot 3: FMC vs Sample Code
# filled = diaphragm ON, hollow = OFF
# -----------------------------
def plot_fmc_by_samplecode(df: pd.DataFrame) -> None:
    d = df.dropna(subset=[KEY, FMC]).copy()
    d[KEY] = d[KEY].astype(str)

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
    ax.set_ylabel("Final moisture content, FMC (%)")
    ax.set_title("FMC by sample code (filled = diaphragm ON, hollow = OFF)")
    ax.grid(True, axis="y", alpha=0.25)

    handles = [plt.Line2D([0], [0], marker=mm[t], linestyle="",
                          markerfacecolor="none", markeredgecolor="black", markersize=8)
               for t in types]
    ax.legend(handles, types, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, title="Sample type (shape)")

    savefig(fig, os.path.join(OUTDIR, "03_FMC_vs_SampleCode_diaphragmFill.png"))


def main():
    ensure_outdir(OUTDIR)
    df = load_data()

    # sanity prints (important)
    print("Rows:", len(df))
    print("Unique SampleType:", df["SampleType"].value_counts(dropna=False))
    print("Fail counts:", df["IsFail"].value_counts(dropna=False))
    print("Diaphragm counts:", df["Diaphragm"].value_counts(dropna=False))

    plot_dual_axis(df)
    plot_cloth_vs_fmc(df)
    plot_fmc_by_samplecode(df)

    print("Saved plots to:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
