import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIG
# ============================================================

FILE_PATH = r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx"

SHEET_DB  = "DB_2"   # sheet with final moisture + Test_procedure
SHEET_PSD = "PSD"    # sheet with D10, D80_over_D20 etc.

# Common key between DB_2 and PSD:
COL_SAMPLE   = "Sample Code"   # change if different

# PSD sheet columns
COL_D10      = "D10"           # in PSD sheet
COL_SPAN     = "D80/D20"  # in PSD sheet (span = D80/D20)

# DB sheet columns
COL_MC_FINAL = "Mc_%"          # final moisture content in DB_2
COL_DIAPH    = "Test_procedure"  # diaphragm on/off encoded as STD / No_press

# If you already have coded D10/Span levels, set these to the column names
EXISTING_D10_LEVEL_COL  = None   # e.g. "D10_level"
EXISTING_SPAN_LEVEL_COL = None   # e.g. "Span_level"

# ============================================================
# 2. LOAD & MERGE DATA (PSD + DB_2)
# ============================================================

# Load DB (final moisture, Test_procedure, etc.)
db = pd.read_excel(FILE_PATH, sheet_name=SHEET_DB)

# Load PSD (D10, span, etc.)
psd = pd.read_excel(FILE_PATH, sheet_name=SHEET_PSD)

# Keep only the columns we need
db = db[[COL_SAMPLE, COL_MC_FINAL, COL_DIAPH]].copy()
psd = psd[[COL_SAMPLE, COL_D10, COL_SPAN]].copy()

# Rename to standard internal names
db = db.rename(columns={
    COL_SAMPLE:   "Sample",
    COL_MC_FINAL: "Mc_final",
    COL_DIAPH:    "Test_procedure"
})

psd = psd.rename(columns={
    COL_SAMPLE: "Sample",
    COL_D10:    "D10",
    COL_SPAN:   "Span"
})

# Inner join: only samples that appear in BOTH DB_2 and PSD
df = pd.merge(db, psd, on="Sample", how="inner")

# Drop rows with missing essentials
df = df.dropna(subset=["D10", "Span", "Mc_final", "Test_procedure"]).copy()

# ============================================================
# 3. CREATE FACTORS / LEVELS
# ============================================================

def make_three_level_factor(series, labels=("low", "mid", "high"),
                            method="quantile", cutpoints=None):
    """
    Convert a numeric series into a 3-level categorical factor.
    method='quantile' -> equal-sized bins by data quantiles
    method='cut'      -> use explicit cutpoints (e.g. [0, 10, 40, 100])
    """
    if cutpoints is not None:
        return pd.cut(series, bins=cutpoints, labels=labels, include_lowest=True)
    else:
        return pd.qcut(series, q=3, labels=labels)

# ---- D10 levels ----
if EXISTING_D10_LEVEL_COL is not None and EXISTING_D10_LEVEL_COL in df.columns:
    df["D10_level"] = df[EXISTING_D10_LEVEL_COL].astype("category")
else:
    df["D10_level"] = make_three_level_factor(df["D10"])

# ---- Span levels (optional, for grouping if needed later) ----
if EXISTING_SPAN_LEVEL_COL is not None and EXISTING_SPAN_LEVEL_COL in df.columns:
    df["Span_level"] = df[EXISTING_SPAN_LEVEL_COL].astype("category")
else:
    df["Span_level"] = make_three_level_factor(df["Span"])

# ---- Diaphragm factor from Test_procedure ----
# STD = On, No_press = Off
proc_raw = df["Test_procedure"].astype(str).str.strip()

map_proc_to_diag = {
    "STD": "On",
    "No_press": "Off",
    "NO_PRESS": "Off",
    "No Press": "Off"
}

df["Diaphragm_fac"] = proc_raw.map(map_proc_to_diag)
# If there are any weird values, just keep their original label:
df.loc[df["Diaphragm_fac"].isna(), "Diaphragm_fac"] = proc_raw[df["Diaphragm_fac"].isna()]

df["D10_level"]      = pd.Categorical(df["D10_level"],
                                      categories=["low", "mid", "high"],
                                      ordered=True)
df["Span_level"]     = pd.Categorical(df["Span_level"],
                                      categories=["low", "mid", "high"],
                                      ordered=True)
df["Diaphragm_fac"]  = pd.Categorical(df["Diaphragm_fac"],
                                      categories=["Off", "On"],
                                      ordered=True)

# ============================================================
# 4. HELPER: GROUP TO MEAN ± STD FOR PLOTTING
# ============================================================

def summarize_for_interaction(df, x_col, group_col, y_col):
    """
    Returns a tidy summary table indexed by (group_col, x_col) with:
      - mean_y
      - std_y
      - count
    """
    summary = (
        df
        .groupby([group_col, x_col], observed=True)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_y", "std": "std_y"})
    )
    return summary

# ============================================================
# 5. PLOT 1: Span vs Final Moisture, lines = D10_level
#    (Interaction: D10 × Span)
# ============================================================

summary1 = summarize_for_interaction(df, x_col="Span",
                                     group_col="D10_level",
                                     y_col="Mc_final")

plt.figure(figsize=(7, 5))
for level in df["D10_level"].cat.categories:
    sub = summary1[summary1["D10_level"] == level].sort_values("Span")
    if len(sub) == 0:
        continue
    plt.errorbar(
        sub["Span"], sub["mean_y"],
        yerr=sub["std_y"],
        marker="o", linestyle="-", capsize=4, label=f"D10 {level}"
    )

plt.xlabel("Span (D80/D20)")
plt.ylabel("Final moisture (%)")
plt.title("Interaction: Span vs Final Moisture\nLines = D10 level")
plt.legend(title="D10 level")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 6. PLOT 2: D10 vs Final Moisture, lines = Diaphragm on/off
#    (Interaction: D10 × Diaphragm)
# ============================================================

summary2 = summarize_for_interaction(df, x_col="D10",
                                     group_col="Diaphragm_fac",
                                     y_col="Mc_final")

plt.figure(figsize=(7, 5))
for level in df["Diaphragm_fac"].cat.categories:
    sub = summary2[summary2["Diaphragm_fac"] == level].sort_values("D10")
    if len(sub) == 0:
        continue
    plt.errorbar(
        sub["D10"], sub["mean_y"],
        yerr=sub["std_y"],
        marker="o", linestyle="-", capsize=4, label=f"Diaphragm {level}"
    )

plt.xlabel("D10 (\u00B5m)")
plt.ylabel("Final moisture (%)")
plt.title("Interaction: D10 vs Final Moisture\nLines = Diaphragm press")
plt.legend(title="Diaphragm")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 7. PLOT 3: Span vs Final Moisture, lines = Diaphragm on/off
#    (Interaction: Span × Diaphragm)
# ============================================================

summary3 = summarize_for_interaction(df, x_col="Span",
                                     group_col="Diaphragm_fac",
                                     y_col="Mc_final")

plt.figure(figsize=(7, 5))
for level in df["Diaphragm_fac"].cat.categories:
    sub = summary3[summary3["Diaphragm_fac"] == level].sort_values("Span")
    if len(sub) == 0:
        continue
    plt.errorbar(
        sub["Span"], sub["mean_y"],
        yerr=sub["std_y"],
        marker="o", linestyle="-", capsize=4, label=f"Diaphragm {level}"
    )

plt.xlabel("Span (D80/D20)")
plt.ylabel("Final moisture (%)")
plt.title("Interaction: Span vs Final Moisture\nLines = Diaphragm press")
plt.legend(title="Diaphragm")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
