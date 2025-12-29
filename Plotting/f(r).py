import numpy as np
import matplotlib.pyplot as plt

def lognormal_pdf(r, mu, sigma):
    pdf = np.zeros_like(r, dtype=float)
    m = r > 0
    rr = r[m]
    pdf[m] = (1.0 / (rr * sigma * np.sqrt(2*np.pi))) * np.exp(-((np.log(rr) - mu)**2) / (2*sigma**2))
    return pdf

# -----------------------------
# Setup
# -----------------------------
r_max = 60.0
r = np.linspace(0.0, r_max, 4000)

# Move threshold left
Pmax_threshold = 10.0

# Narrow: mostly small r but with tail past Pmax
mu_narrow, sigma_narrow = np.log(8.0), 0.35

# Wide: broader + more large r (right-skew)
mu_wide, sigma_wide = np.log(14.0), 0.65

y_narrow = lognormal_pdf(r, mu_narrow, sigma_narrow)
y_wide   = lognormal_pdf(r, mu_wide, sigma_wide)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(
    r, y_narrow,
    label="Narrow f(r)",
    color="0.4",          # grey (0 = black, 1 = white)
    linestyle="solid",
    linewidth=2.5
)

ax.plot(
    r, y_wide,
    label="Coarse 'shifted' f(r)",
    color="0.4",
    linestyle="dotted",   # or "--" if you prefer dashed
    linewidth=2.5
)

# Pmax vertical line
ax.axvline(Pmax_threshold, linewidth=2, color="black")

# Hatched region ONLY under each curve (to the right of Pmax)
ax.fill_between(
    r, 0, y_narrow,
    where=(r <= Pmax_threshold),
    hatch="///",
    facecolor="none",
    edgecolor="grey",
    linewidth=0.0
)
ax.fill_between(
    r, 0, y_wide,
    where=(r <= Pmax_threshold),
    hatch="\\\\\\",
    facecolor="none",
    edgecolor="grey",
    linewidth=0.0
)

# Labels
ax.set_xlabel(" increasing pore throat radius $ \\rightarrow$ \n $\\leftarrow$ increasing capillary pressure ")


ax.set_ylabel("Probability density")

# Annotation
y_top = max(y_narrow.max(), y_wide.max())
ax.text(
    Pmax_threshold, y_top * 0.95,
    "Pmax threshold",
    rotation=90,
    va="top",
    ha="left"
)

# Remove ALL x-axis ticks/markers (no 0,10,20 etc; no tick marks)
ax.set_xticks([])
ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

ax.set_xlim(0, r_max)
ax.set_ylim(0, y_top * 1.05)
ax.legend()
fig.tight_layout()
plt.show()
