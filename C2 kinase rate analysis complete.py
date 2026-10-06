"""
Lipid kinase initial rate analysis — complete pipeline
--------------------------------------------------------------------------
Input: concentration-converted luminescence time courses, expressed as
uM ATP consumed / nM kinase (i.e. already normalized to enzyme
concentration via an ADP-Glo standard curve).

This script:
  1. Scans R^2 across candidate fit windows to help you judge linearity
  2. Computes initial rate (linear regression slope) per replicate,
     in uM ATP / nM kinase / min
  3. Converts to mol ATP / mol kinase / min (turnover number per min)
     by multiplying by 1000 (since 1 uM / 1 nM = 1000 mol/mol)
  4. Runs Dunnett's test (each group vs one control group only —
     use this instead of Tukey when you're comparing treatments to a
     single control rather than doing all pairwise comparisons)
  5. Produces a bar graph (mean +/- SEM, individual points overlaid)
  6. Saves a CSV of per-replicate results and the bar graph as PNG

Requirements:
    pip install numpy pandas scipy matplotlib

Usage:
    python kinase_rate_analysis_complete.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import dunnett
import matplotlib.pyplot as plt

# =====================================================================
# 1. TIME POINTS (min) — edit to match your own data
# =====================================================================
t = np.array([0, 0.6253, 1.250717, 1.876183, 2.501733, 3.127133, 3.75265,
              4.37825, 5.003683, 5.62905, 6.254567, 6.880033, 7.50545,
              8.131067, 8.756433, 9.382017, 10.0078])

# =====================================================================
# 2. DATA — uM ATP / nM kinase, one list per replicate
# >>> EDIT THIS SECTION TO USE YOUR OWN DATA <<<
# =====================================================================
data = {
    "C2": [
        [0.055315288,0.060936625,0.066311282,0.071887768,0.077023218,0.082539903,0.086569026,0.09307991,0.09574855,0.10024861,0.102879874,0.105952174,0.109547737,0.112223852,0.113756264,0.117419104,0.118435728],
        [0.045246218,0.048303567,0.053162436,0.05725136,0.060488114,0.064165904,0.06729053,0.07239608,0.076043969,0.076776537,0.080947689,0.083324797,0.086449423,0.086897934,0.09082988,0.092078235,0.094298364],
        [0.054478068,0.057505517,0.060944101,0.065436685,0.069435908,0.072560534,0.077195147,0.079617106,0.08263708,0.087705254,0.090321567,0.091816604,0.094552521,0.096144734,0.097520168,0.100764397,0.102902299],
        [0.078331375,0.083930286,0.090201964,0.095913004,0.101556767,0.104808471,0.110392432,0.11543818,0.118159147,0.123406725,0.128758955,0.1275106,0.132989909,0.135927655,0.13837204,0.14023336,0.141907801],
    ],
    "C2+WT": [
        [0.040813434,0.042308471,0.044700529,0.047070162,0.049873355,0.051622548,0.053401642,0.054926579,0.058619319,0.058581943,0.062274683,0.063433336,0.065219905,0.066341182,0.066311282,0.068307155,0.070220802],
        [0.038324199,0.040985364,0.041635704,0.043594202,0.045732104,0.046584275,0.048415695,0.050478845,0.052138336,0.053110109,0.054806976,0.056638396,0.056892552,0.060577817,0.060241433,0.062850272,0.063687493],
        [0.039124043,0.040521902,0.041538527,0.044349196,0.04454355,0.046883283,0.049417369,0.049522022,0.051824378,0.055330239,0.05680285,0.057684921,0.059538766,0.059441589,0.060555391,0.062312059,0.062865222],
        [0.043242869,0.044304345,0.047002885,0.04983598,0.049978008,0.053244663,0.054792025,0.05622726,0.058507191,0.060331136,0.062184981,0.063747294,0.064719068,0.067380233,0.068239879,0.067469935,0.06890517],
    ],
    "C2+mut": [
        [0.058125957,0.063029676,0.070175951,0.075961742,0.079848837,0.085582302,0.088490148,0.094918805,0.099725347,0.103903974,0.107596714,0.109375807,0.114623385,0.117404153,0.119437403,0.119998042,0.12342915],
        [0.055980579,0.060069504,0.066356133,0.069981596,0.076641984,0.077994992,0.083511676,0.086210217,0.090964433,0.092952832,0.097542594,0.102072554,0.104741194,0.106998699,0.108957197,0.111364206,0.114316903],
        [0.061803747,0.064278032,0.069361156,0.074167698,0.078226722,0.083197719,0.087847282,0.088871382,0.093685399,0.098865701,0.102154781,0.105675592,0.108045225,0.110975496,0.114346804,0.116574408,0.1182937],
        [0.079519929,0.086882983,0.093745201,0.098775999,0.107656515,0.115213925,0.12240505,0.128407622,0.132182589,0.13692933,0.140599644,0.145989251,0.148201905,0.153815767,0.155527583,0.161679659,0.161657233],
    ],
}

GROUP_ORDER = ["C2", "C2+WT", "C2+mut"]   # order used for plotting/stats
CONTROL_GROUP = "C2+WT"                    # Dunnett's control group
COLORS = ['white', '#e74c3c', '#2962ff']   # bar colors, edit as you like

# =====================================================================
# 3. CHOOSE THE FIT WINDOW (0 to FIT_WINDOW_MIN)
# Snaps to the nearest available time point.
# =====================================================================
FIT_WINDOW_MIN = 8.131067   # change to e.g. 10.0078 for the full range

# ---------------------------------------------------------------------
# (Optional) scan R^2 across candidate windows to help you pick one
# ---------------------------------------------------------------------
print("=" * 70)
print("R^2 by fit window (per replicate)")
print("=" * 70)
for group, reps in data.items():
    print(f"\n--- {group} ---")
    for i, y in enumerate(reps):
        y = np.array(y)
        slope, intercept, r, p, se = stats.linregress(t, y)
        print(f"  rep{i+1} (full range): slope={slope:.5f}, R2={r**2:.4f}")

# =====================================================================
# 4. COMPUTE INITIAL RATE (linear regression slope) PER REPLICATE
#    Units: uM ATP / nM kinase / min
# =====================================================================
n_pts = int(np.argmin(np.abs(t - FIT_WINDOW_MIN))) + 1
used_window = t[n_pts - 1]

records = []
for group, reps in data.items():
    for i, y in enumerate(reps):
        y = np.array(y)
        slope, intercept, r, p, se = stats.linregress(t[:n_pts], y[:n_pts])
        records.append({
            "Group": group,
            "Replicate": i + 1,
            "Rate_uMATP_per_nMkinase_per_min": slope,
            "R2": r ** 2,
        })

df = pd.DataFrame(records)

# =====================================================================
# 5. CONVERT TO mol ATP / mol kinase / min (turnover number per min)
#    1 uM / 1 nM = (1e-6 mol/L) / (1e-9 mol/L) = 1000 mol/mol
# =====================================================================
df["Rate_molATP_per_molkinase_per_min"] = df["Rate_uMATP_per_nMkinase_per_min"] * 1000

print("\n" + "=" * 70)
print(f"Initial rates (linear fit, 0-{used_window:.2f} min)")
print("=" * 70)
print(df.to_string(index=False))

summary = df.groupby("Group")["Rate_molATP_per_molkinase_per_min"].agg(["mean", "std", "sem", "count"])
summary = summary.reindex(GROUP_ORDER)
print("\nSummary (mol ATP / mol kinase / min):")
print(summary)

# =====================================================================
# 6. STATISTICS — Dunnett's test (each group vs CONTROL_GROUP only)
# =====================================================================
other_groups = [g for g in GROUP_ORDER if g != CONTROL_GROUP]
group_vals = [df[df.Group == g]["Rate_molATP_per_molkinase_per_min"].values for g in other_groups]
control_vals = df[df.Group == CONTROL_GROUP]["Rate_molATP_per_molkinase_per_min"].values

result = dunnett(*group_vals, control=control_vals)
print(f"\nDunnett's test (control = {CONTROL_GROUP}):")
print(result)
for g, p in zip(other_groups, result.pvalue):
    print(f"  {g} vs {CONTROL_GROUP}: p = {p:.6f}")

# =====================================================================
# 7. BAR GRAPH (mean + SEM, individual points overlaid)
# =====================================================================
means = [summary.loc[g, "mean"] for g in GROUP_ORDER]
sems = [summary.loc[g, "sem"] for g in GROUP_ORDER]

fig, ax = plt.subplots(figsize=(4.5, 5))
x = np.arange(len(GROUP_ORDER))
ax.bar(x, means, yerr=sems, capsize=5, width=0.6,
       color=COLORS, edgecolor='black', linewidth=1.5)

rng = np.random.default_rng(0)  # fixed seed so jitter is reproducible
for i, g in enumerate(GROUP_ORDER):
    yvals = df[df.Group == g]["Rate_molATP_per_molkinase_per_min"].values
    xvals = rng.normal(i, 0.05, size=len(yvals))
    ax.scatter(xvals, yvals, color='black', s=30, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(GROUP_ORDER)
ax.set_ylabel(f"Initial rate (0-{used_window:.2f} min)\n(mol ATP / mol kinase / min)")
plt.tight_layout()

df.to_csv("kinase_rates.csv", index=False)
plt.savefig("kinase_rates_barplot.png", dpi=300)
print("\nSaved: kinase_rates.csv, kinase_rates_barplot.png")

plt.show()
