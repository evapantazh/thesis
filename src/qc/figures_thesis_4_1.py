"""
export_baseline_scatter.py
==========================

Scatter plot of estimated HR vs ECG ground truth for the n=13 baseline
configuration recordings (T-shirt + 1200 mm). For Section 4.1.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Thesis style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CSV_PATH = Path(r"C:\Projects\thesis\data\PULSE_files_SQI\summaries\summary_20260622_160818.csv")
OUTPUT_DIR = Path(r"C:\Projects\thesis\thesis_latex\images")

df = pd.read_csv(CSV_PATH)

# Filter to baseline configuration using the dedicated columns
sub = df[(df['dist'] == 1200) & (df['cloth'] == 'tshirt') & (df['status'] == 'ok')].copy()

print(f"n = {len(sub)} baseline recordings")
print(f"MAE = {sub['err_smoothed'].mean():.2f} BPM")
print(f"Median err = {sub['err_smoothed'].median():.2f} BPM")
print(f"SD err = {sub['err_smoothed'].std():.2f} BPM")

gt = sub['gt_bpm'].values
est = sub['bpm_smoothed'].values
err = sub['err_smoothed'].values

# Colors — thesis slate-blue / red-accent palette
C_POINTS   = '#5b7c99'   # primary slate blue
C_IDENTITY = '#2b2b2b'   # dark gray/near-black
C_BAND     = '#5b7c99'   # slate blue, low alpha
C_OUTLIER  = '#a01818'   # red accent

# Figure
fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)

# Identity line and ±5 BPM band
lim_lo = min(gt.min(), est.min()) - 5
lim_hi = max(gt.max(), est.max()) + 5
xs = np.linspace(lim_lo, lim_hi, 100)
ax.fill_between(xs, xs - 5, xs + 5, color=C_BAND, alpha=0.12,
                label=r'$\pm 5$ BPM band')
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_IDENTITY, ls='--',
        lw=1.0, label='Identity')

# Scatter — solid slate blue, no colorbar
ax.scatter(gt, est, color=C_POINTS, s=60,
           edgecolor='black', linewidth=0.4, zorder=3,
           label='Recordings')

# Mark the GBA outlier
gba_mask = sub['subject'] == 'GBA'
if gba_mask.any():
    gba = sub[gba_mask].iloc[0]
    ax.scatter(gba['gt_bpm'], gba['bpm_smoothed'],
               color=C_OUTLIER, s=70, edgecolor='black',
               linewidth=0.5, zorder=4)
    ax.annotate('GBA', (gba['gt_bpm'], gba['bpm_smoothed']),
                xytext=(8, -2), textcoords='offset points',
                fontsize=9, color=C_OUTLIER)

ax.set_xlabel(r'ECG ground truth $\mathrm{HR}_{\mathrm{GT}}$ (BPM)')
ax.set_ylabel(r'Estimated $\widehat{\mathrm{HR}}$ (BPM)')
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(OUTPUT_DIR / 'baseline_scatter.pdf')
plt.savefig(OUTPUT_DIR / 'baseline_scatter.png')
print(f"\nSaved to {OUTPUT_DIR}")
plt.show()