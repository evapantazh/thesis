"""
export_baseline_bland_altman.py
================================

Bland-Altman plot for the n=13 baseline configuration recordings
(T-shirt + 1200 mm). For Section 4.1.

X-axis: mean of (estimate + ground truth) / 2
Y-axis: signed error (estimate - ground truth)
Horizontal lines: bias, 95% limits of agreement (bias +/- 1.96 * SD_signed)
GBA outlier labeled.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Thesis style — matches Figure 4.1 scatter
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

# Filter to baseline: T-shirt + 1200 mm, status = ok
sub = df[(df['dist'] == 1200) & (df['cloth'] == 'tshirt') & (df['status'] == 'ok')].copy()

print(f"n = {len(sub)} baseline recordings")

gt = sub['gt_bpm'].values
est = sub['bpm_smoothed'].values

# Bland-Altman axes
mean_hr = (gt + est) / 2.0
signed_err = est - gt

# Statistics
bias = signed_err.mean()
sd_signed = signed_err.std(ddof=1)  # sample SD, matches Bland-Altman convention
loa_lower = bias - 1.96 * sd_signed
loa_upper = bias + 1.96 * sd_signed

print(f"Bias       : {bias:.2f} BPM")
print(f"SD signed  : {sd_signed:.2f} BPM")
print(f"95% LoA    : [{loa_lower:.2f}, {loa_upper:.2f}] BPM")

# Colors — thesis slate-blue / red-accent palette
C_POINTS  = '#5b7c99'   # primary slate blue
C_BIAS    = '#5b7c99'   # slate blue (bias line)
C_LOA     = '#a01818'   # red accent (limits of agreement)
C_ZERO    = '0.5'       # neutral gray (zero-error reference)
C_OUTLIER = '#a01818'   # red accent

# Figure
fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)

# Zero line (perfect agreement)
ax.axhline(0, color=C_ZERO, linestyle=':', linewidth=0.8, alpha=0.5)

# Bias line (solid)
ax.axhline(bias, color=C_BIAS, linestyle='-', linewidth=1.5,
           label=f'Bias = {bias:.2f} BPM')

# 95% Limits of agreement (dashed)
ax.axhline(loa_upper, color=C_LOA, linestyle='--', linewidth=1.2,
           label=f'95% LoA = [{loa_lower:.2f}, {loa_upper:.2f}] BPM')
ax.axhline(loa_lower, color=C_LOA, linestyle='--', linewidth=1.2)

# Scatter points — solid slate blue, no colorbar
ax.scatter(mean_hr, signed_err, color=C_POINTS, s=70,
           edgecolor='black', linewidth=0.4, zorder=3,
           label='Recordings')

# Label the GBA outlier
gba_mask = sub['subject'] == 'GBA'
if gba_mask.any():
    gba = sub[gba_mask].iloc[0]
    gba_mean = (gba['gt_bpm'] + gba['bpm_smoothed']) / 2.0
    gba_err = gba['bpm_smoothed'] - gba['gt_bpm']
    ax.scatter(gba_mean, gba_err, color=C_OUTLIER, s=80,
               edgecolor='black', linewidth=0.5, zorder=4)
    ax.annotate('GBA', (gba_mean, gba_err),
                xytext=(10, -4), textcoords='offset points',
                fontsize=9, color=C_OUTLIER, fontweight='bold')

# Bias and LoA text labels (right side of plot)
ax.text(max(mean_hr) + 1, bias,
        '  Bias', va='center', fontsize=8, color=C_BIAS)
ax.text(max(mean_hr) + 1, loa_upper,
        '  +1.96 SD', va='center', fontsize=8, color=C_LOA)
ax.text(max(mean_hr) + 1, loa_lower,
        '  -1.96 SD', va='center', fontsize=8, color=C_LOA)

# Axes
ax.set_xlabel(r'Mean of $\widehat{\mathrm{HR}}$ and $\mathrm{HR}_{\mathrm{GT}}$ (BPM)')
ax.set_ylabel(r'$\widehat{\mathrm{HR}} - \mathrm{HR}_{\mathrm{GT}}$ (BPM)')

# Legend
ax.legend(loc='upper left', framealpha=0.9, frameon=False)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save
plt.savefig(OUTPUT_DIR / 'baseline_bland_altman.pdf')
plt.savefig(OUTPUT_DIR / 'baseline_bland_altman.png')
print(f"\nSaved to {OUTPUT_DIR}")
plt.show()