#!/usr/bin/env python3
"""Figure 4.x - Sensitivity of GBA_1200_tshirt error to RPCA gamma.
Data from Table 4.8. Ground truth 78.3 BPM."""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUTDIR = r"C:\Projects\thesis\thesis_latex\images"
os.makedirs(OUTDIR, exist_ok=True)

MAIN = "#5b7c99"
ACCENT = "#a01818"

# Table 4.8, no-SQI sweep (evaluable range)
g   = [0.07, 0.09, 0.10, 0.50, 1.10, 1.30]
err = [12.3, 8.8, 12.6, 13.2, 13.2, 13.2]

# Special points
g_sqi, err_sqi = 0.10, 8.8      # production setting with SQI=25
pca_limit = 13.2                # PCA-on-raw-data upper bound

fig, ax = plt.subplots(figsize=(6.2, 3.4))

# main no-SQI curve
ax.plot(g, err, '-o', color=MAIN, markersize=5, linewidth=1.3,
        markeredgecolor='black', markeredgewidth=0.4,
        label='no SQI filter', zorder=3)

# SQI=25 point
ax.plot(g_sqi, err_sqi, 'D', color=ACCENT, markersize=7,
        markeredgecolor='black', markeredgewidth=0.4,
        label='SQI = 25', zorder=4)

# PCA-on-raw-data asymptote
ax.axhline(pca_limit, color='0.45', linestyle=':', linewidth=1.0, zorder=1)
ax.text(1.32, pca_limit + 0.12, 'PCA-on-raw-data limit',
        ha='right', va='bottom', fontsize=8, color='0.4')

# annotate the gamma=0.02 rejection
ax.annotate('$\\gamma=0.02$:\nall windows\nrejected',
            xy=(0.021, 9.2), fontsize=7.8, color='0.4',
            ha='left', va='center')
ax.axvline(0.02, color='0.7', linestyle='--', linewidth=0.8, zorder=0)

ax.set_xscale('log')
ax.set_xlabel('RPCA regularisation parameter $\\gamma$')
ax.set_ylabel('Absolute error (BPM)')
ax.set_ylim(7.5, 14)
ax.set_xlim(0.015, 1.6)
ax.set_xticks([0.02, 0.1, 0.5, 1.0])
ax.set_xticklabels(['0.02', '0.1', '0.5', '1.0'])
ax.legend(frameon=False, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', length=3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_gamma_sensitivity.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_gamma_sensitivity.png'), dpi=150)
print(f"saved to {OUTDIR}")