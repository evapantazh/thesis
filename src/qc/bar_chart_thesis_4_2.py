"""
export_mae_by_dist_cloth.py
============================

Bar chart of MAE per distance × clothing condition.
For Section 4.2 of the thesis.
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CSV_PATH = Path(r"C:\Projects\thesis\data\PULSE_files_SQI\summaries\summary_20260622_160818.csv")
OUTPUT_DIR = Path(r"C:\Projects\thesis\thesis_latex\images")

df = pd.read_csv(CSV_PATH)
df = df[df['status'] == 'ok'].copy()

# Compute per-cell MAE and SD
grouped = df.groupby(['dist', 'cloth'])['err_smoothed'].agg(['mean', 'std', 'count']).reset_index()
print(grouped)

# Order
distances = [800, 1200, 1800]
cloths = ['tshirt', 'hoodie']

# Build arrays
means = {(d, c): grouped[(grouped['dist'] == d) & (grouped['cloth'] == c)]['mean'].values[0]
         for d in distances for c in cloths}
sds = {(d, c): grouped[(grouped['dist'] == d) & (grouped['cloth'] == c)]['std'].values[0]
       for d in distances for c in cloths}

tshirt_means = [means[(d, 'tshirt')] for d in distances]
hoodie_means = [means[(d, 'hoodie')] for d in distances]
tshirt_sds = [sds[(d, 'tshirt')] for d in distances]
hoodie_sds = [sds[(d, 'hoodie')] for d in distances]

# Figure
fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

x = np.arange(len(distances))
width = 0.38

bars1 = ax.bar(x - width/2, tshirt_means, width,
               yerr=tshirt_sds, capsize=4,
               label='T-shirt', color='#5b8db8', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, hoodie_means, width,
               yerr=hoodie_sds, capsize=4,
               label='Hoodie', color='#d4915c', edgecolor='black', linewidth=0.5)

# Reference line: 5 BPM clinical threshold
ax.axhline(5, color='grey', linestyle=':', linewidth=1.0, alpha=0.7)
ax.text(0.02, 5.15, r'$\pm 5$ BPM threshold', fontsize=8, color='grey',
        transform=ax.get_yaxis_transform())

# Value labels on top of bars
for bars, sds_list in [(bars1, tshirt_sds), (bars2, hoodie_sds)]:
    for bar, sd in zip(bars, sds_list):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + sd + 0.15,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f'{d} mm' for d in distances])
ax.set_xlabel('Subject-to-camera distance')
ax.set_ylabel('Mean absolute error (BPM)')
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(max(tshirt_means), max(hoodie_means)) + 5)

plt.savefig(OUTPUT_DIR / 'mae_by_dist_cloth.pdf')
plt.savefig(OUTPUT_DIR / 'mae_by_dist_cloth.png')
print(f"Saved to {OUTPUT_DIR}")
plt.show()