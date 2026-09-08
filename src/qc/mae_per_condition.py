#!/usr/bin/env python3
"""Figure 4.2 — Per-condition MAE (distance x clothing). Data from Table 4.2."""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = r"C:\Projects\thesis\thesis_latex\images"
os.makedirs(OUTDIR, exist_ok=True)

# Table 4.2 — mean MAE per cell
distances = ['800 mm', '1200 mm', '1800 mm']
tshirt = [5.11, 3.45, 5.76]
hoodie = [6.05, 4.81, 5.10]

C_TSHIRT = "#5b7c99"   # slate blue-grey (matches per-subject fig)
C_HOODIE = "#b0c4d4"   # lighter tint

x = np.arange(len(distances))
w = 0.38

fig, ax = plt.subplots(figsize=(6.0, 3.2))
b1 = ax.bar(x - w/2, tshirt, w, label='T-shirt', color=C_TSHIRT,
            edgecolor='black', linewidth=0.4)
b2 = ax.bar(x + w/2, hoodie, w, label='Hoodie', color=C_HOODIE,
            edgecolor='black', linewidth=0.4)

# Highlight baseline (1200 mm T-shirt) with a red edge
b1[1].set_edgecolor('#a01818')
b1[1].set_linewidth(1.4)

# value labels
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7.5, color='0.25')

ax.set_xticks(x)
ax.set_xticklabels(distances)
ax.set_ylabel('Mean absolute error (BPM)')
ax.set_xlabel('Subject-to-camera distance')
ax.set_ylim(0, 7)
ax.legend(frameon=False, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', length=3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_per_condition_mae.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_per_condition_mae.png'), dpi=150)
print(f"saved to {OUTDIR}")