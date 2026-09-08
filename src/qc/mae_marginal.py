#!/usr/bin/env python3
"""Figure 4.x - Marginal effects of distance and clothing on MAE.
Data: distance/clothing marginal tables (Section 4.2)."""
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

BAR_COLOR = "#5b7c99"

# Marginal tables
dist_labels = ['800 mm', '1200 mm', '1800 mm']
dist_mae = [5.60, 4.13, 5.44]
cloth_labels = ['T-shirt', 'Hoodie']
cloth_mae = [4.76, 5.32]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                               gridspec_kw={'width_ratios': [3, 2.2]})

YMAX = 6.5

# --- Panel (a): distance ---
x1 = np.arange(len(dist_labels))
bars1 = ax1.bar(x1, dist_mae, color=BAR_COLOR, edgecolor='black',
                linewidth=0.4, width=0.6)
for b in bars1:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f'{b.get_height():.2f}', ha='center', va='bottom',
             fontsize=8, color='0.25')
ax1.set_xticks(x1); ax1.set_xticklabels(dist_labels)
ax1.set_ylabel('MAE (BPM)')
ax1.set_xlabel('(a) Distance')
ax1.set_ylim(0, YMAX)

# --- Panel (b): clothing ---
x2 = np.arange(len(cloth_labels))
bars2 = ax2.bar(x2, cloth_mae, color=BAR_COLOR, edgecolor='black',
                linewidth=0.4, width=0.6)
for b in bars2:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f'{b.get_height():.2f}', ha='center', va='bottom',
             fontsize=8, color='0.25')
ax2.set_xticks(x2); ax2.set_xticklabels(cloth_labels)
ax2.set_xlabel('(b) Clothing')
ax2.set_ylim(0, YMAX)
ax2.set_yticklabels([])

for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_mae_marginal.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_mae_marginal.png'), dpi=150)
print(f"saved to {OUTDIR}")