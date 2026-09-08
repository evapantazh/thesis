#!/usr/bin/env python3
"""Figure 4.x - Per-condition comparison of source-separation methods.
Data from Table 4.4."""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUTDIR = r"C:\Projects\thesis\thesis_latex\images"
os.makedirs(OUTDIR, exist_ok=True)

# Table 4.4 - condition x method MAE
conditions = ['800\nT-shirt', '800\nHoodie', '1200\nT-shirt',
              '1200\nHoodie', '1800\nT-shirt', '1800\nHoodie']
rpca = [5.11, 6.05, 3.45, 4.81, 5.76, 5.10]
pca  = [5.02, 5.17, 4.87, 4.71, 4.62, 4.77]
ica  = [4.92, 4.53, 4.48, 4.68, 5.07, 5.60]

C_RPCA = "#5b7c99"   # main slate blue (proposed)
C_PCA  = "#a9b8c4"   # light grey-blue
C_ICA  = "#d4cfc4"   # warm grey

x = np.arange(len(conditions))
w = 0.26

fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.bar(x - w, rpca, w, label='RPCA (proposed)', color=C_RPCA,
       edgecolor='black', linewidth=0.4)
ax.bar(x,     pca,  w, label='PCA', color=C_PCA,
       edgecolor='black', linewidth=0.4)
ax.bar(x + w, ica,  w, label='ICA', color=C_ICA,
       edgecolor='black', linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('MAE (BPM)')
ax.set_xlabel('Condition (distance in mm $\\times$ clothing)')
ax.set_ylim(0, 7)
ax.legend(frameon=False, loc='upper right', ncol=3,
          columnspacing=1.0, handlelength=1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', length=3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_method_comparison.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_method_comparison.png'), dpi=150)
print(f"saved to {OUTDIR}")