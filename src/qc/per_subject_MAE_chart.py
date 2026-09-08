#!/usr/bin/env python3
"""Figure 4.x — Per-subject MAE across all conditions. Data from Table 4.7."""
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

# Output directory
OUTDIR = r"C:\Projects\thesis\thesis_latex\images"
os.makedirs(OUTDIR, exist_ok=True)

# Table 4.7 — subject, MAE (all conditions)
data = [
    ("MGK", 0.89), ("GPA", 2.17), ("GVA", 3.70), ("MST", 4.30),
    ("MPA", 4.43), ("AVE", 4.44), ("IMA", 5.05), ("AGA", 5.38),
    ("GAX", 5.48), ("GBA", 5.90), ("AGE", 7.25), ("EST", 7.91),
    ("KGI", 8.47),
]
subjects = [d[0] for d in data]
mae = np.array([d[1] for d in data])

BAR_COLOR = "#5b7c99"  # neutral slate blue-grey

fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(subjects))
ax.bar(x, mae, color=BAR_COLOR, edgecolor='black', linewidth=0.4, width=0.72)

cohort_mae = 5.04
ax.axhline(cohort_mae, color='0.35', linestyle='--', linewidth=0.9, zorder=0)
ax.text(len(subjects) - 0.4, cohort_mae + 0.12, f'cohort MAE = {cohort_mae:.2f}',
        ha='right', va='bottom', fontsize=8, color='0.35')

ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylabel('MAE (BPM)')
ax.set_xlabel('Subject')
ax.set_ylim(0, 9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', length=3)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig_per_subject_mae.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fig_per_subject_mae.png'), dpi=150)
print(f"saved to {OUTDIR}")