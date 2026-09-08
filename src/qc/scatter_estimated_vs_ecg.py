#!/usr/bin/env python3
"""
scatter_estimated_vs_ecg.py
===========================
Scatter of estimated HR (depth camera) vs ECG ground truth for all
recordings, with the identity line y = x and Pearson r annotated.
Reads the batch summary CSV produced by 07_batch.py.

Usage:
  python scatter_estimated_vs_ecg.py
  python scatter_estimated_vs_ecg.py PATH/TO/summary_XXXX.csv
"""

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SUMMARY_DIR = Path(r"C:\Projects\thesis\data\PULSE_files_SQI\summaries")
OUT_DIR     = Path(r"C:\Projects\thesis\thesis_latex\images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# colour per distance — three shades of blue (light -> dark), matching #2b5c8f
C_DIST = {
    "800":  "#9dc3e6",   # light blue
    "1200": "#4a89c0",   # mid blue
    "1800": "#2b5c8f",   # dark blue (user's raw-signal colour)
}
M_DIST = {"800": "o", "1200": "o", "1800": "o"}


def find_summary():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    files = sorted(SUMMARY_DIR.glob("summary_*.csv"))
    if not files:
        print(f"No summary CSV in {SUMMARY_DIR}")
        sys.exit(1)
    return files[-1]   # most recent


def main():
    csv = find_summary()
    df = pd.read_csv(csv)
    df = df[df["status"] == "ok"].copy()
    df = df.dropna(subset=["bpm_smoothed", "gt_bpm"])
    df["dist"] = df["dist"].astype(str)

    gt = df["gt_bpm"].to_numpy(float)
    est = df["bpm_smoothed"].to_numpy(float)

    # Pearson r
    r = np.corrcoef(gt, est)[0, 1]
    mae = np.mean(np.abs(est - gt))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # identity line y = x
    lo = min(gt.min(), est.min()) - 3
    hi = max(gt.max(), est.max()) + 3
    ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0,
            zorder=1, label="identity ($y=x$)")

    # points by distance
    for dist in ["800", "1200", "1800"]:
        m = df["dist"] == dist
        ax.scatter(gt[m], est[m], s=42,
                   color=C_DIST.get(dist, "#888"),
                   marker=M_DIST.get(dist, "o"),
                   edgecolor="black", linewidth=0.4, alpha=0.85,
                   label=f"{dist} mm", zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ECG ground-truth HR (BPM)")
    ax.set_ylabel("Estimated HR (BPM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)

    # annotate r and MAE
    ax.text(0.04, 0.96, f"$r = {r:.2f}$\nMAE $= {mae:.2f}$ BPM\n$n = {len(df)}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.8", linewidth=0.6))

    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out = OUT_DIR / "scatter_estimated_vs_ecg.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"r = {r:.3f}, MAE = {mae:.2f}, n = {len(df)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()