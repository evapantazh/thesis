#!/usr/bin/env python3
"""
make_pulse_montage.py
=====================
Stacked montage of recovered pulse waveforms - one row per recording,
sharing a common time axis (EEG-trace style). Shows the range of signal
the depth-camera pipeline recovers across the dataset in a single figure.

For each recording it uses the highest-scoring window (cleanest) and plots
that window's pulse_signal. Rows are labelled with the rec_id on the left.

Usage:
  python make_pulse_montage.py                          # first N found
  python make_pulse_montage.py MGK_1200_tshirt GBA_1200_tshirt ...
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PULSE_DIR = Path(r"C:\Projects\thesis\data\PULSE_files_SQI")
OUT_DIR   = Path(r"C:\Projects\thesis\thesis_latex\images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which recordings to show, in order. Edit this list to pick your 10.
# If empty, the script auto-picks the first N_DEFAULT it finds.
RECORDINGS = [
    
    "MGK_800_tshirt",
    "IMA_800_tshirt",
    "GPA_800_hoodie",
    "AGE_1200_tshirt",
    "EST_1200_hoodie",
    "MST_1200_tshirt",
    "GAX_1200_hoodie",
    "GPA_1800_tshirt",
    "EST_1800_hoodie",
    "AGA_1800_tshirt"
]
N_DEFAULT = 10

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_PULSE = "#5b7c99"


def load_best_window(rec_id):
    """Return (t, sig) for the highest-scoring window, normalised, or None."""
    jp = PULSE_DIR / f"selection_windowed_{rec_id}.json"
    if not jp.exists():
        return None
    r = json.load(open(jp))
    cand = [w for w in r.get("per_window", []) if w.get("pulse_signal")]
    if not cand:
        return None
    best = max(cand, key=lambda w: w.get("score", -np.inf))
    sig = np.asarray(best["pulse_signal"], float)
    fs = float(best.get("pulse_fs", r.get("fs", 15.0)))
    sig = sig - np.mean(sig)
    s = np.std(sig)
    if s > 1e-12:
        sig = sig / s
    t = np.arange(len(sig)) / fs
    return t, sig


def main():
    if len(sys.argv) > 1:
        rec_ids = sys.argv[1:]
    elif RECORDINGS:
        rec_ids = RECORDINGS
    else:
        rec_ids = [p.stem.replace("selection_windowed_", "")
                   for p in sorted(PULSE_DIR.glob("selection_windowed_*.json"))
                   ][:N_DEFAULT]

    # load all
    rows = []
    for rec_id in rec_ids:
        res = load_best_window(rec_id)
        if res is None:
            print(f"  skip (no pulse): {rec_id}")
            continue
        rows.append((rec_id, *res))

    if not rows:
        print("Nothing to plot.")
        sys.exit(1)

    n = len(rows)
    # one row per recording, shared x-axis, tight vertical stacking
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 0.62 * n + 0.6),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (rec_id, t, sig) in zip(axes, rows):
        ax.plot(t, sig, color=C_PULSE, lw=0.8)
        # strip chrome: no y ticks, no top/right/left spines
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0)
        # rec_id label on the left, vertically centered
        ax.set_ylabel(rec_id, rotation=0, ha="right", va="center",
                      fontsize=8, labelpad=8)

    # only the bottom axis keeps the time axis
    axes[-1].set_xlabel("Time (s)")
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].tick_params(axis="x", length=3)

    fig.subplots_adjust(hspace=0.15)
    fig.tight_layout()
    out = OUT_DIR / "pulse_montage.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nMontage of {n} recordings -> {out}")


if __name__ == "__main__":
    main()