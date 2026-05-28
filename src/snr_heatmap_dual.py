"""
sternum_dual.py
==============================
Apply the SNR_local metric to ALL 28 cells of a single recording,
not just the 6 sternum cells.

Purpose: verify that AGE-class recordings don't have cardiac signal
hiding in non-sternum cells that we might have wrongly excluded.

If any non-sternum cell shows SNR_local >= 2, that's a real signal
in an unexpected location — possibly indicating wrong anatomical
assumption, ROI offset, or non-typical BCG distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import detrend, welch

from utils import (
    load_tap_info, load_movesense_hr, load_timestamps, load_fps,
    GRID_DIR, BASE,
)

# Settings — same as before
SETTLE_SEC       = 5.0
HR_TOLERANCE_BPM = 3.0
NPERSEG_SEC      = 30.0
NOVERLAP_FRAC    = 0.5
ROWS, COLS       = 7, 4
NBR_HZ           = 0.4
SNR_BG_LO_HZ     = 1.5
SNR_BG_HI_HZ     = 3.5
STERNUM_CELLS    = [5, 6, 9, 10, 13, 14]

OUT_DIR = BASE / "dual_snr_heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze(rec_id):
    print(f"\n{'='*70}\n  {rec_id}\n{'='*70}")

    tap = load_tap_info(rec_id, show_stats=True)
    hr = load_movesense_hr(rec_id, tap_info=tap, settle_sec=SETTLE_SEC,
                            show_stats=True)
    ts_df = load_timestamps(rec_id, show_stats=False)
    fs = load_fps(rec_id)

    grid_df = pd.read_csv(GRID_DIR / f"GRID_{rec_id}.csv")
    grid_frames = grid_df["frame"].values
    ts_values = ts_df.loc[grid_frames, "depth_ts"].values
    t_cam = (ts_values - ts_values[0]) / 1000.0

    keep = t_cam >= tap["cam_tap_sec"] + SETTLE_SEC
    grid_df = grid_df.iloc[keep].reset_index(drop=True)

    nperseg = int(NPERSEG_SEC * fs)
    if nperseg > len(grid_df):
        nperseg = len(grid_df) // 2
    noverlap = int(nperseg * NOVERLAP_FRAC)

    gt_hr_hz = hr["hr_mean"] / 60.0
    tol_hz = HR_TOLERANCE_BPM / 60.0

    snr_bg_grid = np.full((ROWS, COLS), np.nan)
    snr_local_grid = np.full((ROWS, COLS), np.nan)

    for c in range(ROWS * COLS):
        sig = grid_df[f"cell_{c}"].values.astype(float)
        zero_mask = sig == 0
        if zero_mask.any():
            idx = np.arange(len(sig))
            good = ~zero_mask
            if good.sum() < 2:
                continue
            sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])

        try:
            bp = list(range(int(10*fs), len(sig), int(10*fs)))
            sig_dt = detrend(sig, type="linear", bp=bp) if bp else \
                     detrend(sig, type="linear")
        except ValueError:
            sig_dt = detrend(sig, type="linear")

        if len(sig_dt) < nperseg:
            continue

        f, pxx = welch(sig_dt, fs=fs, nperseg=nperseg, noverlap=noverlap)

        hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
        peak = pxx[hr_mask].max() if hr_mask.any() else 0.0

        bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
        bg = max(np.median(pxx[bg_mask]), 1e-20)

        nbr_mask = ((f >= gt_hr_hz - NBR_HZ) & (f <= gt_hr_hz + NBR_HZ)
                    & ~hr_mask)
        local = max(np.median(pxx[nbr_mask]), 1e-20)

        row, col = c // COLS, c % COLS
        snr_bg_grid[row, col] = peak / bg
        snr_local_grid[row, col] = peak / local

    # Annotation: mark sternum cells with asterisks
    annot_bg = np.empty((ROWS, COLS), dtype=object)
    annot_local = np.empty((ROWS, COLS), dtype=object)
    for r in range(ROWS):
        for col in range(COLS):
            c = r * COLS + col
            star = "*" if c in STERNUM_CELLS else ""
            annot_bg[r, col] = f"c{c}{star}\n{snr_bg_grid[r, col]:.1f}"
            annot_local[r, col] = f"c{c}{star}\n{snr_local_grid[r, col]:.1f}"

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 11))
    sns.heatmap(snr_bg_grid, annot=annot_bg, fmt="", cmap="YlOrRd",
                ax=ax1, linewidths=0.5,
                cbar_kws={"label": "SNR_bg"},
                annot_kws={"fontsize": 8})
    ax1.set_title("SNR_bg (peak vs background)\n* = sternum cell")

    sns.heatmap(snr_local_grid, annot=annot_local, fmt="", cmap="YlGnBu",
                ax=ax2, linewidths=0.5,
                cbar_kws={"label": "SNR_local"},
                annot_kws={"fontsize": 8})
    ax2.set_title("SNR_local (is there a bump?)\n* = sternum cell")

    fig.suptitle(f"Full-grid local-prominence diagnostic — {rec_id}\n"
                  f"GT HR = {hr['hr_mean']:.1f} BPM", fontsize=12)

    plt.tight_layout()
    path = OUT_DIR / f"fullgrid_local_{rec_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path.name}")
    plt.close(fig)

    # Print top-10 cells by SNR_local across the whole grid
    flat = [(r*COLS+c, snr_local_grid[r,c], snr_bg_grid[r,c])
             for r in range(ROWS) for c in range(COLS)
             if not np.isnan(snr_local_grid[r,c])]
    flat.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 cells by SNR_local:")
    print(f"  {'cell':<8} {'(row,col)':<10} {'sternum?':<10} "
          f"{'SNR_local':>10} {'SNR_bg':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for cell_idx, local, bg in flat[:10]:
        r, col = cell_idx // COLS, cell_idx % COLS
        is_stern = "yes" if cell_idx in STERNUM_CELLS else "no"
        print(f"  cell_{cell_idx:<3} ({r},{col})      "
              f"{is_stern:<10} {local:>8.2f}  {bg:>6.2f}")

    # Verdict
    max_nonstern_local = max(
        (snr_local_grid[c // COLS, c % COLS]
         for c in range(ROWS * COLS) if c not in STERNUM_CELLS
         and not np.isnan(snr_local_grid[c // COLS, c % COLS])),
        default=0
    )
    max_stern_local = max(
        (snr_local_grid[c // COLS, c % COLS]
         for c in STERNUM_CELLS
         if not np.isnan(snr_local_grid[c // COLS, c % COLS])),
        default=0
    )

    print(f"\n  Best non-sternum SNR_local: {max_nonstern_local:.2f}")
    print(f"  Best sternum SNR_local:     {max_stern_local:.2f}")

    if max_nonstern_local > max_stern_local + 1.0:
        print(f"  ⚠ Non-sternum cell has notably higher SNR_local — "
              f"investigate whether sternum-cell assumption holds.")
    elif max_nonstern_local >= 2.0 and max_stern_local < 2.0:
        print(f"  ⚠ Cardiac may be in non-sternum location for this recording.")
    elif max_nonstern_local < 2.0 and max_stern_local < 2.0:
        print(f"  ✓ No cell anywhere in the grid shows detectable cardiac.")
    else:
        print(f"  ✓ Sternum cells contain the strongest cardiac signal.")


if __name__ == "__main__":
    # Run on AGE specifically, plus AVE as control
    for rec_id in ["GBA_1200_tshirt", "EST_1200_tshirt", "KGI_1200_tshirt", "MST_1200_tshirt", "MPA_1200_tshirt"]:
        analyze(rec_id)