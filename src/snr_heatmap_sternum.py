"""
snr_heatmap_sternum.py
===============================
Restricted diagnostic: cardiac signal presence in STERNUM cells only.

The full-grid diagnostic revealed that AGE's "high SNR" cells were
edge/boundary artifacts (hair, jeans waistband, ROI jitter), not
real cardiac signal. This script restricts analysis to the 6 cells
that anatomically correspond to the sternum:

    cells [5, 6, 9, 10, 13, 14] = rows 1-3, cols 1-2

These are the cells used for tap detection and considered the most
likely to contain real cardiac BCG signal.

For each recording, generates:
  1. A focused 3x2 heatmap of sternum-cell SNR
  2. Spectra of all 6 sternum cells (log scale, 0-5 Hz) with GT HR
     and respiration harmonics overlaid

Compared to the full-grid diagnostic, this answers:
  - Is cardiac visible in the anatomically correct cells?
  - For AGE/GAX-type recordings, was the previous "high SNR" just
    ROI contamination?

Run from C:\\Projects\\thesis\\src\\:
    python snr_hratmap_sternum.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import detrend, welch

from utils import (
    load_tap_info,
    load_movesense_hr,
    load_timestamps,
    load_fps,
    GRID_DIR,
    BASE,
)


# ─────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────
REC_IDS = [
    "AVE_800_tshirt",      # known good — sanity check
    "AGE_800_tshirt",      # the test case
    "GAX_1200_tshirt",     # second test case
    "AVE_1800_hoodie",     # known good despite hard condition
]

# The 6 cells anatomically corresponding to the sternum
# Grid is 7 rows x 4 cols, cell_idx = row*4 + col
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]

# For mapping cells back to (row, col) for the 3x2 heatmap
# Sternum cells span: rows 1-3, cols 1-2
# Mapping:
#   cell_5  = (row 1, col 1) → heatmap position (0, 0)
#   cell_6  = (row 1, col 2) → heatmap position (0, 1)
#   cell_9  = (row 2, col 1) → heatmap position (1, 0)
#   cell_10 = (row 2, col 2) → heatmap position (1, 1)
#   cell_13 = (row 3, col 1) → heatmap position (2, 0)
#   cell_14 = (row 3, col 2) → heatmap position (2, 1)
STERNUM_HEATMAP_LAYOUT = np.array([
    [5,  6],
    [9,  10],
    [13, 14],
])

SETTLE_SEC          = 5.0
HR_TOLERANCE_BPM    = 3.0
NPERSEG_SEC         = 30.0
NOVERLAP_FRAC       = 0.5
ROWS, COLS          = 7, 4

# SNR background band
SNR_BG_LO_HZ = 1.5
SNR_BG_HI_HZ = 3.5

# Output directory for sternum-restricted diagnostics
DIAG_DIR = BASE / "sternum_snr_heatmap"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  CORE: process one recording
# ─────────────────────────────────────────────────────────────
def process_recording(rec_id):
    print(f"\n{'='*70}")
    print(f"  {rec_id}")
    print(f"{'='*70}")

    # ── Load all the data ──────────────────────────────────────
    tap = load_tap_info(rec_id, show_stats=True)
    if tap is None:
        return None

    hr = load_movesense_hr(rec_id, tap_info=tap,
                            settle_sec=SETTLE_SEC, show_stats=True)
    if hr is None:
        return None

    ts_df = load_timestamps(rec_id, show_stats=True)
    if ts_df is None:
        return None

    fs = load_fps(rec_id)

    grid_path = GRID_DIR / f"GRID_{rec_id}.csv"
    if not grid_path.exists():
        print(f"  ✗ Skipping: no grid file")
        return None
    grid_df = pd.read_csv(grid_path)
    print(f"✓ Grid loaded: {len(grid_df)} frames")

    # Build time axis from real timestamps
    grid_frames = grid_df["frame"].values
    try:
        ts_values = ts_df.loc[grid_frames, "depth_ts"].values
    except KeyError as e:
        print(f"  ✗ Some grid frames missing from timestamps")
        return None

    t_cam = (ts_values - ts_values[0]) / 1000.0

    # Trim to post-tap+settle
    trim_start_sec = tap["cam_tap_sec"] + SETTLE_SEC
    keep_mask = t_cam >= trim_start_sec
    if not keep_mask.any():
        return None
    grid_df = grid_df.iloc[keep_mask].reset_index(drop=True)
    t_cam = t_cam[keep_mask]
    print(f"✓ Analyzing {len(grid_df)} frames over "
          f"{t_cam[-1] - t_cam[0]:.1f}s (sternum cells only)")

    # ── Welch settings ─────────────────────────────────────────
    nperseg = int(NPERSEG_SEC * fs)
    if nperseg > len(grid_df):
        nperseg = len(grid_df) // 2
    noverlap = int(nperseg * NOVERLAP_FRAC)

    gt_hr_bpm = hr["hr_mean"]
    gt_hr_hz = gt_hr_bpm / 60.0
    tol_hz = HR_TOLERANCE_BPM / 60.0

    print(f"  GT HR: {gt_hr_bpm:.1f} BPM ({gt_hr_hz:.3f} Hz)")
    print(f"  Welch: nperseg={nperseg} ({nperseg/fs:.1f}s), "
          f"bin width≈{fs/nperseg*60:.2f} BPM")

    # ── Compute PSD per sternum cell ───────────────────────────
    cell_results = {}  # cell_idx -> dict with psd, snr, etc.
    freqs = None

    for c in STERNUM_CELLS:
        sig = grid_df[f"cell_{c}"].values.astype(float)

        # Zero-pixel interpolation
        zero_mask = sig == 0
        if zero_mask.any():
            idx = np.arange(len(sig))
            good = ~zero_mask
            if good.sum() < 2:
                cell_results[c] = None
                continue
            sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])

        # Piecewise linear detrend every 10s
        try:
            bp = list(range(int(10 * fs), len(sig), int(10 * fs)))
            sig_dt = detrend(sig, type="linear", bp=bp) if bp else \
                     detrend(sig, type="linear")
        except ValueError:
            sig_dt = detrend(sig, type="linear")

        # Welch PSD
        if len(sig_dt) < nperseg:
            cell_results[c] = None
            continue

        f, pxx = welch(sig_dt, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = f

        # Background SNR (peak power / median background)
        hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
        peak_power = pxx[hr_mask].max() if hr_mask.any() else 0.0

        bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
        bg_power = np.median(pxx[bg_mask]) if bg_mask.any() else 1e-20
        bg_power = max(bg_power, 1e-20)
        snr_bg = peak_power / bg_power

        # Local prominence SNR (peak / local neighborhood)
        # This checks whether there's an actual bump, not just relative quietness
        nbr_hz = 0.4
        nbr_mask = ((f >= gt_hr_hz - nbr_hz) & (f <= gt_hr_hz + nbr_hz)
                    & ~hr_mask)
        local_bg = np.median(pxx[nbr_mask]) if nbr_mask.any() else 1e-20
        local_bg = max(local_bg, 1e-20)
        snr_local = peak_power / local_bg

        cell_results[c] = {
            "psd": pxx,
            "snr_bg": snr_bg,
            "snr_local": snr_local,
            "peak_power": peak_power,
        }

    # ── Estimate respiration fundamental for plot annotations ──
    f_resp = None
    if freqs is not None:
        ref = next((r["psd"] for r in cell_results.values() if r is not None), None)
        if ref is not None:
            resp_band = (freqs >= 0.15) & (freqs <= 0.5)
            f_resp = freqs[resp_band][np.argmax(ref[resp_band])]
            print(f"✓ Respiration fundamental: {f_resp:.3f} Hz "
                  f"({f_resp*60:.1f} BPM)")

    # ── Print sternum-cell summary table ───────────────────────
    print(f"\n  Sternum cell SNRs:")
    print(f"  {'cell':<6} {'(row,col)':<10} {'SNR_bg':>8} {'SNR_local':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
    for c in STERNUM_CELLS:
        row, col = c // COLS, c % COLS
        if cell_results[c] is None:
            print(f"  cell_{c:<2}  ({row},{col})       —          —")
        else:
            r = cell_results[c]
            print(f"  cell_{c:<2}  ({row},{col})      "
                  f"{r['snr_bg']:>6.2f}    {r['snr_local']:>6.2f}")

    # Best sternum cell by each metric
    valid = {c: r for c, r in cell_results.items() if r is not None}
    if not valid:
        print(f"  ✗ No valid sternum cells")
        return None

    best_bg = max(valid.items(), key=lambda kv: kv[1]["snr_bg"])
    best_local = max(valid.items(), key=lambda kv: kv[1]["snr_local"])
    print(f"\n  Best by SNR_bg:    cell_{best_bg[0]} = {best_bg[1]['snr_bg']:.2f}")
    print(f"  Best by SNR_local: cell_{best_local[0]} = {best_local[1]['snr_local']:.2f}")

    # ── Plot 1: 3x2 sternum heatmap (both metrics) ─────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

    # Build value grids in (row, col) layout
    snr_bg_grid = np.zeros((3, 2))
    snr_local_grid = np.zeros((3, 2))
    for r in range(3):
        for col in range(2):
            c = STERNUM_HEATMAP_LAYOUT[r, col]
            if cell_results[c] is not None:
                snr_bg_grid[r, col] = cell_results[c]["snr_bg"]
                snr_local_grid[r, col] = cell_results[c]["snr_local"]

    # Annotations show cell number AND value
    annot_bg = np.empty((3, 2), dtype=object)
    annot_local = np.empty((3, 2), dtype=object)
    for r in range(3):
        for col in range(2):
            c = STERNUM_HEATMAP_LAYOUT[r, col]
            annot_bg[r, col] = f"cell_{c}\n{snr_bg_grid[r, col]:.1f}"
            annot_local[r, col] = f"cell_{c}\n{snr_local_grid[r, col]:.1f}"

    sns.heatmap(snr_bg_grid, annot=annot_bg, fmt="", cmap="YlOrRd",
                cbar_kws={"label": "SNR (peak / background)"},
                ax=ax1, linewidths=1, linecolor="gray",
                annot_kws={"fontsize": 11})
    ax1.set_title(f"Background-band SNR\n(peak vs noise floor)", fontsize=11)
    ax1.set_xlabel("Sternum column")
    ax1.set_ylabel("Sternum row")
    ax1.set_xticklabels(["L", "R"])
    ax1.set_yticklabels(["upper", "mid", "lower"], rotation=0)

    sns.heatmap(snr_local_grid, annot=annot_local, fmt="", cmap="YlGnBu",
                cbar_kws={"label": "SNR (peak / local neighborhood)"},
                ax=ax2, linewidths=1, linecolor="gray",
                annot_kws={"fontsize": 11})
    ax2.set_title(f"Local-prominence SNR\n(is there actually a bump?)",
                  fontsize=11)
    ax2.set_xlabel("Sternum column")
    ax2.set_xticklabels(["L", "R"])
    ax2.set_yticklabels(["upper", "mid", "lower"], rotation=0)

    fig.suptitle(
        f"Sternum cells only — {rec_id}\n"
        f"GT HR = {gt_hr_bpm:.1f} ± {hr['hr_std']:.1f} BPM "
        f"({gt_hr_hz:.2f} Hz)",
        fontsize=12
    )
    plt.tight_layout()

    heatmap_path = DIAG_DIR / f"sternum_heatmap_{rec_id}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"\n  Heatmap saved → {heatmap_path.name}")
    plt.close(fig)

    # ── Plot 2: spectra of all 6 sternum cells ────────────────
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)

    for r in range(3):
        for col in range(2):
            ax = axes[r, col]
            c = STERNUM_HEATMAP_LAYOUT[r, col]
            result = cell_results[c]

            if result is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            pxx = result["psd"]
            ax.semilogy(freqs, pxx, linewidth=0.8, color="black")

            # GT HR
            ax.axvline(gt_hr_hz, color="green", linestyle="--",
                       linewidth=1.8, alpha=0.8, label=f"GT HR")
            ax.axvspan(gt_hr_hz - tol_hz, gt_hr_hz + tol_hz,
                       alpha=0.15, color="green")

            # Respiration harmonics
            if f_resp is not None:
                for k in range(1, 7):
                    fk = k * f_resp
                    if fk < 5.0:
                        ax.axvline(fk, color="red", linestyle=":",
                                   alpha=0.5, linewidth=1)

            # Background band
            ax.axvspan(SNR_BG_LO_HZ, SNR_BG_HI_HZ, alpha=0.05, color="blue")

            grid_row, grid_col = c // COLS, c % COLS
            ax.set_title(
                f"cell_{c} (grid row {grid_row}, col {grid_col}) — "
                f"SNR_bg={result['snr_bg']:.1f}, "
                f"SNR_local={result['snr_local']:.1f}",
                fontsize=9
            )
            ax.set_xlim(0, 5)
            ax.set_ylabel("PSD (log)")
            ax.grid(True, which="both", alpha=0.3)
            if r == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)

    for col in range(2):
        axes[-1, col].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        f"Sternum cell spectra — {rec_id}\n"
        f"GT HR={gt_hr_bpm:.1f} BPM (green), "
        f"resp f={f_resp*60:.1f} BPM harmonics (red dots)",
        fontsize=11, y=1.00
    )
    plt.tight_layout()

    spectra_path = DIAG_DIR / f"sternum_spectra_{rec_id}.png"
    plt.savefig(spectra_path, dpi=150, bbox_inches="tight")
    print(f"  Spectra saved → {spectra_path.name}")
    plt.close(fig)

    # ── Return summary ─────────────────────────────────────────
    return {
        "rec_id":          rec_id,
        "gt_hr_bpm":       gt_hr_bpm,
        "f_resp_bpm":      f_resp * 60 if f_resp is not None else None,
        "best_sternum_cell":      int(best_bg[0]),
        "best_sternum_snr_bg":    float(best_bg[1]["snr_bg"]),
        "best_sternum_snr_local": float(best_local[1]["snr_local"]),
        "sternum_snr_bg": {f"cell_{c}": float(r["snr_bg"]) if r else None
                            for c, r in cell_results.items()},
        "sternum_snr_local": {f"cell_{c}": float(r["snr_local"]) if r else None
                                for c, r in cell_results.items()},
    }


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("STERNUM-RESTRICTED DIAGNOSTIC: Cardiac signal in sternum cells")
    print(f"Sternum cells: {STERNUM_CELLS}")
    print(f"Outputs → {DIAG_DIR}")

    summaries = []
    for rec_id in REC_IDS:
        try:
            result = process_recording(rec_id)
            if result is not None:
                summaries.append(result)
        except Exception as e:
            print(f"\n  ⚠ Error processing {rec_id}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────
    if summaries:
        print(f"\n{'='*75}")
        print(f"  SUMMARY — Best sternum cell per recording")
        print(f"{'='*75}")
        print(f"  {'rec_id':<25} {'GT HR':>7} {'f_resp':>7} "
              f"{'best cell':>10} {'SNR_bg':>8} {'SNR_local':>10}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*10}")
        for s in summaries:
            resp_str = f"{s['f_resp_bpm']:.1f}" if s['f_resp_bpm'] else "—"
            print(f"  {s['rec_id']:<25} "
                  f"{s['gt_hr_bpm']:>5.1f}   "
                  f"{resp_str:>5}    "
                  f"cell_{s['best_sternum_cell']:<5} "
                  f"{s['best_sternum_snr_bg']:>6.2f}    "
                  f"{s['best_sternum_snr_local']:>6.2f}")

        summary_path = DIAG_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n  Summary saved → {summary_path.name}")