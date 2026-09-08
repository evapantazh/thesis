"""
sternum_dual_filtered.py
========================
Oracle-cell finder that measures SNR on the SAME signal the methodology
actually consumes (the FILTERED matrix), so the oracle is a fair test of
"if I fed RPCA the best cells, would it find the pulse?".

Key differences vs the original sternum_dual.py:
  1. Cardiac SNR is computed on FILTERED_{rec}.csv (bandpass 0.5-4 Hz, detrended,
     mean-centered) — identical to what selection_windowed sees.
  2. Respiration power is computed on the RAW GRID_{rec}.csv, because the
     bandpass in the FILTERED pipeline has already removed the 0.15-0.4 Hz band.
     A cell can look "clean" in FILTERED yet be respiration-dominated upstream;
     RPCA sees the upstream structure, so we judge respiration there.
  3. Exports oracle cell lists per recording to JSON:
        top5_snr, top14_snr, top5_resp_penalized, top14_resp_penalized
     so you can drop any of them into ORACLE_CELLS and compare error.

The FILTERED file is already trimmed (tap + settle) by the filter stage, so we
do NOT re-trim it. The RAW grid IS trimmed here to match, using cam_tap_sec.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend, welch

from utils import (
    load_tap_info, load_movesense_hr, load_timestamps, load_fps,
    GRID_DIR, FILTERED_DIR, BASE,
)

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────
SETTLE_SEC       = 5.0
HR_TOLERANCE_BPM = 3.0
NPERSEG_SEC      = 30.0
NOVERLAP_FRAC    = 0.5
ROWS, COLS       = 7, 4
N_CELLS          = ROWS * COLS

# Cardiac SNR bands (peak vs local background / vs broadband background)
NBR_HZ           = 0.4          # local-prominence neighbourhood half-width
SNR_BG_LO_HZ     = 1.5          # broadband background band
SNR_BG_HI_HZ     = 3.5

# Respiration band (measured on RAW grid, since FILTERED has bandpassed it out)
RESP_LO_HZ       = 0.15         # ~9 BPM
RESP_HI_HZ       = 0.40         # ~24 BPM

# How harshly to penalise respiration when ranking cells.
# penalized_score = snr_local / (1 + RESP_PENALTY_WEIGHT * resp_ratio)
# resp_ratio = resp_power / cardiac_peak_power  (both from RAW grid)
RESP_PENALTY_WEIGHT = 1.0

STERNUM_CELLS    = [5, 6, 9, 10, 13, 14]

# Oracle list sizes to export
ORACLE_SIZES     = [5, 14]

OUT_DIR    = BASE / "dual_snr_heatmap"
ORACLE_DIR = BASE / "oracle_cells"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────
def _interp_zeros(sig):
    """Replace zeros with linear interpolation (same idea as the filter stage)."""
    sig = sig.astype(float).copy()
    zero_mask = sig == 0
    if zero_mask.any():
        idx = np.arange(len(sig))
        good = ~zero_mask
        if good.sum() < 2:
            return None
        sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])
    return sig


def _piecewise_detrend(sig, fs):
    """Linear detrend with 10 s breakpoints, matching the diagnostic style."""
    try:
        bp = list(range(int(10 * fs), len(sig), int(10 * fs)))
        return detrend(sig, type="linear", bp=bp) if bp else detrend(sig, type="linear")
    except ValueError:
        return detrend(sig, type="linear")


def _welch(sig, fs, nperseg):
    nperseg = min(nperseg, len(sig))
    noverlap = int(nperseg * NOVERLAP_FRAC)
    return welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)


# ─────────────────────────────────────────────────────────────
# Per-recording analysis
# ─────────────────────────────────────────────────────────────
def analyze(rec_id):
    print(f"\n{'='*70}\n  {rec_id}\n{'='*70}")

    tap = load_tap_info(rec_id, show_stats=True)
    hr = load_movesense_hr(rec_id, tap_info=tap, settle_sec=SETTLE_SEC,
                           show_stats=True)
    fs = load_fps(rec_id)

    if tap is None or hr is None:
        print(f"  ✗ Skipping {rec_id} (missing tap or HR).")
        return None

    gt_hr_hz = hr["hr_mean"] / 60.0
    tol_hz   = HR_TOLERANCE_BPM / 60.0

    # ── Load FILTERED (already trimmed by the filter stage) ──────────────
    filt_path = FILTERED_DIR / f"FILTERED_{rec_id}.csv"
    if not filt_path.exists():
        print(f"  ✗ No FILTERED file: {filt_path}")
        return None
    filt_df = pd.read_csv(filt_path)

    # ── Load RAW grid and trim to tap+settle so respiration is measured ──
    #    on the same span as the filtered signal.
    grid_df = pd.read_csv(GRID_DIR / f"GRID_{rec_id}.csv")
    ts_df = load_timestamps(rec_id, show_stats=False)
    if ts_df is not None:
        grid_frames = grid_df["frame"].values
        ts_values = ts_df.loc[grid_frames, "depth_ts"].values
        t_cam = (ts_values - ts_values[0]) / 1000.0
        keep = t_cam >= tap["cam_tap_sec"] + SETTLE_SEC
        grid_df = grid_df.iloc[keep].reset_index(drop=True)
    else:
        print(f"  ⚠ No timestamps; respiration measured on untrimmed grid.")

    nperseg = int(NPERSEG_SEC * fs)

    # ── Per-cell metrics ─────────────────────────────────────────────────
    snr_local_grid = np.full((ROWS, COLS), np.nan)   # cardiac, from FILTERED
    snr_bg_grid    = np.full((ROWS, COLS), np.nan)    # cardiac, from FILTERED
    resp_ratio_grid = np.full((ROWS, COLS), np.nan)   # respiration, from RAW

    for c in range(N_CELLS):
        col = f"cell_{c}"
        row_i, col_i = c // COLS, c % COLS

        # ---- Cardiac SNR on FILTERED ----
        if col in filt_df.columns:
            fsig = filt_df[col].values.astype(float)
            if np.std(fsig) > 1e-12 and len(fsig) >= 16:
                # FILTERED is already detrended+centered; just PSD it.
                f, pxx = _welch(fsig, fs, nperseg)
                hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
                peak = pxx[hr_mask].max() if hr_mask.any() else 0.0

                bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
                bg = max(np.median(pxx[bg_mask]), 1e-20) if bg_mask.any() else 1e-20

                nbr_mask = ((f >= gt_hr_hz - NBR_HZ) & (f <= gt_hr_hz + NBR_HZ)
                            & ~hr_mask)
                local = max(np.median(pxx[nbr_mask]), 1e-20) if nbr_mask.any() else 1e-20

                snr_bg_grid[row_i, col_i]    = peak / bg
                snr_local_grid[row_i, col_i] = peak / local

        # ---- Respiration ratio on RAW grid ----
        if col in grid_df.columns:
            rsig = _interp_zeros(grid_df[col].values)
            if rsig is not None and len(rsig) >= 16:
                rsig = _piecewise_detrend(rsig, fs)
                fr, pxr = _welch(rsig, fs, nperseg)

                resp_mask = (fr >= RESP_LO_HZ) & (fr <= RESP_HI_HZ)
                resp_power = pxr[resp_mask].max() if resp_mask.any() else 0.0

                card_mask = (fr >= gt_hr_hz - tol_hz) & (fr <= gt_hr_hz + tol_hz)
                card_power = max(pxr[card_mask].max() if card_mask.any() else 0.0,
                                 1e-20)

                resp_ratio_grid[row_i, col_i] = resp_power / card_power

    # ── Build ranking tables ─────────────────────────────────────────────
    cells = []
    for c in range(N_CELLS):
        r_i, c_i = c // COLS, c % COLS
        snr_l = snr_local_grid[r_i, c_i]
        if np.isnan(snr_l):
            continue
        resp = resp_ratio_grid[r_i, c_i]
        resp = 0.0 if np.isnan(resp) else resp
        penalized = snr_l / (1.0 + RESP_PENALTY_WEIGHT * resp)
        cells.append({
            "cell": c,
            "snr_local": float(snr_l),
            "snr_bg": float(snr_bg_grid[r_i, c_i])
                       if not np.isnan(snr_bg_grid[r_i, c_i]) else None,
            "resp_ratio": float(resp),
            "penalized_score": float(penalized),
            "is_sternum": c in STERNUM_CELLS,
        })

    if not cells:
        print(f"  ✗ No usable cells for {rec_id}.")
        return None

    by_snr = sorted(cells, key=lambda d: d["snr_local"], reverse=True)
    by_pen = sorted(cells, key=lambda d: d["penalized_score"], reverse=True)

    oracle = {}
    for n in ORACLE_SIZES:
        oracle[f"top{n}_snr"] = [d["cell"] for d in by_snr[:n]]
        oracle[f"top{n}_resp_penalized"] = [d["cell"] for d in by_pen[:n]]

    # ── Console report ───────────────────────────────────────────────────
    print(f"\n  GT HR = {hr['hr_mean']:.1f} BPM   ({gt_hr_hz:.3f} Hz),  fs={fs:.2f}")
    print(f"\n  {'cell':<7}{'stern':<7}{'SNR_loc':>9}{'resp/card':>11}{'penalized':>11}")
    print(f"  {'-'*7}{'-'*7}{'-'*9}{'-'*11}{'-'*11}")
    for d in by_snr[:10]:
        print(f"  c{d['cell']:<6}{'yes' if d['is_sternum'] else 'no':<7}"
              f"{d['snr_local']:>9.2f}{d['resp_ratio']:>11.2f}"
              f"{d['penalized_score']:>11.2f}")

    print(f"\n  top5  by SNR        : {oracle['top5_snr']}")
    print(f"  top5  by penalized  : {oracle['top5_resp_penalized']}")
    print(f"  top14 by SNR        : {oracle['top14_snr']}")
    print(f"  top14 by penalized  : {oracle['top14_resp_penalized']}")

    # Did respiration penalty change the top-5 selection?
    if set(oracle["top5_snr"]) != set(oracle["top5_resp_penalized"]):
        swapped_out = set(oracle["top5_snr"]) - set(oracle["top5_resp_penalized"])
        print(f"  → respiration penalty dropped cells {sorted(swapped_out)} "
              f"from the top-5 (likely breathing-contaminated).")
    else:
        print(f"  → respiration penalty did not change the top-5.")

    # ── Heatmaps (cardiac SNR_local + respiration ratio) ─────────────────
    annot_snr = np.empty((ROWS, COLS), dtype=object)
    annot_resp = np.empty((ROWS, COLS), dtype=object)
    for r in range(ROWS):
        for col_i in range(COLS):
            c = r * COLS + col_i
            star = "*" if c in STERNUM_CELLS else ""
            sl = snr_local_grid[r, col_i]
            rr = resp_ratio_grid[r, col_i]
            annot_snr[r, col_i] = f"c{c}{star}\n{sl:.1f}" if not np.isnan(sl) else f"c{c}{star}\n—"
            annot_resp[r, col_i] = f"c{c}{star}\n{rr:.1f}" if not np.isnan(rr) else f"c{c}{star}\n—"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 11))
    sns.heatmap(snr_local_grid, annot=annot_snr, fmt="", cmap="YlGnBu",
                ax=ax1, linewidths=0.5, cbar_kws={"label": "SNR_local (FILTERED)"},
                annot_kws={"fontsize": 8})
    ax1.set_title("Cardiac SNR_local — measured on FILTERED\n* = sternum cell")

    sns.heatmap(resp_ratio_grid, annot=annot_resp, fmt="", cmap="OrRd",
                ax=ax2, linewidths=0.5, cbar_kws={"label": "resp/cardiac (RAW)"},
                annot_kws={"fontsize": 8})
    ax2.set_title("Respiration ratio — measured on RAW grid\n* = sternum cell")

    fig.suptitle(f"Oracle-cell diagnostic — {rec_id}   GT HR = {hr['hr_mean']:.1f} BPM",
                 fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / f"oracle_diag_{rec_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved heatmap → {path.name}")

    # ── Save oracle JSON ─────────────────────────────────────────────────
    out = {
        "rec_id": rec_id,
        "gt_hr_bpm": hr["hr_mean"],
        "gt_hr_hz": gt_hr_hz,
        "fs": fs,
        "resp_penalty_weight": RESP_PENALTY_WEIGHT,
        "oracle": oracle,
        "per_cell": by_snr,   # full table, SNR-sorted
    }
    oracle_path = ORACLE_DIR / f"oracle_{rec_id}.json"
    with open(oracle_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved oracle  → {oracle_path}")

    return out


if __name__ == "__main__":
    for rec_id in ["GBA_1200_tshirt",
                   ]:
        analyze(rec_id)