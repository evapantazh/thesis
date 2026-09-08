"""
sternum_dual_filtered_BATCH.py
==============================
Oracle-cell finder that measures SNR on the SAME signal the methodology
actually consumes (the FILTERED matrix), so the oracle is a fair test of
"if I fed RPCA the best cells, would it find the pulse?".

BATCH VERSION:
  - Auto-discovers all recordings with both GRID and FILTERED files
  - Runs analyze() on every recording
  - Produces a summary table + CSV across the entire dataset
  - Saves per-recording oracle JSONs to data/oracle_cells/

Key methodology vs the original sternum_dual.py:
  1. Cardiac SNR is computed on FILTERED_{rec}.csv (bandpass 0.5-4 Hz, detrended,
     mean-centered) — identical to what selection_windowed sees.
  2. Respiration power is computed on the RAW GRID_{rec}.csv, because the
     bandpass in the FILTERED pipeline has already removed the 0.15-0.4 Hz band.
  3. Exports oracle cell lists per recording to JSON:
        top5_snr, top14_snr, top5_resp_penalized, top14_resp_penalized
"""

import json
import os
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

# Cardiac SNR bands
NBR_HZ           = 0.4
SNR_BG_LO_HZ     = 1.5
SNR_BG_HI_HZ     = 3.5

# Respiration band
RESP_LO_HZ       = 0.15
RESP_HI_HZ       = 0.40

# Penalty weight
RESP_PENALTY_WEIGHT = 1.0

STERNUM_CELLS    = [5, 6, 9, 10, 13, 14]

# Oracle list sizes
ORACLE_SIZES     = [5, 14]

# Save plots only for selected recordings (to avoid 76 plot files)
# Set to None to save all, or list of rec_ids for selective saving
SAVE_PLOTS_FOR = None  # None = save all; or e.g. ["GBA_1200_tshirt", "AGE_800_tshirt"]

OUT_DIR    = BASE / "dual_snr_heatmap"
ORACLE_DIR = BASE / "oracle_cells"
SUMMARY_DIR = BASE / "oracle_cells" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────
def _interp_zeros(sig):
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
def analyze(rec_id, save_plot=True, verbose=True):
    if verbose:
        print(f"\n{'='*70}\n  {rec_id}\n{'='*70}")

    try:
        tap = load_tap_info(rec_id, show_stats=False)
        hr = load_movesense_hr(rec_id, tap_info=tap, settle_sec=SETTLE_SEC,
                               show_stats=False)
        fs = load_fps(rec_id)
    except Exception as e:
        if verbose:
            print(f"  ✗ Loading error: {e}")
        return None

    if tap is None or hr is None:
        if verbose:
            print(f"  ✗ Skipping (missing tap or HR).")
        return None

    gt_hr_hz = hr["hr_mean"] / 60.0
    tol_hz   = HR_TOLERANCE_BPM / 60.0

    # Load FILTERED
    filt_path = FILTERED_DIR / f"FILTERED_{rec_id}.csv"
    if not filt_path.exists():
        if verbose:
            print(f"  ✗ No FILTERED file")
        return None
    filt_df = pd.read_csv(filt_path)

    # Load RAW and trim
    grid_path = GRID_DIR / f"GRID_{rec_id}.csv"
    if not grid_path.exists():
        if verbose:
            print(f"  ✗ No GRID file")
        return None
    grid_df = pd.read_csv(grid_path)
    ts_df = load_timestamps(rec_id, show_stats=False)
    if ts_df is not None:
        try:
            grid_frames = grid_df["frame"].values
            ts_values = ts_df.loc[grid_frames, "depth_ts"].values
            t_cam = (ts_values - ts_values[0]) / 1000.0
            keep = t_cam >= tap["cam_tap_sec"] + SETTLE_SEC
            grid_df = grid_df.iloc[keep].reset_index(drop=True)
        except (KeyError, ValueError):
            pass

    nperseg = int(NPERSEG_SEC * fs)

    # Per-cell metrics
    snr_local_grid = np.full((ROWS, COLS), np.nan)
    snr_bg_grid    = np.full((ROWS, COLS), np.nan)
    resp_ratio_grid = np.full((ROWS, COLS), np.nan)

    for c in range(N_CELLS):
        col = f"cell_{c}"
        row_i, col_i = c // COLS, c % COLS

        # Cardiac SNR on FILTERED
        if col in filt_df.columns:
            fsig = filt_df[col].values.astype(float)
            if np.std(fsig) > 1e-12 and len(fsig) >= 16:
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

        # Respiration on RAW
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

    # Build ranking
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
        if verbose:
            print(f"  ✗ No usable cells.")
        return None

    by_snr = sorted(cells, key=lambda d: d["snr_local"], reverse=True)
    by_pen = sorted(cells, key=lambda d: d["penalized_score"], reverse=True)

    oracle = {}
    for n in ORACLE_SIZES:
        oracle[f"top{n}_snr"] = [d["cell"] for d in by_snr[:n]]
        oracle[f"top{n}_resp_penalized"] = [d["cell"] for d in by_pen[:n]]

    # Best metrics
    best_snr_cell = by_snr[0]["cell"]
    best_snr_value = by_snr[0]["snr_local"]
    best_pen_cell = by_pen[0]["cell"]
    best_pen_value = by_pen[0]["penalized_score"]

    sternum_snrs = [d["snr_local"] for d in cells if d["is_sternum"]]
    nonstern_snrs = [d["snr_local"] for d in cells if not d["is_sternum"]]
    best_sternum_snr = max(sternum_snrs) if sternum_snrs else 0.0
    best_nonstern_snr = max(nonstern_snrs) if nonstern_snrs else 0.0

    n_sternum_in_top5 = sum(1 for c in oracle["top5_snr"] if c in STERNUM_CELLS)
    n_sternum_in_top5_pen = sum(1 for c in oracle["top5_resp_penalized"]
                                  if c in STERNUM_CELLS)

    if verbose:
        print(f"  GT HR={hr['hr_mean']:.1f} BPM | "
              f"best SNR={best_snr_value:.2f} (cell {best_snr_cell}) | "
              f"sternum in top5: {n_sternum_in_top5}/5")

    # Save plot only if requested
    if save_plot:
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
                    ax=ax1, linewidths=0.5,
                    cbar_kws={"label": "SNR_local (FILTERED)"},
                    annot_kws={"fontsize": 8})
        ax1.set_title("Cardiac SNR_local — measured on FILTERED\n* = sternum cell")

        sns.heatmap(resp_ratio_grid, annot=annot_resp, fmt="", cmap="OrRd",
                    ax=ax2, linewidths=0.5,
                    cbar_kws={"label": "resp/cardiac (RAW)"},
                    annot_kws={"fontsize": 8})
        ax2.set_title("Respiration ratio — measured on RAW grid\n* = sternum cell")

        fig.suptitle(f"Oracle-cell diagnostic — {rec_id}   GT HR = {hr['hr_mean']:.1f} BPM",
                     fontsize=12)
        plt.tight_layout()
        path = OUT_DIR / f"oracle_diag_{rec_id}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Save oracle JSON
    out = {
        "rec_id": rec_id,
        "gt_hr_bpm": hr["hr_mean"],
        "gt_hr_hz": gt_hr_hz,
        "fs": fs,
        "resp_penalty_weight": RESP_PENALTY_WEIGHT,
        "oracle": oracle,
        "per_cell": by_snr,
        "summary_stats": {
            "best_snr_cell": int(best_snr_cell),
            "best_snr_value": float(best_snr_value),
            "best_penalized_cell": int(best_pen_cell),
            "best_penalized_value": float(best_pen_value),
            "best_sternum_snr": float(best_sternum_snr),
            "best_nonsternum_snr": float(best_nonstern_snr),
            "n_sternum_in_top5_snr": int(n_sternum_in_top5),
            "n_sternum_in_top5_penalized": int(n_sternum_in_top5_pen),
        }
    }
    oracle_path = ORACLE_DIR / f"oracle_{rec_id}.json"
    with open(oracle_path, "w") as f:
        json.dump(out, f, indent=2)

    return out


# ─────────────────────────────────────────────────────────────
# Main: batch over all recordings
# ─────────────────────────────────────────────────────────────
def list_all_recordings():
    """Find all recordings that have both GRID and FILTERED files."""
    rec_ids = []
    for grid_file in sorted(GRID_DIR.glob("GRID_*.csv")):
        rec_id = grid_file.stem.replace("GRID_", "")
        filt_file = FILTERED_DIR / f"FILTERED_{rec_id}.csv"
        if filt_file.exists():
            rec_ids.append(rec_id)
    return rec_ids


def main():
    rec_ids = list_all_recordings()
    print(f"{'='*70}")
    print(f"  BATCH ORACLE ANALYSIS")
    print(f"{'='*70}")
    print(f"Found {len(rec_ids)} recordings with both GRID and FILTERED files")
    print(f"Settings:")
    print(f"  HR tolerance:         {HR_TOLERANCE_BPM} BPM")
    print(f"  Resp penalty weight:  {RESP_PENALTY_WEIGHT}")
    print(f"  Saving plots for:     {'ALL' if SAVE_PLOTS_FOR is None else SAVE_PLOTS_FOR}")
    print()

    results = []
    for i, rec_id in enumerate(rec_ids, 1):
        save_plot = (SAVE_PLOTS_FOR is None) or (rec_id in SAVE_PLOTS_FOR)
        print(f"[{i:3d}/{len(rec_ids)}] {rec_id:<30}", end=" ")
        try:
            result = analyze(rec_id, save_plot=save_plot, verbose=False)
            if result is not None:
                stats = result["summary_stats"]
                print(f"GT={result['gt_hr_bpm']:5.1f} | "
                      f"best_snr={stats['best_snr_value']:5.2f} "
                      f"(c{stats['best_snr_cell']:<2}) | "
                      f"stern_in_top5={stats['n_sternum_in_top5_snr']}/5")
                results.append(result)
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {e}")

    # ─────────────────────────────────────────────────────────
    # Build summary table
    # ─────────────────────────────────────────────────────────
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    rows = []
    for r in results:
        parts = r["rec_id"].split("_")
        subject = parts[0]
        distance = parts[1] if len(parts) > 1 else ""
        clothing = parts[2] if len(parts) > 2 else ""
        s = r["summary_stats"]
        rows.append({
            "rec_id": r["rec_id"],
            "subject": subject,
            "distance": distance,
            "clothing": clothing,
            "gt_hr_bpm": r["gt_hr_bpm"],
            "best_snr_cell": s["best_snr_cell"],
            "best_snr_value": s["best_snr_value"],
            "best_sternum_snr": s["best_sternum_snr"],
            "best_nonsternum_snr": s["best_nonsternum_snr"],
            "n_sternum_in_top5_snr": s["n_sternum_in_top5_snr"],
            "n_sternum_in_top5_penalized": s["n_sternum_in_top5_penalized"],
            "top5_snr": r["oracle"]["top5_snr"],
            "top5_resp_penalized": r["oracle"]["top5_resp_penalized"],
        })

    df = pd.DataFrame(rows)

    # Aggregate statistics
    print(f"\nDetectability across {len(df)} recordings:")
    print(f"  best_snr_value > 5.0:  {(df['best_snr_value'] > 5.0).sum()} "
          f"({100*(df['best_snr_value'] > 5.0).mean():.1f}%) "
          f"-- strong cardiac signal exists somewhere")
    print(f"  best_snr_value > 3.0:  {(df['best_snr_value'] > 3.0).sum()} "
          f"({100*(df['best_snr_value'] > 3.0).mean():.1f}%) "
          f"-- detectable cardiac signal")
    print(f"  best_snr_value < 2.0:  {(df['best_snr_value'] < 2.0).sum()} "
          f"({100*(df['best_snr_value'] < 2.0).mean():.1f}%) "
          f"-- no detectable cardiac signal anywhere")

    print(f"\nSternum cell coverage:")
    print(f"  All 5 top-SNR cells are sternum: "
          f"{(df['n_sternum_in_top5_snr'] == 5).sum()}/{len(df)} recordings")
    print(f"  0 sternum cells in top-5:        "
          f"{(df['n_sternum_in_top5_snr'] == 0).sum()}/{len(df)} recordings")
    print(f"  Mean sternum cells in top-5:     "
          f"{df['n_sternum_in_top5_snr'].mean():.2f}")

    print(f"\nSternum vs non-sternum SNR:")
    print(f"  best_sternum > best_nonsternum:  "
          f"{(df['best_sternum_snr'] > df['best_nonsternum_snr']).sum()}/{len(df)}")
    print(f"  Mean best_sternum_snr:    {df['best_sternum_snr'].mean():.2f}")
    print(f"  Mean best_nonsternum_snr: {df['best_nonsternum_snr'].mean():.2f}")

    # Breakdown by condition
    print(f"\nBreakdown by condition (mean best_snr_value):")
    for cloth in sorted(df["clothing"].unique()):
        for dist in sorted(df["distance"].unique()):
            sub = df[(df["clothing"] == cloth) & (df["distance"] == dist)]
            if len(sub) == 0:
                continue
            print(f"  {cloth:<8} @ {dist}mm  (n={len(sub):2d})  "
                  f"mean_best_snr={sub['best_snr_value'].mean():5.2f}  "
                  f"detectable={(sub['best_snr_value'] > 3.0).sum()}/{len(sub)}")

    # Save CSV
    csv_path = SUMMARY_DIR / "oracle_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved: {csv_path}")

    # Save aggregate JSON
    agg_json = SUMMARY_DIR / "oracle_summary.json"
    with open(agg_json, "w") as f:
        json.dump({
            "n_recordings": len(df),
            "config": {
                "hr_tolerance_bpm": HR_TOLERANCE_BPM,
                "resp_penalty_weight": RESP_PENALTY_WEIGHT,
                "sternum_cells": STERNUM_CELLS,
                "oracle_sizes": ORACLE_SIZES,
            },
            "aggregate": {
                "mean_best_snr": float(df["best_snr_value"].mean()),
                "median_best_snr": float(df["best_snr_value"].median()),
                "n_strong_signal_gt5": int((df["best_snr_value"] > 5.0).sum()),
                "n_detectable_gt3": int((df["best_snr_value"] > 3.0).sum()),
                "n_undetectable_lt2": int((df["best_snr_value"] < 2.0).sum()),
                "mean_sternum_in_top5": float(df["n_sternum_in_top5_snr"].mean()),
            },
        }, f, indent=2)
    print(f"Aggregate JSON saved: {agg_json}")


if __name__ == "__main__":
    main()