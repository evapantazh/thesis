"""
batch_snr_sternum.py
============================
Run the sternum-restricted spectral diagnostic across the entire dataset.

For each recording, computes SNR_bg and SNR_local at the GT heart rate
in the 6 sternum cells. No per-recording plots are saved — instead, the
results are saved to a single CSV for analysis, and a population-level
summary plot is generated.

Outputs:
  data/diagnostics_batch/results.csv    : per-recording metrics
  data/diagnostics_batch/results.json   : same data as JSON
  data/diagnostics_batch/population_summary.png : histogram + scatter plots

Use this to identify which recordings are 'detectable' (SNR_local >= 2
in at least one sternum cell) vs which are not. The detectable subset
is what downstream methodology improvements can recover.

Run from C:\\Projects\\thesis\\src\\:
    python batch_snr_sternum.py
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import detrend, welch

from utils import (
    load_tap_info,
    load_movesense_hr,
    load_timestamps,
    load_fps,
    list_recordings,
    GRID_DIR,
    BASE,
)


# ─────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]

SETTLE_SEC          = 5.0
HR_TOLERANCE_BPM    = 3.0
NPERSEG_SEC         = 30.0
NOVERLAP_FRAC       = 0.5
COLS                = 4

SNR_BG_LO_HZ = 1.5
SNR_BG_HI_HZ = 3.5
NBR_HZ       = 0.4

# Thresholds for classification (used in summary plot)
SNR_LOCAL_DETECTABLE = 2.0
SNR_LOCAL_CLEAR      = 3.0

OUT_DIR = BASE / "batch_snr_sternum"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  CORE: compute metrics for one recording (no plots)
# ─────────────────────────────────────────────────────────────
def compute_metrics(rec_id):
    """
    Compute SNR_bg and SNR_local for each sternum cell.
    Returns a dict of results, or None on failure.
    """
    tap = load_tap_info(rec_id, show_stats=False)
    if tap is None:
        return None, "no_tap_info"

    hr = load_movesense_hr(rec_id, tap_info=tap,
                            settle_sec=SETTLE_SEC, show_stats=False)
    if hr is None:
        return None, "no_hr"

    ts_df = load_timestamps(rec_id, show_stats=False)
    if ts_df is None:
        return None, "no_timestamps"

    fs = load_fps(rec_id)

    grid_path = GRID_DIR / f"GRID_{rec_id}.csv"
    if not grid_path.exists():
        return None, "no_grid"
    grid_df = pd.read_csv(grid_path)

    # Build time axis
    grid_frames = grid_df["frame"].values
    try:
        ts_values = ts_df.loc[grid_frames, "depth_ts"].values
    except KeyError:
        return None, "frame_mismatch"
    t_cam = (ts_values - ts_values[0]) / 1000.0

    # Trim to post-tap+settle
    trim_start_sec = tap["cam_tap_sec"] + SETTLE_SEC
    keep_mask = t_cam >= trim_start_sec
    if not keep_mask.any():
        return None, "trim_too_aggressive"
    grid_df = grid_df.iloc[keep_mask].reset_index(drop=True)

    # Welch settings
    nperseg = int(NPERSEG_SEC * fs)
    if nperseg > len(grid_df):
        nperseg = len(grid_df) // 2
    if nperseg < 64:
        return None, "recording_too_short"
    noverlap = int(nperseg * NOVERLAP_FRAC)

    gt_hr_bpm = hr["hr_mean"]
    gt_hr_hz = gt_hr_bpm / 60.0
    tol_hz = HR_TOLERANCE_BPM / 60.0

    # Per-cell SNR
    per_cell_bg = {}
    per_cell_local = {}
    f_resp_hz = None

    for c in STERNUM_CELLS:
        sig = grid_df[f"cell_{c}"].values.astype(float)

        # Zero-pixel interpolation
        zero_mask = sig == 0
        if zero_mask.any():
            idx = np.arange(len(sig))
            good = ~zero_mask
            if good.sum() < 2:
                per_cell_bg[c] = np.nan
                per_cell_local[c] = np.nan
                continue
            sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])

        # Piecewise linear detrend every 10s
        try:
            bp = list(range(int(10 * fs), len(sig), int(10 * fs)))
            sig_dt = detrend(sig, type="linear", bp=bp) if bp else \
                     detrend(sig, type="linear")
        except ValueError:
            sig_dt = detrend(sig, type="linear")

        if len(sig_dt) < nperseg:
            per_cell_bg[c] = np.nan
            per_cell_local[c] = np.nan
            continue

        f, pxx = welch(sig_dt, fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Estimate respiration fundamental from the first cell
        if f_resp_hz is None:
            resp_band = (f >= 0.15) & (f <= 0.5)
            if resp_band.any():
                f_resp_hz = float(f[resp_band][np.argmax(pxx[resp_band])])

        # Peak at GT HR
        hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
        peak_power = float(pxx[hr_mask].max()) if hr_mask.any() else 0.0

        # SNR_bg
        bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
        bg_power = float(np.median(pxx[bg_mask])) if bg_mask.any() else 1e-20
        bg_power = max(bg_power, 1e-20)
        per_cell_bg[c] = peak_power / bg_power

        # SNR_local
        nbr_mask = ((f >= gt_hr_hz - NBR_HZ) & (f <= gt_hr_hz + NBR_HZ)
                    & ~hr_mask)
        local_bg = float(np.median(pxx[nbr_mask])) if nbr_mask.any() else 1e-20
        local_bg = max(local_bg, 1e-20)
        per_cell_local[c] = peak_power / local_bg

    # Parse rec_id into subject / distance / clothing
    parts = rec_id.split("_")
    subject = parts[0] if len(parts) > 0 else ""
    distance = parts[1] if len(parts) > 1 else ""
    clothing = parts[2] if len(parts) > 2 else ""

    # Best of the 6 cells by each metric
    valid_bg = {c: v for c, v in per_cell_bg.items() if not np.isnan(v)}
    valid_local = {c: v for c, v in per_cell_local.items() if not np.isnan(v)}

    if not valid_local:
        return None, "all_cells_invalid"

    best_bg_cell, best_bg_val = max(valid_bg.items(), key=lambda kv: kv[1])
    best_local_cell, best_local_val = max(valid_local.items(),
                                            key=lambda kv: kv[1])

    result = {
        "rec_id":          rec_id,
        "subject":         subject,
        "distance":        distance,
        "clothing":        clothing,
        "gt_hr_bpm":       float(gt_hr_bpm),
        "gt_hr_std":       float(hr["hr_std"]),
        "f_resp_bpm":      f_resp_hz * 60 if f_resp_hz is not None else np.nan,
        "fs":              float(fs),
        "n_frames_used":   int(len(grid_df)),
        "best_bg_cell":    int(best_bg_cell),
        "best_bg":         float(best_bg_val),
        "best_local_cell": int(best_local_cell),
        "best_local":      float(best_local_val),
        "median_local":    float(np.median(list(valid_local.values()))),
        # Per-cell values for later analysis
        **{f"snr_bg_cell_{c}": float(per_cell_bg[c])
            if not np.isnan(per_cell_bg[c]) else np.nan
            for c in STERNUM_CELLS},
        **{f"snr_local_cell_{c}": float(per_cell_local[c])
            if not np.isnan(per_cell_local[c]) else np.nan
            for c in STERNUM_CELLS},
    }
    return result, "ok"


# ─────────────────────────────────────────────────────────────
#  POPULATION SUMMARY PLOT
# ─────────────────────────────────────────────────────────────
def plot_population_summary(df):
    """
    4-panel summary figure:
      A. Histogram of best SNR_local across all recordings
      B. Scatter: best_local vs best_bg
      C. Best SNR_local by subject (boxplot)
      D. Best SNR_local by distance (boxplot)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A. Histogram of best SNR_local
    ax = axes[0, 0]
    ax.hist(df["best_local"].dropna(), bins=25,
            color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(SNR_LOCAL_DETECTABLE, color="orange", linestyle="--",
                linewidth=2, label=f"Detectable threshold ({SNR_LOCAL_DETECTABLE})")
    ax.axvline(SNR_LOCAL_CLEAR, color="green", linestyle="--",
                linewidth=2, label=f"Clear-peak threshold ({SNR_LOCAL_CLEAR})")
    ax.set_xlabel("Best SNR_local across sternum cells")
    ax.set_ylabel("Number of recordings")
    ax.set_title(f"A. Distribution of cardiac signal strength "
                  f"(N = {len(df)} recordings)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # B. Scatter: best_local vs best_bg
    ax = axes[0, 1]
    ax.scatter(df["best_bg"], df["best_local"],
                alpha=0.6, s=40, color="steelblue", edgecolor="black")
    ax.axhline(SNR_LOCAL_DETECTABLE, color="orange", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.axhline(SNR_LOCAL_CLEAR, color="green", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Best SNR_bg (peak vs noise floor)")
    ax.set_ylabel("Best SNR_local (is there a bump?)")
    ax.set_title("B. SNR_bg vs SNR_local — disagreement = false alarm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # C. Best SNR_local by subject
    ax = axes[1, 0]
    subjects = sorted(df["subject"].unique())
    data_by_subj = [df.loc[df["subject"] == s, "best_local"].dropna().values
                     for s in subjects]
    bp = ax.boxplot(data_by_subj, labels=subjects, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.axhline(SNR_LOCAL_DETECTABLE, color="orange", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.axhline(SNR_LOCAL_CLEAR, color="green", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Best SNR_local")
    ax.set_title("C. Cardiac detectability by subject")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    # D. Best SNR_local by distance × clothing
    ax = axes[1, 1]
    df["condition"] = df["distance"] + "_" + df["clothing"]
    conditions = sorted(df["condition"].unique())
    data_by_cond = [df.loc[df["condition"] == c, "best_local"].dropna().values
                     for c in conditions]
    bp = ax.boxplot(data_by_cond, labels=conditions, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("seagreen")
        patch.set_alpha(0.6)
    ax.axhline(SNR_LOCAL_DETECTABLE, color="orange", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.axhline(SNR_LOCAL_CLEAR, color="green", linestyle="--",
                linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Condition (distance_clothing)")
    ax.set_ylabel("Best SNR_local")
    ax.set_title("D. Cardiac detectability by condition")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Population-level cardiac signal detectability "
                  "(sternum cells, post-tap, raw)", fontsize=13, y=1.00)
    plt.tight_layout()

    plot_path = OUT_DIR / "population_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Summary plot saved → {plot_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("BATCH STERNUM DIAGNOSTIC")
    print(f"Outputs → {OUT_DIR}\n")

    rec_ids = list_recordings(verify_files=True)
    print(f"\nProcessing {len(rec_ids)} recordings...\n")

    results = []
    failures = []
    t_start = time.time()

    for i, rec_id in enumerate(rec_ids, 1):
        t0 = time.time()
        try:
            result, status = compute_metrics(rec_id)
        except Exception as e:
            result, status = None, f"exception: {type(e).__name__}: {e}"

        elapsed = time.time() - t0
        if result is None:
            failures.append((rec_id, status))
            print(f"  [{i:3d}/{len(rec_ids)}] {rec_id:<30}  "
                  f"FAIL ({status})  [{elapsed:.1f}s]")
        else:
            results.append(result)
            mark = ("●●●" if result["best_local"] >= SNR_LOCAL_CLEAR
                    else "●●○" if result["best_local"] >= SNR_LOCAL_DETECTABLE
                    else "●○○")
            print(f"  [{i:3d}/{len(rec_ids)}] {rec_id:<30}  "
                  f"{mark}  SNR_local={result['best_local']:5.2f}  "
                  f"[{elapsed:.1f}s]")

    t_total = time.time() - t_start
    print(f"\n  Done in {t_total:.1f}s "
          f"({t_total/max(len(rec_ids),1):.1f}s per recording).")

    if not results:
        print("\n  No successful recordings — nothing to save.")
        exit(1)

    # ── Save results ──────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path.name}")

    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON saved → {json_path.name}")

    if failures:
        fail_path = OUT_DIR / "failures.txt"
        with open(fail_path, "w") as f:
            for rec_id, status in failures:
                f.write(f"{rec_id}\t{status}\n")
        print(f"  Failures ({len(failures)}) → {fail_path.name}")

    # ── Headline numbers ──────────────────────────────────────
    n_total = len(df)
    n_detectable = int((df["best_local"] >= SNR_LOCAL_DETECTABLE).sum())
    n_clear = int((df["best_local"] >= SNR_LOCAL_CLEAR).sum())

    print(f"\n{'='*70}")
    print(f"  HEADLINE")
    print(f"{'='*70}")
    print(f"  Total recordings processed:  {n_total}")
    print(f"  Failures:                    {len(failures)}")
    print(f"  Clear peak (SNR_local ≥ {SNR_LOCAL_CLEAR}): "
          f"{n_clear} ({100*n_clear/n_total:.0f}%)")
    print(f"  Detectable  (SNR_local ≥ {SNR_LOCAL_DETECTABLE}): "
          f"{n_detectable} ({100*n_detectable/n_total:.0f}%)")
    print(f"  Not detectable:              "
          f"{n_total - n_detectable} ({100*(n_total-n_detectable)/n_total:.0f}%)")

    # ── Breakdown by subject ─────────────────────────────────
    print(f"\n  By subject (clear / detectable / total):")
    for subj in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subj]
        c = int((sub["best_local"] >= SNR_LOCAL_CLEAR).sum())
        d = int((sub["best_local"] >= SNR_LOCAL_DETECTABLE).sum())
        t = len(sub)
        print(f"    {subj}:  {c} / {d} / {t}")

    # ── Breakdown by condition ───────────────────────────────
    df["condition"] = df["distance"] + "_" + df["clothing"]
    print(f"\n  By condition (clear / detectable / total):")
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        c = int((sub["best_local"] >= SNR_LOCAL_CLEAR).sum())
        d = int((sub["best_local"] >= SNR_LOCAL_DETECTABLE).sum())
        t = len(sub)
        print(f"    {cond}:  {c} / {d} / {t}")

    # ── Generate summary plot ─────────────────────────────────
    plot_population_summary(df)

    print(f"\n{'='*70}")
    print(f"  All results in: {OUT_DIR}")
    print(f"{'='*70}")