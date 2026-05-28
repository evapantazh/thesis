"""
snr_heatmap_fullgrid.py
======================
Diagnostic to verify whether the cardiac signal is actually present in the
raw cell signals, BEFORE any bandpass filter or RPCA processing.

For each recording, produces two plots:

  1. Cardiac SNR heatmap (7x4) — where on the chest is cardiac strongest?
  2. Spectra of top-4 cells (log scale, 0–5 Hz) with GT HR and respiration
     harmonics overlaid — is the cardiac peak actually visible at GT HR?

Verdict logic:
  - Top cells show clear peak at GT HR  → signal present, scoring is the
    bottleneck. Respiration-aware scoring should fix the pipeline.
  - No peak at GT HR even in best cells → front end needs work (filter,
    ROI, or hardware-limited).

Run from C:\\Projects\\thesis\\src\\:
    python snr_heatmap_fullgrid.py
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
    "AGE_800_tshirt",      # known failure, easy condition
    "GAX_1200_tshirt",     # known failure, harder condition
    "AVE_1800_hoodie",     # known good despite hard condition
    "IMA_1200_tshirt",
]

SETTLE_SEC          = 5.0    # seconds after tap to ignore
HR_TOLERANCE_BPM    = 3.0    # ± window around GT HR for SNR calculation
NPERSEG_SEC         = 30.0   # Welch segment length in seconds
NOVERLAP_FRAC       = 0.5    # Welch overlap fraction
ROWS, COLS          = 7, 4   # grid dimensions

# SNR background band — used to normalize peak power
# Choose a band that excludes the very-low-frequency respiration
# fundamental and the cardiac region itself
SNR_BG_LO_HZ = 1.5    # 90 BPM
SNR_BG_HI_HZ = 3.5    # 210 BPM

DIAG_DIR = BASE / "fullgrid_snr_heatmap"
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
        print(f"  ✗ Skipping {rec_id}: no tap info")
        return None

    hr = load_movesense_hr(rec_id, tap_info=tap,
                            settle_sec=SETTLE_SEC, show_stats=True)
    if hr is None:
        print(f"  ✗ Skipping {rec_id}: no HR data")
        return None

    ts_df = load_timestamps(rec_id, show_stats=True)
    if ts_df is None:
        print(f"  ✗ Skipping {rec_id}: no timestamps")
        return None

    fs = load_fps(rec_id)

    # Load raw grid (before filter, before RPCA)
    grid_path = GRID_DIR / f"GRID_{rec_id}.csv"
    if not grid_path.exists():
        print(f"  ✗ Skipping {rec_id}: no grid file at {grid_path}")
        return None
    grid_df = pd.read_csv(grid_path)
    print(f"✓ Grid loaded: {len(grid_df)} frames, {ROWS*COLS} cells")

    # ── Build the camera time axis using real timestamps ───────
    # The grid frames may not be contiguous (some were dropped); use the
    # depth_ts for each grid frame to build a proper time axis
    grid_frames = grid_df["frame"].values
    try:
        ts_values = ts_df.loc[grid_frames, "depth_ts"].values
    except KeyError as e:
        print(f"  ✗ Some grid frames missing from timestamps: {e}")
        return None

    # Convert to seconds from recording start
    t_cam = (ts_values - ts_values[0]) / 1000.0
    print(f"✓ Time axis: 0 to {t_cam[-1]:.1f}s")

    # ── Trim to post-tap+settle ────────────────────────────────
    trim_start_sec = tap["cam_tap_sec"] + SETTLE_SEC
    keep_mask = t_cam >= trim_start_sec
    n_trimmed = int(np.sum(~keep_mask))
    if not keep_mask.any():
        print(f"  ✗ Skipping {rec_id}: all frames before tap+settle")
        return None
    grid_df = grid_df.iloc[keep_mask].reset_index(drop=True)
    t_cam = t_cam[keep_mask]
    print(f"✓ Trimmed first {n_trimmed} frames ({trim_start_sec:.1f}s pre-cardiac)")
    print(f"  Analyzing {len(grid_df)} frames over {t_cam[-1] - t_cam[0]:.1f}s")

    # ── Compute PSD per cell ───────────────────────────────────
    nperseg = int(NPERSEG_SEC * fs)
    if nperseg > len(grid_df):
        nperseg = len(grid_df) // 2
        print(f"  ⚠ Recording short; reduced nperseg to {nperseg}")
    noverlap = int(nperseg * NOVERLAP_FRAC)

    print(f"✓ Welch settings: nperseg={nperseg} ({nperseg/fs:.1f}s), "
          f"noverlap={noverlap}, bin width≈{fs/nperseg*60:.2f} BPM")

    gt_hr_bpm = hr["hr_mean"]
    gt_hr_hz = gt_hr_bpm / 60.0
    tol_hz = HR_TOLERANCE_BPM / 60.0

    cardiac_snr = np.zeros(ROWS * COLS)
    cardiac_power = np.zeros(ROWS * COLS)
    all_psds = []
    freqs = None

    for c in range(ROWS * COLS):
        sig = grid_df[f"cell_{c}"].values.astype(float)

        # Zero-pixel interpolation (consistent with 03_filter.py)
        zero_mask = sig == 0
        if zero_mask.any():
            idx = np.arange(len(sig))
            good = ~zero_mask
            if good.sum() < 2:
                cardiac_snr[c] = 0.0
                cardiac_power[c] = 0.0
                all_psds.append(None)
                continue
            sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])

        # Piecewise linear detrend every 10s to remove slow drift
        try:
            bp = list(range(int(10 * fs), len(sig), int(10 * fs)))
            sig_dt = detrend(sig, type="linear", bp=bp) if bp else \
                     detrend(sig, type="linear")
        except ValueError:
            sig_dt = detrend(sig, type="linear")

        # Welch PSD
        if len(sig_dt) < nperseg:
            all_psds.append(None)
            cardiac_snr[c] = 0.0
            cardiac_power[c] = 0.0
            continue

        f, pxx = welch(sig_dt, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = f
        all_psds.append(pxx)

        # Power at GT HR ± tolerance
        hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
        peak_power = pxx[hr_mask].max() if hr_mask.any() else 0.0

        # Background power: median in the SNR background band,
        # excluding the GT HR region itself
        bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
        bg_power = np.median(pxx[bg_mask]) if bg_mask.any() else 1e-20
        bg_power = max(bg_power, 1e-20)  # avoid div by zero

        cardiac_snr[c] = peak_power / bg_power
        cardiac_power[c] = peak_power

    # ── Estimate respiration fundamental (for plot annotations) ─
    if freqs is not None:
        # Use the first valid PSD to estimate respiration
        ref_psd = next((p for p in all_psds if p is not None), None)
        if ref_psd is not None:
            resp_band = (freqs >= 0.15) & (freqs <= 0.5)
            f_resp = freqs[resp_band][np.argmax(ref_psd[resp_band])]
            print(f"✓ Respiration fundamental: {f_resp:.3f} Hz "
                  f"({f_resp*60:.1f} BPM)")
        else:
            f_resp = None
    else:
        f_resp = None

    # ── Identify top cells by SNR ──────────────────────────────
    best_cells = np.argsort(cardiac_snr)[::-1][:4]
    print(f"\n  Cardiac SNR ranking (top 4):")
    for rank, c in enumerate(best_cells):
        row, col = c // COLS, c % COLS
        print(f"    {rank+1}. cell_{c:2d} (row {row}, col {col}): "
              f"SNR = {cardiac_snr[c]:6.2f}")

    print(f"\n  Best cell SNR: {cardiac_snr[best_cells[0]]:.2f}")
    print(f"  Median cell SNR: {np.median(cardiac_snr):.2f}")
    print(f"  Worst cell SNR: {cardiac_snr[best_cells[-1]]:.2f}")

    # ── Plot 1: Cardiac SNR heatmap ────────────────────────────
    grid_snr = cardiac_snr.reshape(ROWS, COLS)
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(
        grid_snr, annot=True, fmt=".1f",
        cmap="YlOrRd", cbar_kws={"label": "Cardiac SNR (peak/background)"},
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_title(
        f"Cardiac SNR per cell — {rec_id}\n"
        f"GT HR = {gt_hr_bpm:.1f} ± {hr['hr_std']:.1f} BPM "
        f"({gt_hr_hz:.2f} Hz)",
        fontsize=11
    )
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row (0=top)")
    plt.tight_layout()

    heatmap_path = DIAG_DIR / f"snr_heatmap_{rec_id}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"\n  Heatmap saved → {heatmap_path.name}")
    plt.close(fig)

    # ── Plot 2: Spectra of top-4 cells ─────────────────────────
    fig, axes = plt.subplots(len(best_cells), 1,
                              figsize=(14, 2.8 * len(best_cells)),
                              sharex=True)
    if len(best_cells) == 1:
        axes = [axes]

    for ax, c in zip(axes, best_cells):
        pxx = all_psds[c]
        if pxx is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        # Plot the spectrum (log y)
        ax.semilogy(freqs, pxx, linewidth=0.8, color="black")

        # GT HR marker (green)
        ax.axvline(gt_hr_hz, color="green", linestyle="--",
                   linewidth=1.8, alpha=0.8,
                   label=f"GT HR = {gt_hr_bpm:.1f} BPM")
        ax.axvspan(gt_hr_hz - tol_hz, gt_hr_hz + tol_hz,
                   alpha=0.15, color="green")

        # Respiration harmonics (red dotted)
        if f_resp is not None:
            for k in range(1, 7):
                fk = k * f_resp
                if fk < 5.0:
                    ax.axvline(fk, color="red", linestyle=":",
                               alpha=0.5, linewidth=1)
            ax.axvline(f_resp, color="red", linestyle=":",
                       alpha=0.0,  # invisible duplicate for legend
                       label=f"Resp harmonics (f={f_resp*60:.1f} BPM)")

        # Background band (subtle shading)
        ax.axvspan(SNR_BG_LO_HZ, SNR_BG_HI_HZ,
                   alpha=0.05, color="blue")

        row, col = c // COLS, c % COLS
        ax.set_title(
            f"cell_{c:2d} (row {row}, col {col}) — "
            f"SNR = {cardiac_snr[c]:.2f}",
            fontsize=10
        )
        ax.set_xlim(0, 5)
        ax.set_ylabel("PSD (log)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"Raw cell spectra (post-trim, detrended, no bandpass) — {rec_id}",
        fontsize=12, y=1.00
    )
    plt.tight_layout()

    spectra_path = DIAG_DIR / f"spectra_{rec_id}.png"
    plt.savefig(spectra_path, dpi=150, bbox_inches="tight")
    print(f"  Spectra saved → {spectra_path.name}")
    plt.close(fig)

    # ── Return summary for batch table ─────────────────────────
    return {
        "rec_id":          rec_id,
        "gt_hr_bpm":       gt_hr_bpm,
        "gt_hr_std":       hr["hr_std"],
        "f_resp_bpm":      f_resp * 60 if f_resp is not None else None,
        "best_cell":       int(best_cells[0]),
        "best_snr":        float(cardiac_snr[best_cells[0]]),
        "median_snr":      float(np.median(cardiac_snr)),
        "snr_top4":        [float(cardiac_snr[c]) for c in best_cells],
        "n_samples_used":  len(grid_df),
        "duration_sec":    float(t_cam[-1] - t_cam[0]),
    }


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DIAGNOSTIC: Cardiac signal presence in raw cells")
    print(f"Outputs → {DIAG_DIR}")
    print(f"Recordings to process: {len(REC_IDS)}")

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
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'rec_id':<25} {'GT HR':>8} {'f_resp':>8} "
              f"{'best cell':>10} {'best SNR':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for s in summaries:
            resp_str = f"{s['f_resp_bpm']:.1f}" if s['f_resp_bpm'] else "—"
            print(f"  {s['rec_id']:<25} "
                  f"{s['gt_hr_bpm']:>6.1f}   "
                  f"{resp_str:>6}    "
                  f"cell_{s['best_cell']:<4} "
                  f"{s['best_snr']:>9.2f}")

        # Save summary as JSON for later reference
        summary_path = DIAG_DIR / "summary.json"
        with open(summary_path, "w") as f:
            # Make it JSON-serializable
            json.dump(summaries, f, indent=2)
        print(f"\n  Summary saved → {summary_path.name}")