#!/usr/bin/env python3
"""
make_waveform_figures.py
========================
Two figure types per recording:

  (A) per-window variability : all per-window BPM estimates over time,
      with the final median estimate and the ground-truth mean. Shows how
      the individual windows scatter and why the median is the sensible
      aggregate.

  (B) stacked waveform vs ECG : the recovered pulse waveform (top) over the
      SAME time window as the ECG ground truth (bottom), for the window
      whose estimate is closest to the ground-truth mean. Both panels share
      the x-axis, so beats line up in time.

Reads:
  - pipeline JSON : PULSE_DIR/selection_windowed_{REC}.json  (per_window with
                    pulse_signal / pulse_fs)
  - ECG ground truth : MOVESENSE_DIR/{REC}/ecg_stream.json
  - HR ground truth  : MOVESENSE_DIR/{REC}/heartRate_stream.json (for GT mean)

Usage:
  python make_waveform_figures.py                 # all recordings with a JSON
  python make_waveform_figures.py AGA_800_hoodie  # one or more
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PULSE_DIR     = Path(r"C:\Projects\thesis\data\PULSE_files_SQI")
MOVESENSE_DIR = Path(r"C:\Projects\thesis\data\movesense")
OUT_A = Path(r"C:\Projects\thesis\thesis_latex\images\variability")
OUT_B = Path(r"C:\Projects\thesis\thesis_latex\images\waveform_vs_ecg")
OUT_A.mkdir(parents=True, exist_ok=True)
OUT_B.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_PULSE = "#5b7c99"
C_ECG   = "#2e2e2e"
C_RPEAK = "#a01818"
C_FINAL = "#1f3b52"
C_GT    = "#2e7d5b"


# ------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------
def load_pipeline(rec_id):
    jp = PULSE_DIR / f"selection_windowed_{rec_id}.json"
    if not jp.exists():
        return None
    return json.load(open(jp))


def load_ecg(rec_id):
    f = MOVESENSE_DIR / rec_id / "ecg_stream.json"
    if not f.exists():
        return None
    d = json.load(open(f))["data"]
    samples, ts = [], []
    for p in d:
        e = p["ecg"]
        samples.append(np.asarray(e["Samples"], dtype=float))
        ts.append(e["Timestamp"])
    sig = np.concatenate(samples)
    ts = np.asarray(ts, dtype=float)
    spp = len(d[0]["ecg"]["Samples"])
    fs = spp / (np.median(np.diff(ts)) / 1000.0) if len(ts) > 1 else 125.0
    t = np.arange(len(sig)) / fs
    return t, sig, fs


def gt_mean_from_json(rec):
    """Prefer the gt_bpm already stored in the pipeline JSON."""
    return rec.get("gt_bpm")


# ------------------------------------------------------------------
# (A) PER-WINDOW VARIABILITY
# ------------------------------------------------------------------
def plot_variability(rec_id, rec):
    pw = rec.get("per_window", [])
    if not pw:
        return None
    t = np.array([w["t_center_s"] for w in pw], float)
    bpm = np.array([w["bpm"] for w in pw], float)
    scr = np.array([w.get("score", np.nan) for w in pw], float)
    order = np.argsort(t)
    t, bpm, scr = t[order], bpm[order], scr[order]

    final = rec.get("bpm_smoothed")
    gt = gt_mean_from_json(rec)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(t, bpm, color=C_PULSE, lw=1.3, marker="o", markersize=4,
            markerfacecolor=C_PULSE, markeredgecolor=C_PULSE, zorder=3,
            label="per-window estimate")
    if final is not None:
        ax.axhline(final, color=C_FINAL, ls="--", lw=1.4, zorder=2,
                   label=f"final median ({final:.1f} BPM)")
    if gt is not None:
        ax.axhline(gt, color=C_GT, ls="-", lw=1.4, zorder=2,
                   label=f"ground truth ({gt:.1f} BPM)")

    ax.set_xlabel("Window centre (s)")
    ax.set_ylabel("Estimated HR (BPM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="best", fontsize=8)

    fig.tight_layout()
    out = OUT_A / f"variability_{rec_id}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# (B) STACKED WAVEFORM vs ECG (closest-to-GT window)
# ------------------------------------------------------------------
def plot_stacked(rec_id, rec, ecg):
    pw = rec.get("per_window", [])
    if not pw:
        return None
    gt = gt_mean_from_json(rec)
    cand = [w for w in pw if w.get("pulse_signal")]
    if not cand:
        return None

    # window whose estimate is closest to the ground-truth mean
    #if gt is not None:
        best = min(cand, key=lambda w: abs(w["bpm"] - gt))
    
    #else:
        best = max(cand, key=lambda w: w.get("score", -np.inf))

    best = max(cand, key=lambda w: w.get("variance_pct", -np.inf))

    sig = np.asarray(best["pulse_signal"], float)
    fs_p = float(best.get("pulse_fs", rec.get("fs", 15.0)))
    sig = sig - np.mean(sig)
    s = np.std(sig)
    if s > 1e-12:
        sig = sig / s

    # window time span in the recording's clock
    win_sec = rec.get("window_sec", 15.0)
    t_center = best["t_center_s"]
    t0 = t_center - win_sec / 2.0
    t_pulse = t0 + np.arange(len(sig)) / fs_p   # absolute seconds

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)

    # top: recovered pulse
    ax1.plot(t_pulse, sig, color=C_PULSE, lw=1.1)
    ax1.axhline(0, color="0.7", lw=0.6, zorder=0)
    ax1.set_ylabel("Pulse (a.u.)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", length=3)
    ax1.text(0.99, 0.95, "depth camera",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=8.5, color="0.35")

    # bottom: ECG over the same time window
    if ecg is not None:
        te, se, fse = ecg
        m = (te >= t0) & (te <= t0 + win_sec)
        tp, sp = te[m], se[m]
        # R-peaks within the window
        s0 = se - np.median(se)
        pk, _ = find_peaks(s0, height=np.percentile(s0, 98) * 0.5,
                           distance=int(0.4 * fse))
        pk = pk[(te[pk] >= t0) & (te[pk] <= t0 + win_sec)]
        ax2.plot(tp, sp, color=C_ECG, lw=0.8)
        ax2.plot(te[pk], se[pk], "v", color=C_RPEAK, markersize=5,
                 markeredgecolor="black", markeredgewidth=0.3, label="R-peak")
        ax2.legend(frameon=False, loc="upper right")
        if gt is not None:
            ax2.text(0.99, 0.05, f"ECG - ground truth {gt:.1f} BPM",
                     transform=ax2.transAxes, ha="right", va="bottom",
                     fontsize=8.5, color="0.35")
    ax2.set_ylabel("ECG (a.u.)")
    ax2.set_xlabel("Time (s)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", length=3)
    ax2.set_xlim(t0, t0 + win_sec)

    fig.tight_layout()
    out = OUT_B / f"wave_vs_ecg_{rec_id}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        rec_ids = sys.argv[1:]
    else:
        rec_ids = [p.stem.replace("selection_windowed_", "")
                   for p in sorted(PULSE_DIR.glob("selection_windowed_*.json"))]
    if not rec_ids:
        print(f"No pipeline JSONs in {PULSE_DIR}")
        sys.exit(1)

    n_a = n_b = 0
    for rec_id in rec_ids:
        rec = load_pipeline(rec_id)
        if rec is None:
            print(f"  no JSON: {rec_id}")
            continue
        if plot_variability(rec_id, rec):
            n_a += 1
        ecg = load_ecg(rec_id)
        if plot_stacked(rec_id, rec, ecg):
            n_b += 1
        if ecg is None:
            print(f"  (B) no ECG for {rec_id} - waveform panel only skipped")

    print(f"\n(A) variability plots     : {n_a}  -> {OUT_A}")
    print(f"(B) waveform-vs-ECG plots : {n_b}  -> {OUT_B}")


if __name__ == "__main__":
    main()