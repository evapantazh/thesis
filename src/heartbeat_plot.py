#!/usr/bin/env python3
"""
make_waveforms.py
=================
Produces, per recording, TWO SEPARATE thesis-quality plots:

  (A) pulse waveform  - the continuous oscillating signal recovered by the
      depth-camera pipeline (the BCG-like trace the HR estimate comes from).
  (B) ECG ground truth - the raw Movesense ECG waveform (125 Hz) with the
      detected R-peaks marked, i.e. the beat-to-beat reference.

It also writes a CSV of the ground-truth beat-to-beat series (RR intervals
and instantaneous HR) for each recording.

------------------------------------------------------------------
IMPORTANT - pulse waveform availability
------------------------------------------------------------------
The pipeline (TEST_06_eigenvectors_SQI.py) currently saves only the per-window
BPM, NOT the oscillating pulse signal. To plot (A) you must save the winning
signal. Add this in process_window()'s return dict:

    'pulse_signal': best['signal'].tolist(),
    'pulse_fs':     fs,

and, if you want one continuous trace, also stitch the per-window signals
(see stitch note below). Until then, (A) is skipped with a message and only
(B), the ECG ground truth, is produced.

Usage:
    python make_waveforms.py                    # all recordings
    python make_waveforms.py GBA_1200_tshirt    # one recording
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PULSE_DIR    = Path(r"C:\Projects\thesis\data\PULSE_files_SQI")
MOVESENSE_DIR = Path(r"C:\Projects\thesis\data\movesense")
OUT_PLOTDIR  = Path(r"C:\Projects\thesis\thesis_latex\images\waveforms")
OUT_CSVDIR   = Path(r"C:\Projects\thesis\thesis_latex\images\gt_beat_to_beat")

OUT_PLOTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSVDIR.mkdir(parents=True, exist_ok=True)

ECG_FS = 125.0   # Movesense ECG sampling rate (Hz), verified from timestamps

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_PULSE = "#5b7c99"   # slate blue - recovered pulse
C_ECG   = "#2e2e2e"   # near-black  - ECG trace
C_RPEAK = "#a01818"   # red         - R-peaks


# ------------------------------------------------------------------
# LOAD ECG GROUND TRUTH
# ------------------------------------------------------------------
def load_ecg(rec_id):
    """Return (t, ecg, fs) or None. Concatenates all ECG sample packets."""
    ecg_file = MOVESENSE_DIR / rec_id / "ecg_stream.json"
    if not ecg_file.exists():
        return None
    d = json.load(open(ecg_file))["data"]
    samples, ts = [], []
    for p in d:
        e = p["ecg"]
        samples.append(np.asarray(e["Samples"], dtype=float))
        ts.append(e["Timestamp"])
    sig = np.concatenate(samples)

    # derive fs from packet timestamps (fallback to ECG_FS)
    ts = np.asarray(ts, dtype=float)
    spp = len(d[0]["ecg"]["Samples"])          # samples per packet
    if len(ts) > 1:
        dt = np.median(np.diff(ts)) / 1000.0   # s between packets
        fs = spp / dt if dt > 0 else ECG_FS
    else:
        fs = ECG_FS
    t = np.arange(len(sig)) / fs
    return t, sig, fs


def load_rr(rec_id):
    """Return beat-to-beat RR (ms) and instantaneous HR (BPM), or None."""
    hr_file = MOVESENSE_DIR / rec_id / "heartRate_stream.json"
    if not hr_file.exists():
        return None
    d = json.load(open(hr_file))["data"]
    rr = []
    for e in d:
        vals = e["heartRate"].get("rrData")
        if vals:
            rr.extend(vals)
    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return None
    hr = 60000.0 / rr
    # cumulative time of each beat (s), starting at first RR
    t_beat = np.cumsum(rr) / 1000.0
    return t_beat, rr, hr


# ------------------------------------------------------------------
# LOAD RECOVERED PULSE (if the pipeline saved it)
# ------------------------------------------------------------------
def load_pulse(rec_id):
    """Return (t, signal, fs) if the pipeline saved a pulse_signal, else None."""
    jp = PULSE_DIR / f"selection_windowed_{rec_id}.json"
    if not jp.exists():
        return None
    r = json.load(open(jp))
    # Option 1: a single stitched continuous trace saved at top level
    if "pulse_signal" in r and "pulse_fs" in r:
        sig = np.asarray(r["pulse_signal"], dtype=float)
        fs = float(r["pulse_fs"])
        return np.arange(len(sig)) / fs, sig, fs
    # Option 2: per-window signals -> stitch by taking the middle second of
    # each window (stride = 1 s) to build a continuous, minimally-overlapping trace
    pw = r.get("per_window", [])
    if pw and all("pulse_signal" in w for w in pw):
        fs = r.get("fs", 15.0)
        stride = int(round(r.get("stride_sec", 1.0) * fs))
        win = int(round(r.get("window_sec", 15.0) * fs))
        mid = win // 2
        chunks = []
        for w in sorted(pw, key=lambda x: x["t_center_s"]):
            s = np.asarray(w["pulse_signal"], dtype=float)
            if len(s) >= mid + stride:
                chunks.append(s[mid:mid + stride])
        if chunks:
            sig = np.concatenate(chunks)
            return np.arange(len(sig)) / fs, sig, fs
    return None


# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------
def plot_pulse(rec_id, t, sig, fs, t_window=None):
    """Plot (A): recovered pulse waveform. Optionally zoom to t_window=(a,b) s."""
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    if t_window is not None:
        m = (t >= t_window[0]) & (t <= t_window[1])
        t, sig = t[m], sig[m]
    ax.plot(t, sig, color=C_PULSE, lw=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    fig.tight_layout()
    out = OUT_PLOTDIR / f"pulse_{rec_id}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def plot_ecg(rec_id, t, ecg, fs, t_window=None):
    """Plot (B): ECG ground truth with detected R-peaks."""
    # detect R-peaks on full signal (before any zoom)
    s = ecg - np.median(ecg)
    height = np.percentile(s, 98) * 0.5
    peaks, _ = find_peaks(s, height=height, distance=int(0.4 * fs))

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    if t_window is not None:
        m = (t >= t_window[0]) & (t <= t_window[1])
        tp, sp = t[m], ecg[m]
        pk = peaks[(t[peaks] >= t_window[0]) & (t[peaks] <= t_window[1])]
    else:
        tp, sp, pk = t, ecg, peaks
    ax.plot(tp, sp, color=C_ECG, lw=0.8)
    ax.plot(t[pk], ecg[pk], "v", color=C_RPEAK, markersize=5,
            markeredgecolor="black", markeredgewidth=0.3, label="R-peak")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = OUT_PLOTDIR / f"ecg_{rec_id}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out, len(peaks)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def discover_rec_ids():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    ids = []
    if MOVESENSE_DIR.exists():
        ids = [p.name for p in MOVESENSE_DIR.iterdir() if p.is_dir()]
    return sorted(ids)


def main():
    rec_ids = discover_rec_ids()
    if not rec_ids:
        print(f"No recordings found under {MOVESENSE_DIR}")
        sys.exit(1)

    # optional zoom window (seconds) for a readable beat-by-beat view;
    # set to None to plot the full recording
    ZOOM = (10.0, 20.0)

    n_ecg = n_pulse = 0
    for rec_id in rec_ids:
        # ---- ECG ground truth ----
        ecg = load_ecg(rec_id)
        if ecg is not None:
            t, sig, fs = ecg
            _, npk = plot_ecg(rec_id, t, sig, fs, t_window=ZOOM)
            n_ecg += 1
        else:
            print(f"  no ECG for {rec_id}")

        # ---- beat-to-beat CSV from RR ----
        rr = load_rr(rec_id)
        if rr is not None:
            t_beat, rr_ms, hr_bpm = rr
            pd.DataFrame({
                "beat_index": np.arange(1, len(rr_ms) + 1),
                "t_s": np.round(t_beat, 3),
                "rr_ms": rr_ms,
                "hr_bpm": np.round(hr_bpm, 2),
            }).to_csv(OUT_CSVDIR / f"gt_beats_{rec_id}.csv", index=False)

        # ---- recovered pulse waveform ----
        pulse = load_pulse(rec_id)
        if pulse is not None:
            t, sig, fs = pulse
            plot_pulse(rec_id, t, sig, fs, t_window=ZOOM)
            n_pulse += 1
        # (silent skip if not saved yet; message once below)

    print(f"\nECG ground-truth plots : {n_ecg}")
    print(f"Pulse waveform plots   : {n_pulse}")
    if n_pulse == 0:
        print("\n  NOTE: no pulse waveforms were plotted because the pipeline")
        print("  JSONs don't contain 'pulse_signal'. See the header of this")
        print("  script for the two lines to add to TEST_06 process_window().")
    print(f"Plots dir : {OUT_PLOTDIR}")
    print(f"CSV dir   : {OUT_CSVDIR}")


if __name__ == "__main__":
    main()