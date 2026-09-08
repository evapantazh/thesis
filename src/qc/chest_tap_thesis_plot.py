#!/usr/bin/env python3
"""Figure — Chest-tap synchronisation (before/after), sternum proxy vs Movesense IMU.
Tap times are loaded from previously saved Tap_info_{REC_ID}.json — no manual re-confirmation."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ─────────────────────────────────────────────────────────────
#  RECORDING — edit these
# ─────────────────────────────────────────────────────────────
SUBJECT = "IMA"
DIST    = "800"
CLOTH   = "tshirt"

BASE     = Path(r"C:\Projects\thesis\data")
REC_BASE = Path(r"D:\recordings")
OUTDIR   = Path(r"C:\Projects\thesis\thesis_latex\images")
OUTDIR.mkdir(parents=True, exist_ok=True)

REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

PATH_RECORDING     = REC_BASE / REC_ID
PATH_TIMESTAMPS    = PATH_RECORDING / "timestamps.csv"
PATH_CAMERA_GRID   = BASE / "GRID_files" / f"GRID_{REC_ID}.csv"
PATH_MOVESENSE_ACC = BASE / "movesense" / REC_ID / "acc_stream.json"
PATH_TAP_INFO      = BASE / "tap_info" / "json" / f"Tap_info_{REC_ID}.json"

SEARCH_WINDOW_SEC = 10.0
FS_MS_ACC = 52.0
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]
TAP_SMOOTH_MS = 0.0

# Palette matching Figure 4.2
C_CAMERA    = "#7B52AB"   # slate blue-grey
C_MOVESENSE = "#E07B00"  # 
C_TAP_LINE  = "#a01818"   # red accent


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_timestamps(path):
    df = pd.read_csv(path).set_index("frame")
    df["depth_ts"] = df["depth_timestamp"] if "depth_timestamp" in df.columns else df["timestamp"]
    return df


def build_proxy_from_cells(X, cell_indices):
    X = np.asarray(X, float).copy()
    X = X[:, cell_indices]
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = (X[:, j] == 0) | ~np.isfinite(X[:, j])
        X[bad, j] = col_med[j]
    dX = np.diff(X, axis=0, prepend=X[[0]])
    return np.sum(np.abs(dX), axis=1) * np.abs(np.mean(dX, axis=1))


def smooth_signal(proxy, ms, fs):
    sigma = max(int((ms / 1000.0) * fs), 0)
    return gaussian_filter1d(proxy, sigma=sigma) if sigma > 0 else proxy.copy()


def get_sternum_signal(ts_df):
    df_cam = pd.read_csv(PATH_CAMERA_GRID)
    grid_frames = df_cam["frame"].values.astype(int)
    X = df_cam.drop(columns=["frame"]).values.astype(float)
    grid_ts = ts_df.loc[grid_frames, "depth_ts"].values
    t_cam = (grid_ts - grid_ts[0]) / 1000.0

    valid_cells = [c for c in STERNUM_CELLS if c < X.shape[1]]
    proxy = build_proxy_from_cells(X, valid_cells)
    proxy_smooth = smooth_signal(proxy, TAP_SMOOTH_MS, fs=1.0 / np.median(np.diff(t_cam[t_cam > 0])))
    return t_cam, proxy_smooth


def get_movesense_signal():
    with open(PATH_MOVESENSE_ACC, 'r') as f:
        data = json.load(f)['data']
    start_ts = data[0]['acc']['Timestamp']
    timestamps, magnitudes = [], []
    for entry in data:
        ts = entry['acc']['Timestamp']
        for i, s in enumerate(entry['acc']['ArrayAcc']):
            magnitudes.append(np.sqrt(s['x']**2 + s['y']**2 + s['z']**2))
            timestamps.append(ts + i * (1000.0 / FS_MS_ACC))
    t_ms = (np.array(timestamps) - start_ts) / 1000.0
    proxy = np.abs(np.gradient(magnitudes))
    proxy_smooth = smooth_signal(proxy, TAP_SMOOTH_MS, fs=FS_MS_ACC)
    return t_ms, proxy_smooth


def load_tap_info():
    with open(PATH_TAP_INFO, 'r') as f:
        info = json.load(f)
    manual_t  = info["manual_tap_sec"]
    ms_tap_t  = info["ms_tap_sec"]
    dt_manual = info["offset_sec"]  # = ms_tap_sec - manual_tap_sec
    return manual_t, ms_tap_t, dt_manual


# ─────────────────────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────────────────────

def save_sync_figure(t_cam, stern_smooth, t_ms, ms_smooth, manual_t, ms_tap_t, dt_manual):

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    fs_cam = 1.0 / float(np.median(np.diff(t_cam[t_cam > 0])))

    def display_smooth(x, ms, fs):
        sigma = max((ms / 1000.0) * fs / 2.355, 0.5)
        return gaussian_filter1d(x, sigma=sigma)

    stern_disp = display_smooth(stern_smooth, ms=80, fs=fs_cam)
    ms_disp    = display_smooth(ms_smooth, ms=80, fs=FS_MS_ACC)

    tap_common = ms_tap_t
    a, b = tap_common - 1.2, tap_common + 1.8
    t_cam_aligned = t_cam + dt_manual

    mc_raw = (t_cam >= a) & (t_cam <= b)
    mc_aln = (t_cam_aligned >= a) & (t_cam_aligned <= b)
    mm     = (t_ms >= a) & (t_ms <= b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.16, wspace=0.10)

    ax1.plot(t_cam[mc_raw], norm(stern_disp[mc_raw]), color=C_CAMERA, lw=1.6,
             label="Depth camera (sternum)")
    ax1.plot(t_ms[mm], norm(ms_disp[mm]), color=C_MOVESENSE, lw=1.6,
             label="Movesense IMU")
    ax1.axvline(manual_t, color=C_CAMERA, ls="--", lw=1.0, alpha=0.8)
    ax1.axvline(ms_tap_t, color=C_MOVESENSE, ls="--", lw=1.0, alpha=0.8)
    ax1.annotate("", xy=(ms_tap_t, 2.5), xytext=(manual_t, 2.5),
                 arrowprops=dict(arrowstyle="<->", color=C_TAP_LINE, lw=1.2))
    ax1.text((ms_tap_t + manual_t) / 2, 2.7, rf"$\Delta t$ = {abs(dt_manual):.3f} s",
             ha="center", va="bottom", fontsize=8.5, color=C_TAP_LINE)
    ax1.set_xlim(-0.8, 5.5)
    ax1.set_xlabel("Time (s) — own clocks")
    ax1.set_ylabel("Normalised amplitude")
    ax1.set_title("Before synchronisation")
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', length=3)

    ax2.plot(t_cam_aligned[mc_aln], norm(stern_disp[mc_aln]), color=C_CAMERA, lw=1.6,
             label="Depth camera (aligned)")
    ax2.plot(t_ms[mm], norm(ms_disp[mm]), color=C_MOVESENSE, lw=1.6,
             label="Movesense IMU")
    ax2.axvline(tap_common, color=C_TAP_LINE, ls="--", lw=1.2,
                label=f"Tap  t = {tap_common:.3f} s")
    ax2.set_xlim(-0.8, 5.5)
    ax2.set_xlabel("Time (s) — common clock")
    ax2.set_title("After synchronisation")
    ax2.legend(frameon=False, loc="upper right")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', length=3, labelleft=False)

    fig.savefig(OUTDIR / "fig_tap_sync_IMA.pdf")
    fig.savefig(OUTDIR / "fig_tap_sync_IMA.png", dpi=150)
    print(f"saved to {OUTDIR}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

ts_df = load_timestamps(PATH_TIMESTAMPS)
t_cam, stern_smooth = get_sternum_signal(ts_df)
t_ms, ms_smooth      = get_movesense_signal()
manual_t, ms_tap_t, dt_manual = load_tap_info()

save_sync_figure(t_cam, stern_smooth, t_ms, ms_smooth, manual_t, ms_tap_t, dt_manual)