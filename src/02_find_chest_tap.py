"""
02c_manual_tap_selector.py
==========================
Manual chest tap selector with algorithm comparison.

Two camera proxies are computed and shown:
  - FULL proxy   : all 28 cells (energy x coherence)
  - STERNUM proxy: only cells [5,6,9,10,13,14] (centre chest)

The sternum proxy should produce a cleaner tap signal because
it ignores the hand entering/leaving the depth field and only
reacts to motion at the sternum itself.

Viewer overlays:
  - YELLOW bar  : depth proxy algorithm tap estimate (full proxy)
  - CYAN bar    : sternum proxy algorithm tap estimate
  - RED border  : within 0.12s of Movesense tap
  - GREEN banner: your manually confirmed frame

Controls:
  LEFT / A    : previous frame
  RIGHT / D   : next frame
  SPACE       : confirm this frame as the tap
  Q / ESC     : quit without saving
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  RECORDING  — edit these
# ─────────────────────────────────────────────────────────────
SUBJECT = "GAX"
DIST    = "800"
CLOTH   = "tshirt"

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Projects\thesis\data")
REC_BASE = Path(r"D:\recordings")

REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

PATH_RECORDING     = REC_BASE / REC_ID
PATH_COLOR_FRAMES  = PATH_RECORDING / "color"
PATH_TIMESTAMPS    = PATH_RECORDING / "timestamps.csv"

PATH_CAMERA_GRID   = BASE / "GRID_files" / f"GRID_{REC_ID}.csv"
PATH_MOVESENSE     = BASE / "movesense" / REC_ID
PATH_MOVESENSE_ACC = PATH_MOVESENSE / "acc_stream.json"

PATH_OUTPUT_JSON   = BASE / "tap_info" / "json"/ f"Tap_info_{REC_ID}.json"
PATH_OUTPUT_PLOT   = BASE / "tap_info" / "plots_debug"/ f"tap_sync_{REC_ID}.png"

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
SEARCH_WINDOW_SEC = 10.0
DISPLAY_WIDTH     = 1024
DISPLAY_HEIGHT    = 600
FS_MS_ACC         = 52.0

# Sternum cells confirmed by visualize_grid_cells.py
# Grid is 7 rows x 4 cols, cell_idx = row*4 + col
# Rows 1-3, cols 1-2 = centre chest / sternum
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]

# Depth proxy algorithm settings
TAP_SMOOTH_MS_CAM = 0.0
TAP_SMOOTH_MS_MS  = 0.0
PROMINENCE_SIGMA  = 2.0
MIN_DISTANCE_SEC  = 0.25


# ─────────────────────────────────────────────────────────────
#  TIMESTAMP LOADER
# ─────────────────────────────────────────────────────────────

def load_timestamps(path):
    df = pd.read_csv(path).set_index("frame")
    if "color_timestamp" in df.columns:
        df["color_ts"] = df["color_timestamp"]
    elif "timestamp" in df.columns:
        df["color_ts"] = df["timestamp"]
    else:
        raise KeyError(f"No color timestamp column. Available: {df.columns.tolist()}")

    if "depth_timestamp" in df.columns:
        df["depth_ts"] = df["depth_timestamp"]
    elif "timestamp" in df.columns:
        df["depth_ts"] = df["timestamp"]
    else:
        raise KeyError(f"No depth timestamp column. Available: {df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────────────────────
#  PROXY BUILDERS
# ─────────────────────────────────────────────────────────────

def build_proxy_from_cells(X, cell_indices=None):
    """
    Build energy x coherence proxy.
    If cell_indices given, only use those columns.
    """
    X = np.asarray(X, float).copy()

    if cell_indices is not None:
        X = X[:, cell_indices]

    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = (X[:, j] == 0) | ~np.isfinite(X[:, j])
        X[bad, j] = col_med[j]

    dX        = np.diff(X, axis=0, prepend=X[[0]])
    energy    = np.sum(np.abs(dX), axis=1)
    coherence = np.abs(np.mean(dX, axis=1))
    return energy * coherence


def find_tap_peaks(proxy, t, fs, search_window_sec,
                   smooth_ms, prominence_sigma, min_distance_sec):
    proxy = np.asarray(proxy, float)
    t     = np.asarray(t, float)
    mask  = t <= search_window_sec
    x     = proxy[mask]
    sigma = max(int((smooth_ms / 1000.0) * fs), 0)
    x_s   = gaussian_filter1d(x, sigma=sigma) if sigma > 0 else x.copy()
    proxy_smooth = gaussian_filter1d(proxy, sigma=sigma) if sigma > 0 else proxy.copy()
    prom  = np.std(x_s) * prominence_sigma
    dist  = max(int(min_distance_sec * fs), 1)
    peaks, _ = signal.find_peaks(x_s, prominence=prom, distance=dist)
    global_indices = np.where(mask)[0][peaks]
    peak_times     = t[global_indices]
    return global_indices, peak_times, proxy_smooth


def select_tap_camera(global_indices, peak_times, proxy, t, grid_frames):
    """Returns (frame_id, t_sec, method_str)"""
    if len(global_indices) == 0:
        return None, None, "no_peaks_found"

    if len(global_indices) < 2:
        gi = global_indices[0]
        return int(grid_frames[gi]), float(t[gi]), "single_peak_fallback"

    sorted_by_value = np.argsort(proxy[global_indices])[::-1]
    top2  = sorted(sorted_by_value[:2])
    gi_a  = global_indices[top2[0]]
    gi_b  = global_indices[top2[1]]
    start = gi_a + 1
    end   = gi_b

    if end - start < 2:
        contact_gi = (gi_a + gi_b) // 2
        return int(grid_frames[contact_gi]), float(t[contact_gi]), "midpoint_fallback"

    region     = proxy[start:end]
    region_std = np.std(region)
    local_peaks, _ = signal.find_peaks(region, prominence=region_std * 0.5)

    if len(local_peaks) > 0:
        best_local = local_peaks[np.argmax(region[local_peaks])]
        contact_gi = start + best_local
        return int(grid_frames[contact_gi]), float(t[contact_gi]), "contact_bump"
    else:
        valley_local = np.argmin(region)
        offset     = 1 if valley_local + 1 < len(region) else 0
        contact_gi = start + valley_local + offset
        return int(grid_frames[contact_gi]), float(t[contact_gi]), "post_valley_fallback"


def run_depth_algorithm(ts_df):
    """
    Run depth proxy algorithm with BOTH full and sternum-only proxy.
    Returns dicts for full and sternum results.
    """
    if not PATH_CAMERA_GRID.exists():
        print(f"  [WARN] Grid file not found: {PATH_CAMERA_GRID}")
        empty = (None, None, "grid_missing", None, None, None, None)
        return empty, empty

    df_cam      = pd.read_csv(PATH_CAMERA_GRID)
    grid_frames = df_cam["frame"].values.astype(int)
    X           = df_cam.drop(columns=["frame"]).values.astype(float)

    grid_timestamps = ts_df.loc[grid_frames, "depth_ts"].values
    t_cam = (grid_timestamps - grid_timestamps[0]) / 1000.0

    intervals = np.diff(t_cam)
    FS_CAM    = float(1.0 / np.median(intervals[intervals > 0]))
    print(f"  Camera FPS (grid): {FS_CAM:.2f}")

    # ── Full proxy ────────────────────────────────────────────
    full_proxy = build_proxy_from_cells(X, cell_indices=None)
    full_idx, full_times, full_smooth = find_tap_peaks(
        full_proxy, t_cam, FS_CAM,
        SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_CAM, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
    )
    full_frame, full_t, full_method = select_tap_camera(
        full_idx, full_times, full_proxy, t_cam, grid_frames
    )
    print(f"  [FULL proxy]    tap: frame {full_frame}  t={full_t:.4f}s  [{full_method}]")
    print(f"    peaks: {len(full_idx)}")
    for i, (gi, tp) in enumerate(zip(full_idx, full_times)):
        print(f"      peak {i+1}: frame={grid_frames[gi]}  t={tp:.3f}s  "
              f"proxy={full_proxy[gi]:.1f}")

    # ── Sternum proxy ─────────────────────────────────────────
    # Use valid sternum cell indices that exist in X
    n_cells = X.shape[1]
    valid_sternum = [c for c in STERNUM_CELLS if c < n_cells]
    stern_proxy = build_proxy_from_cells(X, cell_indices=valid_sternum)
    stern_idx, stern_times, stern_smooth = find_tap_peaks(
        stern_proxy, t_cam, FS_CAM,
        SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_CAM, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
    )
    stern_frame, stern_t, stern_method = select_tap_camera(
        stern_idx, stern_times, stern_proxy, t_cam, grid_frames
    )
    print(f"  [STERNUM proxy] tap: frame {stern_frame}  t={stern_t:.4f}s  [{stern_method}]")
    print(f"    peaks: {len(stern_idx)}")
    for i, (gi, tp) in enumerate(zip(stern_idx, stern_times)):
        print(f"      peak {i+1}: frame={grid_frames[gi]}  t={tp:.3f}s  "
              f"proxy={stern_proxy[gi]:.1f}")

    full_result  = (full_frame,  full_t,  full_method,
                    full_proxy,  t_cam,   grid_frames, full_smooth)
    stern_result = (stern_frame, stern_t, stern_method,
                    stern_proxy, t_cam,   grid_frames, stern_smooth)

    return full_result, stern_result


# ─────────────────────────────────────────────────────────────
#  MOVESENSE
# ─────────────────────────────────────────────────────────────

def load_movesense_tap():
    with open(PATH_MOVESENSE_ACC, 'r') as f:
        data = json.load(f)['data']

    start_ts   = data[0]['acc']['Timestamp']
    timestamps, magnitudes = [], []

    for entry in data:
        ts      = entry['acc']['Timestamp']
        samples = entry['acc']['ArrayAcc']
        for i, s in enumerate(samples):
            mag = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            magnitudes.append(mag)
            timestamps.append(ts + i * (1000.0 / FS_MS_ACC))

    t_ms  = (np.array(timestamps) - start_ts) / 1000.0
    proxy = np.abs(np.gradient(magnitudes))

    ms_idx, ms_times, ms_proxy_smooth = find_tap_peaks(
        proxy, t_ms, FS_MS_ACC,
        SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_MS, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
    )

    if len(ms_idx) == 0:
        raise ValueError("No Movesense peaks found!")

    best   = np.argmax(proxy[ms_idx])
    tap_gi = ms_idx[best]
    tap_t  = float(t_ms[tap_gi])

    i = int(tap_gi)
    if 0 < i < len(proxy) - 1:
        y0, y1, y2 = proxy[i-1], proxy[i], proxy[i+1]
        denom = y0 - 2*y1 + y2
        if abs(denom) > 1e-12:
            delta  = np.clip((y0 - y2) / (2*denom), -0.5, 0.5)
            tap_t += delta * float(t_ms[i+1] - t_ms[i])

    print(f"  Movesense tap: {tap_t:.4f}s  "
          f"(raw idx={tap_gi}  proxy={proxy[tap_gi]:.4f})")
    print(f"  Peaks found: {len(ms_idx)}")
    for i, (gi, tp) in enumerate(zip(ms_idx, ms_times)):
        marker = " <<<" if i == best else ""
        print(f"    peak {i+1}: t={tp:.3f}s  proxy={proxy[gi]:.4f}{marker}")

    return tap_t, int(start_ts), t_ms, proxy, ms_proxy_smooth


# ─────────────────────────────────────────────────────────────
#  COLOR FRAMES
# ─────────────────────────────────────────────────────────────

def get_frames_in_window(ts_df):
    jpgs = sorted(PATH_COLOR_FRAMES.glob("frame_*.jpg"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not jpgs:
        raise FileNotFoundError(f"No frame_*.jpg in {PATH_COLOR_FRAMES}")

    frame_ids = np.array([int(p.stem.split("_")[1]) for p in jpgs])
    valid     = np.isin(frame_ids, ts_df.index.values)
    jpgs      = [p for p, v in zip(jpgs, valid) if v]
    frame_ids = frame_ids[valid]

    timestamps = ts_df.loc[frame_ids, "color_ts"].values
    t_sec      = (timestamps - timestamps[0]) / 1000.0

    mask = t_sec <= SEARCH_WINDOW_SEC
    return [
        (int(fid), float(t), p)
        for fid, t, p, ok in zip(frame_ids, t_sec, jpgs, mask) if ok
    ]


# ─────────────────────────────────────────────────────────────
#  OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────

def draw_overlay(frame_bgr, frame_id, t_sec, idx, total,
                 ms_tap_t,
                 full_frame, full_t,
                 stern_frame, stern_t,
                 confirmed_frame):
    img = frame_bgr.copy()
    H, W = img.shape[:2]

    # Top dark bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.putText(img, f"Frame {frame_id:5d}   t = {t_sec:.3f}s",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    cv2.putText(img,
                f"[{idx+1}/{total}]   < LEFT/A    RIGHT/D >   "
                f"SPACE = confirm   Q/ESC = quit",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # Full proxy algorithm (yellow)
    if full_frame is not None:
        col = (0, 220, 220)
        cv2.putText(img,
                    f"FULL algo: frame {full_frame}  t={full_t:.3f}s  "
                    f"(diff {t_sec - full_t:+.3f}s)",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        if frame_id == full_frame:
            cv2.rectangle(img, (W - 18, 0), (W, H), col, -1)
            cv2.putText(img, "FULL", (W - 17, H // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Sternum proxy algorithm (cyan)
    if stern_frame is not None:
        col = (255, 200, 0)
        cv2.putText(img,
                    f"STERN algo: frame {stern_frame}  t={stern_t:.3f}s  "
                    f"(diff {t_sec - stern_t:+.3f}s)",
                    (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        if frame_id == stern_frame:
            cv2.rectangle(img, (W - 36, 0), (W - 18, H), col, -1)
            cv2.putText(img, "STN", (W - 35, H // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    # Bottom: Movesense reference
    diff_ms = t_sec - ms_tap_t
    cv2.putText(img,
                f"Movesense tap: {ms_tap_t:.3f}s   (diff {diff_ms:+.3f}s)",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 200, 255), 1)

    # Red border near Movesense tap
    if abs(diff_ms) < 0.12:
        cv2.rectangle(img, (3, 3), (W - 3, H - 3), (0, 60, 220), 4)

    # Green banner: confirmed
    if confirmed_frame is not None and frame_id == confirmed_frame:
        cv2.rectangle(img, (0, 110), (W, 142), (0, 160, 0), -1)
        cv2.putText(img,
                    f"  TAP CONFIRMED  frame {frame_id}  "
                    f"(SPACE again to change)",
                    (8, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

    return img


# ─────────────────────────────────────────────────────────────
#  SUMMARY PLOT
# ─────────────────────────────────────────────────────────────

def save_summary_plot(t_cam, full_proxy, full_smooth, stern_proxy, stern_smooth,
                      grid_frames, t_ms, ms_proxy, ms_proxy_smooth,
                      full_frame, full_t, stern_frame, stern_t,
                      manual_frame, manual_t, ms_tap_t,
                      dt_full, dt_stern, dt_manual):

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    a = ms_tap_t - 1.5
    b = ms_tap_t + 2.5

    fig, axes = plt.subplots(4, 1, figsize=(14, 17))
    fig.subplots_adjust(top=0.94, hspace=0.45)
    fig.suptitle(
        f"Chest Tap Sync — {REC_ID}\n"
        f"Full algo: fr{full_frame} t={full_t:.3f}s dt={dt_full:+.4f}s   |   "
        f"Sternum algo: fr{stern_frame} t={stern_t:.3f}s dt={dt_stern:+.4f}s   |   "
        f"Manual: fr{manual_frame} t={manual_t:.3f}s dt={dt_manual:+.4f}s",
        fontsize=9
    )

    sw_c = t_cam <= SEARCH_WINDOW_SEC
    sw_m = t_ms  <= SEARCH_WINDOW_SEC

    # Plot 1: full search window — all signals
    ax = axes[0]
    ax.set_title(f"Full search window (0–{SEARCH_WINDOW_SEC}s) — unaligned")
    ax.plot(t_cam[sw_c], norm(full_smooth[sw_c]),
            color="steelblue", lw=1.5, label="Camera full proxy", alpha=0.7)
    ax.plot(t_cam[sw_c], norm(stern_smooth[sw_c]),
            color="mediumorchid", lw=1.5, label="Camera sternum proxy", alpha=0.9)
    ax.plot(t_ms[sw_m],  norm(ms_proxy_smooth[sw_m]),
            color="darkorange", lw=1.5, label="Movesense proxy")
    if full_t is not None:
        ax.axvline(full_t,   color="cyan",        ls="--", lw=1.5,
                   label=f"Full algo: {full_t:.3f}s (fr{full_frame})")
    if stern_t is not None:
        ax.axvline(stern_t,  color="yellow",      ls="--", lw=1.5,
                   label=f"Sternum algo: {stern_t:.3f}s (fr{stern_frame})")
    ax.axvline(manual_t,     color="limegreen",   ls="--", lw=2,
               label=f"Manual: {manual_t:.3f}s (fr{manual_frame})")
    ax.axvline(ms_tap_t,     color="darkorange",  ls="--", lw=2,
               label=f"MS tap: {ms_tap_t:.3f}s")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # Plot 2: zoomed — sternum proxy vs Movesense unaligned
    ax = axes[1]
    ax.set_title("Zoomed — STERNUM proxy vs Movesense (unaligned)")
    mc = (t_cam >= a) & (t_cam <= b)
    mm = (t_ms  >= a) & (t_ms  <= b)
    ax.plot(t_cam[mc], norm(stern_smooth[mc]),
            color="mediumorchid", lw=2, label="Sternum proxy")
    ax.plot(t_ms[mm],  norm(ms_proxy_smooth[mm]),
            color="darkorange", lw=1.5, label="Movesense")
    if stern_t is not None:
        ax.axvline(stern_t, color="yellow",     ls="--", lw=1.8,
                   label=f"Sternum algo: {stern_t:.3f}s")
    ax.axvline(manual_t,    color="limegreen",  ls="--", lw=2,
               label=f"Manual: {manual_t:.3f}s")
    ax.axvline(ms_tap_t,    color="darkorange", ls="--", lw=2,
               label=f"MS: {ms_tap_t:.3f}s")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Plot 3: after alignment with sternum algo dt
    ax = axes[2]
    t_cam_aligned_stern = t_cam + dt_stern
    ax.set_title(f"After alignment — STERNUM algo  (dt={dt_stern:+.4f}s)")
    mc2 = (t_cam_aligned_stern >= a) & (t_cam_aligned_stern <= b)
    ax.plot(t_cam_aligned_stern[mc2], norm(stern_smooth[mc2]),
            color="mediumorchid", lw=2, label="Sternum proxy (aligned)")
    ax.plot(t_ms[mm], norm(ms_proxy_smooth[mm]),
            color="darkorange", lw=1.5, label="Movesense")
    ax.axvline(ms_tap_t, color="black", ls="--", lw=2,
               label=f"Common tap: {ms_tap_t:.3f}s")
    ax.set_xlabel("Time (s) — Movesense clock")
    ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Plot 4: after alignment with manual dt
    ax = axes[3]
    t_cam_aligned_manual = t_cam + dt_manual
    ax.set_title(f"After alignment — MANUAL tap  (dt={dt_manual:+.4f}s)")
    mc3 = (t_cam_aligned_manual >= a) & (t_cam_aligned_manual <= b)
    ax.plot(t_cam_aligned_manual[mc3], norm(full_smooth[mc3]),
            color="steelblue", lw=1.5, label="Full proxy (aligned)", alpha=0.7)
    ax.plot(t_cam_aligned_manual[mc3], norm(stern_smooth[mc3]),
            color="mediumorchid", lw=2, label="Sternum proxy (aligned)")
    ax.plot(t_ms[mm], norm(ms_proxy_smooth[mm]),
            color="darkorange", lw=1.5, label="Movesense")
    ax.axvline(ms_tap_t, color="black", ls="--", lw=2,
               label=f"Common tap: {ms_tap_t:.3f}s")
    ax.set_xlabel("Time (s) — Movesense clock")
    ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    PATH_OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PATH_OUTPUT_PLOT, dpi=150)
    print(f"  Plot saved → {PATH_OUTPUT_PLOT}")
    plt.show(block = False)
# -----------------------------------------------
# THESIS FIGUTE
# --------------------------------
def save_thesis_figure(t_cam, stern_smooth, t_ms, ms_proxy_smooth,
                       stern_t, ms_tap_t, dt_stern,
                       manual_frame, manual_t, dt_manual):
    """
    Clean two-panel thesis figure:
      Left  — signals BEFORE alignment (own clocks)
      Right — signals AFTER alignment (common clock)
 
    Only shows the sternum proxy and Movesense — the two cleanest signals.
    No debug info, no algorithm markers cluttering the plot.
    """
    PATH_THESIS = BASE/ "tap_info"/ "plots_thesis"/ f"thesis_sync_{REC_ID}.png"
 
    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)
    from scipy.ndimage import gaussian_filter1d

    def smooth(x, ms, fs):
        sigma = max((ms / 1000.0) * fs / 2.355, 0.5)  # ms → sigma in samples
        return gaussian_filter1d(x, sigma=sigma)

    # Smooth both signals just for display (does NOT affect dt calculation)
    FS_CAM_EST  = 1.0 / float(np.median(np.diff(t_cam[t_cam > 0])))
    FS_MS_EST   = 52.0

    stern_display = smooth(stern_smooth, ms=80,  fs=FS_CAM_EST)   # ~80ms smooth
    ms_display    = smooth(ms_proxy_smooth, ms=80, fs=FS_MS_EST)  # ~40ms smooth

    # Use manual dt for the thesis (ground truth)
    dt = dt_manual
    tap_t_common = ms_tap_t
 
    # Time window to show: centred on the tap event
    a = tap_t_common - 1.2
    b = tap_t_common + 1.8
 
    t_cam_aligned = t_cam + dt
 
    # Masks for the zoom window
    mc_raw  = (t_cam          >= a) & (t_cam          <= b)
    mc_aln  = (t_cam_aligned  >= a) & (t_cam_aligned  <= b)
    mm      = (t_ms           >= a) & (t_ms           <= b)
 
    # ── Figure setup ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82,
                        bottom=0.14, wspace=0.08)
 
    STERN_COLOR = "#7B52AB"   # purple
    MS_COLOR    = "#E07B00"   # amber
    TAP_COLOR   = "#222222"   # near-black dashed line
 
    lw_sig = 2.0
    lw_tap = 1.6
 
    # ── Left panel: BEFORE alignment ─────────────────────────
    ax1.plot(t_cam[mc_raw], norm(stern_display[mc_raw]),
             color=STERN_COLOR, lw=lw_sig,
             label="Depth camera\n(sternum cells)")
    ax1.plot(t_ms[mm], norm(ms_display[mm]),
             color=MS_COLOR, lw=lw_sig,
             label="Movesense ACC\n(|∇|magnitude)")
 
    # Tap markers on own clocks
    ax1.axvline(manual_t,  color=STERN_COLOR, ls="--", lw=lw_tap, alpha=0.8)
    ax1.axvline(ms_tap_t,  color=MS_COLOR,    ls="--", lw=lw_tap, alpha=0.8)
 
    # Annotation: dt gap
    y_arrow = ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] > 0 else 3.0
    ax1.annotate("",
                 xy=(ms_tap_t, 2.5), xytext=(manual_t, 2.5),
                 arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.8))
    ax1.text((ms_tap_t + manual_t) / 2, 2.7,
             f"Δt = {abs(dt):.3f}s",
             ha="center", va="bottom", fontsize=10, color="gray", fontweight ="bold")
 
    ax1.set_xlim(-0.8, 5.5)
    ax1.set_xlabel("Time (s) — own clocks", fontsize=11)
    ax1.set_ylabel("Normalised amplitude", fontsize=11)
    ax1.set_title("Before synchronisation", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax1.grid(alpha=0.25, lw=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
 
    # ── Right panel: AFTER alignment ─────────────────────────
    ax2.plot(t_cam_aligned[mc_aln], norm(stern_display[mc_aln]),
             color=STERN_COLOR, lw=lw_sig,
             label="Depth camera\n(aligned)")
    ax2.plot(t_ms[mm], norm(ms_display[mm]),
             color=MS_COLOR, lw=lw_sig,
             label="Movesense ACC")
 
    # Common tap line
    ax2.axvline(tap_t_common, color=TAP_COLOR, ls="--", lw=lw_tap,
                label=f"Tap  t = {tap_t_common:.3f}s")
 
    ax2.set_xlim(-0.8, 5.5)
    ax2.set_xlabel("Time (s) — common clock", fontsize=11)
    ax2.set_title("After synchronisation", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax2.grid(alpha=0.25, lw=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(left=False)
 
    # ── Super title ───────────────────────────────────────────
    fig.suptitle(
        f"Chest tap synchronisation — depth camera & Movesense IMU\n"
        f"Recording: {REC_ID}   |   "
        f"Manual tap: frame {manual_frame}   |   "
        f"dt = {dt:+.4f} s",
        fontsize=10, color="#333333"
    )
 
    PATH_THESIS.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PATH_THESIS, dpi=200, bbox_inches="tight")
    print(f"  Thesis figure saved → {PATH_THESIS}")
    plt.show()
 



# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("MANUAL TAP SELECTOR  +  ALGORITHM COMPARISON")
print(f"Recording : {REC_ID}")
print(f"Sternum cells: {STERNUM_CELLS}")
print("=" * 60)

print("\n[1] Timestamps...")
ts_df = load_timestamps(PATH_TIMESTAMPS)
print(f"  {len(ts_df)} frames")

print("\n[2] Movesense ACC...")
ms_tap_t, ms_start_ts, t_ms, ms_proxy, ms_proxy_smooth = load_movesense_tap()

print("\n[3] Depth proxy algorithm (full + sternum)...")
full_result, stern_result = run_depth_algorithm(ts_df)
(full_frame,  full_t,  full_method,
 full_proxy,  t_cam,   grid_frames, full_smooth)  = full_result
(stern_frame, stern_t, stern_method,
 stern_proxy, _,       _,           stern_smooth) = stern_result

print("\n[4] Color frames in search window...")
frames = get_frames_in_window(ts_df)
print(f"  {len(frames)} frames  (t=0 to {frames[-1][1]:.2f}s)")

# Start ~1s before Movesense tap
start_idx = 0
for i, (fid, t, p) in enumerate(frames):
    if t >= ms_tap_t - 1.0:
        start_idx = max(0, i - 5)
        break

print(f"\n  Opening viewer at frame index {start_idx}...")
print("  Controls:  LEFT/A = back   RIGHT/D = forward   "
      "SPACE = confirm   Q/ESC = quit")
print(f"  YELLOW/CYAN bars = algorithm taps")
print(f"  RED border       = within 0.12s of Movesense ({ms_tap_t:.3f}s)")
print(f"  GREEN banner     = confirmed frame\n")

# ── Interactive viewer ────────────────────────────────────────
cv2.namedWindow("Tap Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tap Selector", DISPLAY_WIDTH, DISPLAY_HEIGHT)

idx             = start_idx
confirmed_frame = None
confirmed_t     = None
cache           = {}

def get_frame_img(entry):
    fid, t, path = entry
    if fid not in cache:
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Cannot load frame {fid}",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cache[fid] = img
    return cache[fid]

while True:
    fid, t_sec, _ = frames[idx]
    raw  = get_frame_img(frames[idx])
    disp = draw_overlay(raw, fid, t_sec, idx, len(frames),
                        ms_tap_t,
                        full_frame, full_t,
                        stern_frame, stern_t,
                        confirmed_frame)
    cv2.imshow("Tap Selector", cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    cv2.waitKey(1)   # force render on Windows

    key = cv2.waitKey(0) & 0xFF

    if key in (2, 81, ord('a'), ord('A')):
        idx = max(0, idx - 1)
    elif key in (3, 83, ord('d'), ord('D')):
        idx = min(len(frames) - 1, idx + 1)
    elif key == ord(' '):
        confirmed_frame = fid
        confirmed_t     = t_sec
        print(f"  ✓ Tap confirmed: frame {fid}  t={t_sec:.4f}s")
    elif key in (27, ord('q'), ord('Q')):
        break
    if key != 255:
        print(f"  DEBUG key={key}  char={chr(key) if 32<=key<127 else '?'}")

cv2.destroyAllWindows()

# ── Results ───────────────────────────────────────────────────
if confirmed_frame is None:
    print("\n  No tap confirmed — nothing saved.")
else:
    dt_manual = ms_tap_t - confirmed_t
    dt_full   = (ms_tap_t - full_t)  if full_t  is not None else None
    dt_stern  = (ms_tap_t - stern_t) if stern_t is not None else None

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'─'*60}")
    if full_frame is not None:
        print(f"  Full proxy algo  : frame {full_frame:4d}  t={full_t:.4f}s  "
              f"dt={dt_full:+.4f}s  [{full_method}]")
    if stern_frame is not None:
        print(f"  Sternum algo     : frame {stern_frame:4d}  t={stern_t:.4f}s  "
              f"dt={dt_stern:+.4f}s  [{stern_method}]")
    print(f"  Manual tap       : frame {confirmed_frame:4d}  t={confirmed_t:.4f}s  "
          f"dt={dt_manual:+.4f}s")
    print(f"  Movesense tap    : t={ms_tap_t:.4f}s")
    print(f"{'─'*60}")

    for label, frame_algo, t_algo in [
        ("Full vs Manual",    full_frame,  full_t),
        ("Sternum vs Manual", stern_frame, stern_t),
    ]:
        if frame_algo is not None:
            fd = abs(confirmed_frame - frame_algo)
            td = abs(confirmed_t - t_algo)
            status = "OK" if fd == 0 else ("CLOSE" if fd <= 2 else "WRONG")
            print(f"  {label}: {fd} frame(s) ({td:.4f}s) — {status}")

    print(f"{'='*60}")
    print(f"  Using MANUAL tap. dt = {dt_manual:+.4f}s")
    print(f"{'='*60}")

    # Save JSON
    PATH_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "rec_id"           : REC_ID,
        "manual_tap_frame" : int(confirmed_frame),
        "manual_tap_sec"   : float(confirmed_t),
        "full_algo_frame"  : int(full_frame)  if full_frame  is not None else None,
        "full_algo_sec"    : float(full_t)    if full_t      is not None else None,
        "full_algo_method" : full_method,
        "stern_algo_frame" : int(stern_frame) if stern_frame is not None else None,
        "stern_algo_sec"   : float(stern_t)   if stern_t    is not None else None,
        "stern_algo_method": stern_method,
        "ms_tap_sec"       : float(ms_tap_t),
        "ms_start_ts"      : int(ms_start_ts),
        "offset_sec"       : float(dt_manual),
        "offset_full_algo" : float(dt_full)  if dt_full  is not None else None,
        "offset_stern_algo": float(dt_stern) if dt_stern is not None else None,
        "method"           : "manual_color_frame_selection",
        "sternum_cells"    : STERNUM_CELLS,
    }
    with open(PATH_OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\n  Saved → {PATH_OUTPUT_JSON}")

    # Summary plot
    if full_proxy is not None:
        save_summary_plot(
            t_cam, full_proxy, full_smooth, stern_proxy, stern_smooth,
            grid_frames, t_ms, ms_proxy, ms_proxy_smooth,
            full_frame, full_t, stern_frame, stern_t,
            confirmed_frame, confirmed_t, ms_tap_t,
            dt_full, dt_stern, dt_manual
        )
    # Thesis figure (clean two-panel version)
    if stern_smooth is not None:
        save_thesis_figure(
            t_cam, stern_smooth, t_ms, ms_proxy_smooth,
            stern_t, ms_tap_t, dt_stern,
            confirmed_frame, confirmed_t, dt_manual
        )   