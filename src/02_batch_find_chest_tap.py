"""
02_batch_tap_selector.py
========================
Runs the manual tap selector across all recordings automatically.
Auto-discovers recordings from D:\\recordings — no manual list needed.

For each recording:
  1. Loads timestamps, Movesense ACC, depth grid
  2. Opens the interactive color frame viewer
  3. You confirm the tap with SPACE
  4. Saves JSON + debug plot + thesis figure
  5. Moves to the next recording automatically

Progress is saved after each recording — if you stop midway
just run again and already-processed recordings are skipped.

Controls (per recording):
  A / D    : previous / next frame
  SPACE    : confirm tap frame
  Q / ESC  : skip this recording (no save) and move to next
  CTRL+C   : stop batch entirely
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
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE           = Path(r"C:\Projects\thesis\data")
RECORDINGS_DIR = Path(r"D:\recordings")
PATH_PROGRESS  = BASE / "tap_info" / "batch_progress.json"

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
SEARCH_WINDOW_SEC = 10.0
DISPLAY_WIDTH     = 1024
DISPLAY_HEIGHT    = 600
FS_MS_ACC         = 52.0
STERNUM_CELLS     = [5, 6, 9, 10, 13, 14]
TAP_SMOOTH_MS_CAM = 0.0
TAP_SMOOTH_MS_MS  = 0.0
PROMINENCE_SIGMA  = 2.0
MIN_DISTANCE_SEC  = 0.25


# ─────────────────────────────────────────────────────────────
#  AUTO-DISCOVER RECORDINGS
# ─────────────────────────────────────────────────────────────

def discover_recordings(recordings_dir):
    """
    Scan recordings_dir for valid recording folders.
    A valid folder has: color/ subfolder + timestamps.csv
    Folder name must follow format: SUBJECT_DIST_CLOTH
    Returns list of (subject, dist, cloth, rec_id, folder_path)
    """
    folders = sorted([
        f for f in recordings_dir.iterdir()
        if f.is_dir()
        and (f / "color").exists()
        and (f / "timestamps.csv").exists()
        and not f.name.startswith("Camera_")
    ])

    recordings = []
    for f in folders:
        parts = f.name.split("_")
        if len(parts) == 3:
            subject, dist, cloth = parts
            recordings.append((subject, dist, cloth, f.name, f))
        else:
            print(f"  [WARN] Skipping unexpected folder name: {f.name}")

    return recordings


# ─────────────────────────────────────────────────────────────
#  SHARED FUNCTIONS
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


def build_proxy_from_cells(X, cell_indices=None):
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
        offset       = 1 if valley_local + 1 < len(region) else 0
        contact_gi   = start + valley_local + offset
        return int(grid_frames[contact_gi]), float(t[contact_gi]), "post_valley_fallback"


def load_movesense_tap(acc_file):
    with open(acc_file, 'r') as f:
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
    return tap_t, int(start_ts), t_ms, proxy, ms_proxy_smooth


def run_depth_algorithm(path_grid, ts_df):
    if not path_grid.exists():
        return None, None, "grid_missing", None, None, None, None, None, None, None, None, None
    df_cam      = pd.read_csv(path_grid)
    grid_frames = df_cam["frame"].values.astype(int)
    X           = df_cam.drop(columns=["frame"]).values.astype(float)
    grid_timestamps = ts_df.loc[grid_frames, "depth_ts"].values
    t_cam    = (grid_timestamps - grid_timestamps[0]) / 1000.0
    intervals = np.diff(t_cam)
    FS_CAM    = float(1.0 / np.median(intervals[intervals > 0]))

    full_proxy = build_proxy_from_cells(X, cell_indices=None)
    full_idx, full_times, full_smooth = find_tap_peaks(
        full_proxy, t_cam, FS_CAM,
        SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_CAM, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
    )
    full_frame, full_t, full_method = select_tap_camera(
        full_idx, full_times, full_proxy, t_cam, grid_frames
    )

    valid_sternum = [c for c in STERNUM_CELLS if c < X.shape[1]]
    stern_proxy   = build_proxy_from_cells(X, cell_indices=valid_sternum)
    stern_idx, stern_times, stern_smooth = find_tap_peaks(
        stern_proxy, t_cam, FS_CAM,
        SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_CAM, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
    )
    stern_frame, stern_t, stern_method = select_tap_camera(
        stern_idx, stern_times, stern_proxy, t_cam, grid_frames
    )

    return (full_frame, full_t, full_method, full_proxy, full_smooth,
            stern_frame, stern_t, stern_method, stern_proxy, stern_smooth,
            t_cam, grid_frames)


def get_frames_in_window(color_dir, ts_df):
    jpgs = sorted(color_dir.glob("frame_*.jpg"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not jpgs:
        raise FileNotFoundError(f"No frame_*.jpg in {color_dir}")
    frame_ids  = np.array([int(p.stem.split("_")[1]) for p in jpgs])
    valid      = np.isin(frame_ids, ts_df.index.values)
    jpgs       = [p for p, v in zip(jpgs, valid) if v]
    frame_ids  = frame_ids[valid]
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
                 ms_tap_t, full_frame, full_t,
                 stern_frame, stern_t, confirmed_frame,
                 rec_id, rec_num, rec_total):
    img = frame_bgr.copy()
    H, W = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Recording progress
    cv2.putText(img,
                f"Recording {rec_num}/{rec_total}: {rec_id}",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Frame info
    cv2.putText(img, f"Frame {frame_id:5d}   t = {t_sec:.3f}s",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

    # Controls
    cv2.putText(img,
                f"[{idx+1}/{total}]  A = back   D = forward   "
                f"SPACE = confirm   Q = skip recording",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Full algo (yellow)
    if full_frame is not None:
        cv2.putText(img,
                    f"FULL algo: fr{full_frame}  t={full_t:.3f}s  "
                    f"(diff {t_sec - full_t:+.3f}s)",
                    (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
        if frame_id == full_frame:
            cv2.rectangle(img, (W-18, 0), (W, H), (0, 220, 220), -1)
            cv2.putText(img, "FULL", (W-17, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Sternum algo (cyan)
    if stern_frame is not None:
        cv2.putText(img,
                    f"STERN algo: fr{stern_frame}  t={stern_t:.3f}s  "
                    f"(diff {t_sec - stern_t:+.3f}s)",
                    (12, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        if frame_id == stern_frame:
            cv2.rectangle(img, (W-36, 0), (W-18, H), (255, 200, 0), -1)
            cv2.putText(img, "STN", (W-35, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Movesense reference (bottom)
    diff_ms = t_sec - ms_tap_t
    cv2.putText(img,
                f"Movesense tap: {ms_tap_t:.3f}s   (diff {diff_ms:+.3f}s)",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 200, 255), 1)

    # Red border near Movesense
    if abs(diff_ms) < 0.12:
        cv2.rectangle(img, (3, 3), (W-3, H-3), (0, 60, 220), 4)

    # Green banner confirmed
    if confirmed_frame is not None and frame_id == confirmed_frame:
        cv2.rectangle(img, (0, 130), (W, 162), (0, 160, 0), -1)
        cv2.putText(img,
                    f"  TAP CONFIRMED  frame {frame_id}  (SPACE to change)",
                    (8, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

    return img


# ─────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

def save_outputs(rec_id, confirmed_frame, confirmed_t,
                 full_frame, full_t, full_method,
                 stern_frame, stern_t, stern_method,
                 ms_tap_t, ms_start_ts,
                 t_cam, full_proxy, full_smooth,
                 stern_proxy, stern_smooth,
                 t_ms, ms_proxy, ms_proxy_smooth):

    dt_manual = ms_tap_t - confirmed_t
    dt_full   = (ms_tap_t - full_t)  if full_t  is not None else None
    dt_stern  = (ms_tap_t - stern_t) if stern_t is not None else None

    # ── JSON ─────────────────────────────────────────────────
    path_json = BASE / "tap_info" / "json" / f"Tap_info_{rec_id}.json"
    path_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "rec_id"           : rec_id,
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
    with open(path_json, "w") as f:
        json.dump(out, f, indent=4)
    print(f"  JSON    → {path_json}")

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    a = ms_tap_t - 1.5
    b = ms_tap_t + 2.5
    sw_c = t_cam <= SEARCH_WINDOW_SEC
    sw_m = t_ms  <= SEARCH_WINDOW_SEC
    mc   = (t_cam >= a) & (t_cam <= b)
    mm   = (t_ms  >= a) & (t_ms  <= b)

    # ── Debug plot ────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 17))
    fig.subplots_adjust(top=0.94, hspace=0.45)
    fig.suptitle(
        f"Chest Tap Sync — {rec_id}\n"
        f"Full: fr{full_frame} dt={dt_full:+.4f}s   |   "
        f"Sternum: fr{stern_frame} dt={dt_stern:+.4f}s   |   "
        f"Manual: fr{confirmed_frame} dt={dt_manual:+.4f}s",
        fontsize=9
    )

    ax = axes[0]
    ax.set_title("Full search window — unaligned")
    ax.plot(t_cam[sw_c], norm(full_smooth[sw_c]),      color="steelblue",    lw=1.5, label="Full proxy",    alpha=0.7)
    ax.plot(t_cam[sw_c], norm(stern_smooth[sw_c]),     color="mediumorchid", lw=1.5, label="Sternum proxy", alpha=0.9)
    ax.plot(t_ms[sw_m],  norm(ms_proxy_smooth[sw_m]),  color="darkorange",   lw=1.5, label="Movesense")
    if full_t:   ax.axvline(full_t,       color="cyan",       ls="--", lw=1.5, label=f"Full: {full_t:.3f}s")
    if stern_t:  ax.axvline(stern_t,      color="yellow",     ls="--", lw=1.5, label=f"Stern: {stern_t:.3f}s")
    ax.axvline(confirmed_t, color="limegreen",  ls="--", lw=2, label=f"Manual: {confirmed_t:.3f}s")
    ax.axvline(ms_tap_t,    color="darkorange", ls="--", lw=2, label=f"MS: {ms_tap_t:.3f}s")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Norm. amplitude")

    ax = axes[1]
    ax.set_title("Zoomed — sternum proxy vs Movesense (unaligned)")
    ax.plot(t_cam[mc], norm(stern_smooth[mc]),      color="mediumorchid", lw=2,   label="Sternum proxy")
    ax.plot(t_ms[mm],  norm(ms_proxy_smooth[mm]),   color="darkorange",   lw=1.5, label="Movesense")
    if stern_t: ax.axvline(stern_t,      color="yellow",     ls="--", lw=1.8, label=f"Stern: {stern_t:.3f}s")
    ax.axvline(confirmed_t, color="limegreen",  ls="--", lw=2, label=f"Manual: {confirmed_t:.3f}s")
    ax.axvline(ms_tap_t,    color="darkorange", ls="--", lw=2, label=f"MS: {ms_tap_t:.3f}s")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Norm. amplitude")

    ax = axes[2]
    t_aln_stern = t_cam + dt_stern if dt_stern is not None else t_cam
    mc2 = (t_aln_stern >= a) & (t_aln_stern <= b)
    ax.set_title(f"After alignment — sternum algo (dt={dt_stern:+.4f}s)" if dt_stern else "Sternum algo N/A")
    ax.plot(t_aln_stern[mc2], norm(stern_smooth[mc2]), color="mediumorchid", lw=2,   label="Sternum (aligned)")
    ax.plot(t_ms[mm],         norm(ms_proxy_smooth[mm]), color="darkorange", lw=1.5, label="Movesense")
    ax.axvline(ms_tap_t, color="black", ls="--", lw=2, label=f"Common tap: {ms_tap_t:.3f}s")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("Time (s) — Movesense clock"); ax.set_ylabel("Norm. amplitude")

    ax = axes[3]
    t_aln_man = t_cam + dt_manual
    mc3 = (t_aln_man >= a) & (t_aln_man <= b)
    ax.set_title(f"After alignment — manual (dt={dt_manual:+.4f}s)")
    ax.plot(t_aln_man[mc3], norm(full_smooth[mc3]),  color="steelblue",    lw=1.5, label="Full (aligned)",    alpha=0.7)
    ax.plot(t_aln_man[mc3], norm(stern_smooth[mc3]), color="mediumorchid", lw=2,   label="Sternum (aligned)")
    ax.plot(t_ms[mm],       norm(ms_proxy_smooth[mm]), color="darkorange", lw=1.5, label="Movesense")
    ax.axvline(ms_tap_t, color="black", ls="--", lw=2, label=f"Common tap: {ms_tap_t:.3f}s")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("Time (s) — Movesense clock"); ax.set_ylabel("Norm. amplitude")

    path_debug = BASE / "tap_info" / "plots_debug" / f"tap_sync_{rec_id}.png"
    path_debug.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_debug, dpi=150)
    plt.close()
    print(f"  Debug   → {path_debug}")

    # ── Thesis figure ─────────────────────────────────────────
    def smooth_sig(x, ms, fs):
        sigma = max((ms / 1000.0) * fs / 2.355, 0.5)
        return gaussian_filter1d(x, sigma=sigma)

    FS_CAM_EST = 1.0 / float(np.median(np.diff(t_cam[t_cam > 0])))
    stern_disp = smooth_sig(stern_smooth,    ms=80, fs=FS_CAM_EST)
    ms_disp    = smooth_sig(ms_proxy_smooth, ms=80, fs=52.0)

    t_cam_aln = t_cam + dt_manual
    mc_r = (t_cam     >= a) & (t_cam     <= b)
    mc_a = (t_cam_aln >= a) & (t_cam_aln <= b)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig2.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.14, wspace=0.08)

    SC = "#7B52AB"; MC = "#E07B00"; TC = "#222222"

    ax1.plot(t_cam[mc_r], norm(stern_disp[mc_r]), color=SC, lw=2, label="Depth camera\n(sternum cells)")
    ax1.plot(t_ms[mm],    norm(ms_disp[mm]),       color=MC, lw=2, label="Movesense ACC\n(|∇|magnitude)")
    ax1.axvline(confirmed_t, color=SC, ls="--", lw=1.6, alpha=0.8)
    ax1.axvline(ms_tap_t,    color=MC, ls="--", lw=1.6, alpha=0.8)
    ax1.annotate("", xy=(ms_tap_t, -0.3), xytext=(confirmed_t, -0.3),
                 arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.8))
    ax1.text((ms_tap_t + confirmed_t) / 2, -0.15,
             f"Δt = {abs(dt_manual):.3f} s",
             ha="center", va="bottom", fontsize=10, color="dimgray", fontweight="bold")
    ax1.set_xlim(a, b); ax1.set_ylim(-1.0, 5.2)
    ax1.set_xlabel("Time (s) — own clocks", fontsize=11)
    ax1.set_ylabel("Normalised amplitude", fontsize=11)
    ax1.set_title("Before synchronisation", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(t_cam_aln[mc_a], norm(stern_disp[mc_a]), color=SC, lw=2, label="Depth camera\n(aligned)")
    ax2.plot(t_ms[mm],        norm(ms_disp[mm]),       color=MC, lw=2, label="Movesense ACC")
    ax2.axvline(ms_tap_t, color=TC, ls="--", lw=1.6, label=f"Tap  t = {ms_tap_t:.3f}s")
    ax2.set_xlim(a, b); ax2.set_ylim(-1.0, 5.2)
    ax2.set_xlabel("Time (s) — common clock", fontsize=11)
    ax2.set_title("After synchronisation", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(left=False)

    fig2.suptitle(
        f"Chest tap synchronisation — depth camera & Movesense IMU\n"
        f"Recording: {rec_id}   |   Manual tap: frame {confirmed_frame}   |   "
        f"dt = {dt_manual:+.4f} s",
        fontsize=10, color="#333333"
    )

    path_thesis = BASE / "tap_info" / "plots_thesis" / f"thesis_sync_{rec_id}.png"
    path_thesis.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_thesis, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Thesis  → {path_thesis}")

    return dt_manual


# ─────────────────────────────────────────────────────────────
#  PROGRESS TRACKING
# ─────────────────────────────────────────────────────────────

def load_progress():
    if PATH_PROGRESS.exists():
        with open(PATH_PROGRESS) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    PATH_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_PROGRESS, "w") as f:
        json.dump(progress, f, indent=4)


# ─────────────────────────────────────────────────────────────
#  MAIN BATCH LOOP
# ─────────────────────────────────────────────────────────────

# Discover recordings
print("Discovering recordings ...")
try:
    recordings = discover_recordings(RECORDINGS_DIR)
except Exception as e:
    print(f"ERROR during discovery: {e}")
    import traceback; traceback.print_exc()
    input("Press Enter to exit")
    raise SystemExit(1)
total_recs = len(recordings)
print(f"Discovery complete: {total_recs} recordings found")

progress = load_progress()

print("=" * 60)
print(f"BATCH TAP SELECTOR")
print(f"Found {total_recs} recordings in {RECORDINGS_DIR}")
print("=" * 60)

done_count = sum(1 for _, _, _, rec_id, _ in recordings if progress.get(rec_id) == "done")
print(f"  Already processed : {done_count}/{total_recs}")
print(f"  Remaining         : {total_recs - done_count}")
print()

cv2.namedWindow("Tap Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tap Selector", DISPLAY_WIDTH, DISPLAY_HEIGHT)

for rec_num, (subject, dist, cloth, rec_id, rec_folder) in enumerate(recordings, 1):

    # Skip if already done
    if progress.get(rec_id) == "done":
        print(f"  [{rec_num}/{total_recs}] SKIP {rec_id} (already done)")
        continue

    print(f"\n{'='*60}")
    print(f"  [{rec_num}/{total_recs}] {rec_id}")
    print(f"{'='*60}")

    path_color      = rec_folder / "color"
    path_timestamps = rec_folder / "timestamps.csv"
    path_grid       = BASE / "GRID_files" / f"GRID_{rec_id}.csv"
    path_ms_acc     = BASE / "movesense" / rec_id / "acc_stream.json"

    # Load timestamps
    try:
        ts_df = load_timestamps(path_timestamps)
    except Exception as e:
        print(f"  ERROR timestamps: {e} — skipping")
        progress[rec_id] = "error_timestamps"
        save_progress(progress)
        continue

    # Load Movesense
    try:
        ms_tap_t, ms_start_ts, t_ms, ms_proxy, ms_proxy_smooth = load_movesense_tap(path_ms_acc)
        print(f"  Movesense tap : {ms_tap_t:.4f}s")
    except Exception as e:
        print(f"  ERROR Movesense: {e} — skipping")
        progress[rec_id] = "error_movesense"
        save_progress(progress)
        continue

    # Run depth algorithm
    try:
        result = run_depth_algorithm(path_grid, ts_df)
        (full_frame, full_t, full_method, full_proxy, full_smooth,
         stern_frame, stern_t, stern_method, stern_proxy, stern_smooth,
         t_cam, grid_frames) = result
        print(f"  Full algo     : frame {full_frame}  t={full_t:.4f}s  [{full_method}]")
        print(f"  Sternum algo  : frame {stern_frame}  t={stern_t:.4f}s  [{stern_method}]")
    except Exception as e:
        print(f"  ERROR algorithm: {e} — skipping")
        progress[rec_id] = "error_algorithm"
        save_progress(progress)
        continue

    # Load color frames
    try:
        frames = get_frames_in_window(path_color, ts_df)
        print(f"  Color frames  : {len(frames)} in search window")
    except Exception as e:
        print(f"  ERROR color frames: {e} — skipping")
        progress[rec_id] = "error_color_frames"
        save_progress(progress)
        continue

    # Start near Movesense tap
    start_idx = 0
    for i, (fid, t, p) in enumerate(frames):
        if t >= ms_tap_t - 1.0:
            start_idx = max(0, i - 5)
            break

    # Interactive viewer
    idx             = start_idx
    confirmed_frame = None
    confirmed_t     = None
    cache           = {}
    skipped         = False

    def get_frame_img(entry):
        fid, t, path = entry
        if fid not in cache:
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            cache[fid] = img
        return cache[fid]

    while True:
        fid, t_sec, _ = frames[idx]
        raw  = get_frame_img(frames[idx])
        disp = draw_overlay(raw, fid, t_sec, idx, len(frames),
                            ms_tap_t, full_frame, full_t,
                            stern_frame, stern_t, confirmed_frame,
                            rec_id, rec_num, total_recs)
        cv2.imshow("Tap Selector", cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        cv2.waitKey(1)

        key = cv2.waitKey(0) & 0xFF

        if key in (2, 81, ord('a'), ord('A')):
            idx = max(0, idx - 1)
        elif key in (3, 83, ord('d'), ord('D')):
            idx = min(len(frames) - 1, idx + 1)
        elif key == ord(' '):
            if confirmed_frame == fid:
               break
            else:
               confirmed_frame = fid
               confirmed_t     = t_sec
               print(f"  Tap confirmed : frame {fid}  t={t_sec:.4f}s (PRESS SPACE again to save)")
    
        
        
        elif key in (27, ord('q'), ord('Q')):
            print(f"  Skipped {rec_id}")
            skipped = True
            break

    if skipped or confirmed_frame is None:
        progress[rec_id] = "skipped"
        save_progress(progress)
        continue

    # Save outputs
    dt = save_outputs(
        rec_id, confirmed_frame, confirmed_t,
        full_frame, full_t, full_method,
        stern_frame, stern_t, stern_method,
        ms_tap_t, ms_start_ts,
        t_cam, full_proxy, full_smooth,
        stern_proxy, stern_smooth,
        t_ms, ms_proxy, ms_proxy_smooth
    )

    progress[rec_id] = "done"
    save_progress(progress)
    print(f"  dt = {dt:+.4f}s")
    print(f"  [{rec_num}/{total_recs}] DONE — {total_recs - rec_num} remaining")

cv2.destroyAllWindows()

# Final summary
print(f"\n{'='*60}")
print(f"  BATCH COMPLETE")
done_list    = [k for k, v in progress.items() if v == "done"]
skipped_list = [k for k, v in progress.items() if v == "skipped"]
error_list   = [k for k, v in progress.items() if v.startswith("error")]
print(f"  Done    : {len(done_list)}")
print(f"  Skipped : {len(skipped_list)}")
print(f"  Errors  : {len(error_list)}")
if skipped_list:
    print(f"  Skipped recordings : {skipped_list}")
if error_list:
    print(f"  Error recordings   : {error_list}")
    
print(f"{'='*60}")
input("Press Enter to exit")