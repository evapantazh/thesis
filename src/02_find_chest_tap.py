"""
02_find_chest_tap.py
====================
Detect chest tap in both camera grid signal and Movesense ECG,
compute the time offset dt, and save to JSON.

Protocol: ONE firm tap on sternum at the start of recording.
For Sub01 which has a double tap, SEARCH_WINDOW_SEC is set wide
enough to see both, and we pick the first coherent peak in camera
and the first in Movesense — then verify visually in Plot 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  FLAGS
# ─────────────────────────────────────────────────────────────
PLOT_SYNC = True

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Projects\thesis\data")
SUBJECT  = "Sub03"
DIST     = "800"
CLOTH    = "Tshirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"

PATH_CAMERA_GRID = BASE / "GRID_files"/ f"GRID_{REC_ID}.CSV"
PATH_MOVESENSE   = BASE / f"Movesense_{REC_ID}"
PATH_MOVESENSE_ACC = PATH_MOVESENSE / f"{REC_ID}_acc_stream.json"
PATH_MOVESENSE_ECG = PATH_MOVESENSE / f"{REC_ID}_ecg_stream.json"
PATH_OUTPUT_TAP_JSON = BASE / "tap_info" / f"Tap_info_{REC_ID}.json"


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
# Set to 3.0 for Sub01 (double tap — want to see both peaks and pick first).
# For future subjects with single tap: 2.0 is enough.
SEARCH_WINDOW_SEC  = 8.0   # covers both taps for Sub01 (fr16@0.5s, fr28@0.9s)

# Smoothing before peak detection.
# At 30fps: 150ms = 4.5 samples (sigma=4) — removes noise, keeps tap shape.
TAP_SMOOTH_MS_CAM  = 0.0     # no smoothing — coherence proxy is already clean
TAP_SMOOTH_MS_MS   = 20.0    # ms (Movesense 125Hz)

# How many std above local noise a peak must be to count as a tap.
PROMINENCE_SIGMA   = 2.0

# Minimum gap between two peaks (avoids counting same tap twice).
MIN_DISTANCE_SEC   = 0.25

FS_MS_ACC = 52.0  # The new sample rate from Movesense Showcase

#FS_CAM = 30.0   # camera nominal fps
# Instead of a fixed FPS for the cqmera, compute it through timestamps
# in the main section


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def robust_fs(t):
    t  = np.asarray(t, float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / np.median(dt)

def normalize_zscore(x):
    x = np.asarray(x, float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


# ─────────────────────────────────────────────────────────────
#  PROXY BUILDERS
# ─────────────────────────────────────────────────────────────

def build_camera_proxy(X):
    """
    Build a 1D tap proxy from the (T x N_cells) depth grid matrix.

    A chest tap moves ALL cells simultaneously in the SAME direction
    (coherent). Noise and breathing move cells independently (incoherent).

    proxy = energy x coherence
      energy    = sum(abs(gradient))  -- total motion magnitude
      coherence = abs(mean(gradient)) -- directional agreement across cells

    Product is high ONLY when all cells agree = real tap.
    Noise: high energy, near-zero coherence -> suppressed.
    Breathing: slow gradient -> near-zero energy -> suppressed.
    """
    X = np.asarray(X, float).copy()

    # Replace invalid pixels with column median
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = (X[:, j] == 0) | ~np.isfinite(X[:, j])
        X[bad, j] = col_med[j]

    dX   = np.diff(X, axis=0, prepend = X[[0]])          # T x N_cells, forward looking

    energy    = np.sum(np.abs(dX), axis=1)    # T
    coherence = np.abs(np.mean(dX, axis=1))   # T

    return energy * coherence                  # T


def build_movesense_signal(acc_file):
    with open(acc_file, 'r') as f:
        data = json.load(f)['data']
    
    start_ts = data[0]['acc']['Timestamp']
    all_timestamps = []
    all_magnitudes = []

    for entry in data:
        ts = entry['acc']['Timestamp']
        samples = entry['acc']['ArrayAcc']
        for i, s in enumerate(samples):
            # Calculate 3D magnitude to find the sharp mechanical impact
            mag = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            all_magnitudes.append(mag)
            # Movesense packet internal timing interpolation
            all_timestamps.append(ts + (i * (1000.0 / 52.0))) 

    # Convert the whole time list to "Seconds from Start" to match Camera t
    all_timestamps = np.array(all_timestamps)
    t_ms_relative = (all_timestamps - start_ts) / 1000.0

    ms_proxy = np.abs(np.gradient(all_magnitudes))
    return t_ms_relative, ms_proxy, start_ts


# ─────────────────────────────────────────────────────────────
#  PEAK DETECTION
# ─────────────────────────────────────────────────────────────

def find_tap_peaks(proxy, t, fs,
                   search_window_sec,
                   smooth_ms,
                   prominence_sigma,
                   min_distance_sec):
    """
    Find all tap candidate peaks within the search window.
    Returns global indices, times, full smoothed proxy, and debug dict.
    """
    proxy = np.asarray(proxy, float)
    t     = np.asarray(t, float)

    mask  = t <= search_window_sec
    x     = proxy[mask]

    sigma = max(int((smooth_ms / 1000.0) * fs), 0)
    x_s          = gaussian_filter1d(x, sigma=sigma) if sigma > 0 else x.copy()
    proxy_smooth = gaussian_filter1d(proxy, sigma=sigma) if sigma > 0 else proxy.copy()

    prom  = np.std(x_s) * prominence_sigma
    dist  = max(int(min_distance_sec * fs), 1)
    peaks, props = signal.find_peaks(x_s, prominence=prom, distance=dist)

    global_indices = np.where(mask)[0][peaks]
    peak_times     = t[global_indices]

    debug = {
        "sigma_samples"        : int(sigma),
        "prominence_threshold" : float(prom),
        "min_distance_samples" : int(dist),
        "num_peaks_found"      : int(len(peaks)),
        "all_peak_times_sec"   : peak_times.tolist(),
        "all_peak_values"      : proxy[global_indices].tolist(),
    }

    return global_indices, peak_times, proxy_smooth, debug


def select_tap(global_indices, peak_times, proxy, label="signal"):
    """
    Single-tap protocol: pick the FIRST (earliest) peak. # maybe pick the biggest insteaf of first?????
    Within a short search window the tap always comes before
    any subject movement or settling peaks.
    """
    if len(global_indices) == 0:
        raise ValueError(
            f"No tap peak found in [{label}]!\n"
            f"  Try: lower PROMINENCE_SIGMA (now={PROMINENCE_SIGMA}, try 1.5)\n"
            f"  Try: widen SEARCH_WINDOW_SEC (now={SEARCH_WINDOW_SEC})\n"
            f"  Check: was the tap within the first {SEARCH_WINDOW_SEC}s?"
        )

    if len(global_indices) > 1:
        print(f"  [INFO] {len(global_indices)} peaks in [{label}]- picking FIRST")
        print(f"         All peaks: {[f't={tp:.3f}s' for tp in peak_times]}")
    
    # Pick largest peak
    #best = np.argmax(proxy[global_indices])

    return global_indices[0], float(peak_times[0])

def subframe_peak_time(proxy, peak_idx, t):
    """Parabolic interpolation for sub-frame tap timing."""
    i = int(peak_idx)
    if i <= 0 or i >= len(proxy) - 1:
        return float(t[i])
    
    y0, y1, y2 = float(proxy[i-1]), float(proxy[i]), float(proxy[i+1])
    # Parabola vertex offset from i: delta = (y0 - y2) / (2*(y0 - 2*y1 + y2))
    denom = y0 - 2.0*y1 + y2
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = np.clip((y0 - y2) / (2.0 * denom), -0.5, 0.5)
    
    # Interpolate time linearly between neighboring timestamps
    if delta >= 0:
        dt_step = float(t[i+1] - t[i])
    else:
        dt_step = float(t[i] - t[i-1])
    
    t_peak = float(t[i]) + delta * dt_step
    
    # Forward-diff peaks land on the frame BEFORE the jump.
    # Shift forward by half a frame interval to estimate true contact moment.
    half_frame = float(t[i+1] - t[i]) / 2.0 if i+1 < len(t) else 0.0

    return t_peak + half_frame
# ─────────────────────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────────────────────

def save_tap_info(path, cam_tap, ecg_tap, dt, cam_dbg, ecg_dbg, ms_start_ts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "cam_tap_sec"  : float(cam_tap),
        "ecg_tap_sec"  : float(ecg_tap),
        "ms_start_ts"  : int(ms_start_ts),
        "offset_sec"   : float(dt),
        "protocol"     : "single_tap_first_peak",
        "camera_debug" : cam_dbg,
        "ecg_debug"    : ecg_dbg,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("CHEST TAP DETECTION")
print("=" * 60)

# 1. Camera
print("\n[1] Camera grid...")
df_cam = pd.read_csv(PATH_CAMERA_GRID)
grid_frames = df_cam["frame"].values.astype(int)
X      = df_cam.drop(columns=["frame"]).values.astype(float)

# Rebuild t_cam from grid frames only
ts_df_grid = pd.read_csv(BASE / f"Camera_{REC_ID}" / "timestamps.csv")
ts_df_grid = ts_df_grid.set_index("frame")
grid_timestamps = ts_df_grid.loc[grid_frames, "timestamp"].values
t_cam = (grid_timestamps - grid_timestamps[0]) / 1000.0

# Recompute FS_CAM from grid timestamps
intervals_grid = np.diff(t_cam)
FS_CAM = float(1.0 / np.median(intervals_grid[intervals_grid > 0]))
print(f"Actual camera FPS (grid): {FS_CAM:.2f}")

print(f"    {X.shape[0]} grid_frames x {X.shape[1]} cells  |  {t_cam[-1]:.2f}s  at {FS_CAM}Hz")

cam_proxy = build_camera_proxy(X)

# DEBUG: confirm proxy values match tap_diagnostic.py output
mask_dbg = t_cam <= SEARCH_WINDOW_SEC
top5_idx = np.argsort(cam_proxy[mask_dbg])[-5:][::-1]
print("  Top 5 proxy values in search window:")
for i in top5_idx:
    print(f"    frame={grid_frames[i]:4d}  t={t_cam[i]:.3f}s  proxy={cam_proxy[i]:.1f}")
cam_idx, cam_times, cam_proxy_s, cam_dbg = find_tap_peaks(
    cam_proxy, t_cam, FS_CAM,
    SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_CAM, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
)

print(f"    Peaks found: {cam_dbg['num_peaks_found']}")
for i, (gi, tp) in enumerate(zip(cam_idx, cam_times)):
    print(f"      peak {i+1}: t={tp:.3f}s  frame={grid_frames[gi]}  "
          f"proxy={cam_proxy[gi]:.1f}")

# 2. Movesense
print("\n[2] Movesense ACC for synchronization...")

t_ms, ms_proxy, ms_start_ts = build_movesense_signal(PATH_MOVESENSE_ACC)
# Update fs_ms to your known 52Hz
fs_ms = FS_MS_ACC 

print(f"    {len(ms_proxy)} samples  |  {t_ms[-1]:.2f}s  at {fs_ms}Hz")

# Run peak detection on the new ACC proxy
ms_idx, ms_times, ms_proxy_s, ms_dbg = find_tap_peaks(
    ms_proxy, t_ms, fs_ms,
    SEARCH_WINDOW_SEC, TAP_SMOOTH_MS_MS, PROMINENCE_SIGMA, MIN_DISTANCE_SEC
)

print(f"    Peaks found: {ms_dbg['num_peaks_found']}")
for i, (gi, tp) in enumerate(zip(ms_idx, ms_times)):
    print(f"      peak {i+1}: t={tp:.3f}s  idx={gi}  "
          f"proxy={ms_proxy[gi]:.4f}")


# 3. Select & compute offset
print("\n[3] Selecting tap...")
cam_tap_gi, cam_tap = select_tap(cam_idx, cam_times, cam_proxy, label="camera")
ms_tap_gi,  ms_tap  = select_tap(ms_idx,  ms_times, ms_proxy, label="movesense")

cam_tap = subframe_peak_time(cam_proxy_s, cam_tap_gi, t_cam)
ms_tap = subframe_peak_time(ms_proxy_s, ms_tap_gi, t_ms)
dt = ms_tap - cam_tap

print(f"  Camera tap (refined) : {cam_tap:.4f}s  (frame {grid_frames[cam_tap_gi]})")
print(f"  Movesense tap (refined): {ms_tap:.4f}s")
print(f"  dt: {dt:+.4f}s")


# Also compute nearest-frame approach for comparison
cam_nearest_idx = np.argmin(np.abs(t_cam - ms_tap))
cam_nearest_t = float(t_cam[cam_nearest_idx])
dt_nearest = ms_tap - cam_nearest_t


if abs(dt) > 10.0:
    print("WARNING: dt suspiciously large — likely wrong peak pair")
if abs(dt) < 0.05:
    print("WARNING: dt suspiciously small — may be noise")

if abs(dt_nearest) > 10.0:
    print("WARNING: dt_nearest suspiciously large")


print(f"\n{'─'*50}")
print(f"  Camera tap   : {cam_tap:.4f}s  (frame {grid_frames[cam_tap_gi]})")
print(f"  Camera tap (nearest frame): {cam_nearest_t:.4f}s  (frame {grid_frames[cam_nearest_idx]})")

print(f"  Movesense tap: {ms_tap:.4f}s")

print(f"  dt (largest peak): {dt:+.4f}s")
print(f"  dt (nearest frame): {dt_nearest:+.4f}s")

print(f"  Meaning      : t_camera + ({dt:+.4f}s) = t_movesense")
print(f"{'─'*50}")

save_tap_info(PATH_OUTPUT_TAP_JSON, cam_tap, ms_tap, dt, cam_dbg, ms_dbg, ms_start_ts)

# 4. Plots
if not PLOT_SYNC:
    raise SystemExit(0)

t_cam_aligned = t_cam + dt
a = ms_tap - 1.5
b = ms_tap + 3.0

fig, axes = plt.subplots(3, 1, figsize=(14, 15))
fig.subplots_adjust(top=0.93, hspace=0.45)

fig.suptitle(
    f"Chest Tap Sync — {REC_ID}\n",
    fontsize=12
)

# Plot 1: Full search window unaligned
ax = axes[0]
ax.set_title(f"Search window (0 – {SEARCH_WINDOW_SEC}s) — unaligned, own clocks")
sw_c = t_cam <= SEARCH_WINDOW_SEC
sw_m = t_ms  <= SEARCH_WINDOW_SEC
ax.plot(t_cam[sw_c], normalize_zscore(cam_proxy_s[sw_c]),
        color="steelblue",  label="Camera proxy",    linewidth=1.5)
ax.plot(t_ms[sw_m],  normalize_zscore(ms_proxy_s[sw_m]),
        color="darkorange", label="Movesense proxy", linewidth=1.5)
for tp in cam_times:
    ax.axvline(tp, color="steelblue",  linestyle=":", alpha=0.5)
for tp in ms_times:
    ax.axvline(tp, color="darkorange", linestyle=":", alpha=0.5)
ax.axvline(cam_tap, color="steelblue",  linestyle="--", lw=2,
           label=f"Cam tap: {cam_tap:.3f}s (fr{grid_frames[cam_tap_gi]})")
ax.axvline(ms_tap,  color="darkorange", linestyle="--", lw=2,
           label=f"MS tap:  {ms_tap:.3f}s")
ax.set_xlabel("Time (s) — own clock")
ax.set_ylabel("Normalized amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 2: Zoomed unaligned
ax = axes[1]
ax.set_title("Zoomed around tap — UNALIGNED")
mc = (t_cam >= a) & (t_cam <= b)
mm = (t_ms  >= a) & (t_ms  <= b)
ax.plot(t_cam[mc], normalize_zscore(cam_proxy_s[mc]),
        color="steelblue",  label="Camera", linewidth=1.5)
ax.plot(t_ms[mm],  normalize_zscore(ms_proxy_s[mm]),
        color="darkorange", label="Movesense", linewidth=1.5)
ax.axvline(cam_tap, color="steelblue",  linestyle="--", lw=2,
           label=f"Cam: {cam_tap:.3f}s")
ax.axvline(ms_tap,  color="darkorange", linestyle="--", lw=2,
           label=f"MS:  {ms_tap:.3f}s")
ax.set_xlabel("Time (s) — own clock")
ax.set_ylabel("Normalized amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 3: After alignment — peaks must overlap
ax = axes[2]
ax.set_title(
    f"After alignment (camera +{dt:+.4f}s) — "
)
mc2 = (t_cam_aligned >= a) & (t_cam_aligned <= b)
mm2 = (t_ms          >= a) & (t_ms          <= b)
ax.plot(t_cam_aligned[mc2], normalize_zscore(cam_proxy_s[mc2]),
        color="steelblue",  label="Camera (aligned)", linewidth=1.5)
ax.plot(t_ms[mm2],          normalize_zscore(ms_proxy_s[mm2]),
        color="darkorange", label="Movesense", linewidth=1.5)
ax.axvline(ms_tap, color="black", linestyle="--", lw=2,
           label=f"Common tap: {ms_tap:.3f}s")
ax.set_xlabel("Time (s) — Movesense clock")
ax.set_ylabel("Normalized amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

out = PATH_OUTPUT_TAP_JSON.parent / f"{REC_ID}_tap_sync.png"
plt.savefig(out, dpi=150)
plt.show()
