"""
02_find_chest_tap.py
====================
Detect chest tap in both camera grid signal and Movesense ACC,
compute the time offset dt, and save to JSON.

Protocol: ONE firm tap on sternum at the start of recording.

Camera tap detection:
  Uses energy × coherence proxy. Picks the LARGEST peak in the
  search window — the actual impact + rebound always dominates
  over the hand-entry artifact.

Movesense tap detection:
  Uses |gradient(acceleration magnitude)| on the 52 Hz ACC stream.
  Picks the LARGEST peak — the mechanical impact is always the
  strongest event in the search window.
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
SUBJECT  = "andreas"
DIST     = "800"
CLOTH    = "T-shirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"
movesense_path = Path(r"D:\movesense")

PATH_CAMERA_GRID = BASE / "GRID_files"/ f"GRID_andreas_800_tshirt.CSV"
PATH_MOVESENSE   = movesense_path / f"Movesense_{REC_ID}"
PATH_MOVESENSE_ACC = PATH_MOVESENSE / f"{REC_ID}_acc_stream.json"
PATH_MOVESENSE_ECG = PATH_MOVESENSE / f"{REC_ID}_ecg_stream.json"
PATH_OUTPUT_TAP_JSON = BASE / "tap_info" / f"Tap_info_{REC_ID}.json"


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEARCH_WINDOW_SEC  = 8.0

# No smoothing — avoids shifting the detected peak away from
# the true maximum. Both proxies are already clean enough.
TAP_SMOOTH_MS_CAM  = 0.0
TAP_SMOOTH_MS_MS   = 0.0

# How many std above local noise a peak must be to count as a tap.
PROMINENCE_SIGMA   = 2.0

# Minimum gap between two peaks (avoids counting same tap twice).
MIN_DISTANCE_SEC   = 0.25

FS_MS_ACC = 52.0  # Movesense accelerometer sample rate


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

    proxy = energy × coherence
      energy    = sum(abs(gradient))  -- total motion magnitude
      coherence = abs(mean(gradient)) -- directional agreement across cells

    Product is high ONLY when all cells move together = real impact.
    """
    X = np.asarray(X, float).copy()

    # Replace invalid pixels with column median
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = (X[:, j] == 0) | ~np.isfinite(X[:, j])
        X[bad, j] = col_med[j]

    dX = np.diff(X, axis=0, prepend=X[[0]])   # T x N_cells

    energy    = np.sum(np.abs(dX), axis=1)     # T
    coherence = np.abs(np.mean(dX, axis=1))    # T

    return energy * coherence                   # T


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
            mag = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            all_magnitudes.append(mag)
            all_timestamps.append(ts + (i * (1000.0 / 52.0))) 

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


def select_tap_camera(global_indices, peak_times, proxy, t, label="camera"):
    """
    Find the contact moment for the CAMERA signal.

    Physics of a chest tap in depth data:
      1. Hand enters depth field  → BIG spike (hand appears)
      2. Hand decelerates/hovers  → quiet (near-zero proxy)
      3. Palm hits sternum        → SMALL bump (brief compression)
      4. Hand withdraws + rebound → BIG spike (hand leaves, chest bounces)

    The actual contact is the small local peak (#3) between the two
    dominant peaks (#1 and #4). We find it by running peak detection
    on the region between the two biggest peaks.

    If no bump is found between the two peaks, fall back to the
    valley (minimum) which is the closest approximation.

    Requires at least 2 peaks. If only 1 peak found, falls back to it.
    """
    if len(global_indices) == 0:
        raise ValueError(
            f"No tap peak found in [{label}]!\n"
            f"  Try: lower PROMINENCE_SIGMA (now={PROMINENCE_SIGMA}, try 1.5)\n"
            f"  Try: widen SEARCH_WINDOW_SEC (now={SEARCH_WINDOW_SEC})\n"
            f"  Check: was the tap within the first {SEARCH_WINDOW_SEC}s?"
        )

    if len(global_indices) < 2:
        print(f"  [WARN] Only 1 peak in [{label}] — using it directly")
        return global_indices[0], float(peak_times[0])

    # Find the two largest peaks
    sorted_by_value = np.argsort(proxy[global_indices])[::-1]
    top2 = sorted(sorted_by_value[:2])  # sort by time order
    gi_a = global_indices[top2[0]]  # earlier peak (hand entry)
    gi_b = global_indices[top2[1]]  # later peak  (hand withdrawal)

    print(f"  [{label}] Two dominant peaks:")
    print(f"    Peak A (hand entry):      frame {gi_a}  t={t[gi_a]:.3f}s  proxy={proxy[gi_a]:.1f}")
    print(f"    Peak B (hand withdrawal): frame {gi_b}  t={t[gi_b]:.3f}s  proxy={proxy[gi_b]:.1f}")

    # Search for a small local peak (the contact bump) between A and B
    # Exclude the endpoints (they are the big peaks themselves)
    start = gi_a + 1
    end   = gi_b      # exclusive
    if end - start < 2:
        # Too few frames between peaks — fall back to midpoint
        contact_gi = (gi_a + gi_b) // 2
        contact_t  = float(t[contact_gi])
        print(f"    [WARN] Only {end-start} frames between peaks — using midpoint")
        print(f"    Contact (midpoint):       frame {contact_gi}  t={contact_t:.3f}s  proxy={proxy[contact_gi]:.1f}")
        return contact_gi, contact_t

    region = proxy[start:end]

    # Find local peaks in the region between the two dominant peaks
    # Use a low prominence threshold — the contact bump is small
    region_std = np.std(region)
    local_peaks, _ = signal.find_peaks(region, prominence=region_std * 0.5)

    if len(local_peaks) > 0:
        # Pick the largest local peak between the two dominant peaks
        best_local = local_peaks[np.argmax(region[local_peaks])]
        contact_gi = start + best_local
        contact_t  = float(t[contact_gi])
        print(f"    Contact bump found:       frame {contact_gi}  t={contact_t:.3f}s  proxy={proxy[contact_gi]:.1f}")
        # Show all candidates
        for lp in local_peaks:
            gi_lp = start + lp
            marker = " <<<" if lp == best_local else ""
            print(f"      candidate: frame {gi_lp}  t={t[gi_lp]:.3f}s  proxy={proxy[gi_lp]:.1f}{marker}")
    else:
        # No bump found — fall back to the frame just after the valley
        # (the valley is the hand hovering; one frame later is likely contact)
        valley_local = np.argmin(region)
        # Use the frame after the valley if possible
        if valley_local + 1 < len(region):
            contact_gi = start + valley_local + 1
        else:
            contact_gi = start + valley_local
        contact_t = float(t[contact_gi])
        print(f"    [WARN] No bump between peaks — using frame after valley")
        print(f"    Contact (post-valley):    frame {contact_gi}  t={contact_t:.3f}s  proxy={proxy[contact_gi]:.1f}")

    return contact_gi, contact_t


def select_tap_movesense(global_indices, peak_times, proxy, label="movesense"):
    """
    Find the tap for the MOVESENSE signal.
    Simply picks the LARGEST peak — the mechanical impact is always
    the strongest event in the accelerometer search window.
    """
    if len(global_indices) == 0:
        raise ValueError(
            f"No tap peak found in [{label}]!\n"
            f"  Try: lower PROMINENCE_SIGMA (now={PROMINENCE_SIGMA}, try 1.5)\n"
            f"  Try: widen SEARCH_WINDOW_SEC (now={SEARCH_WINDOW_SEC})\n"
            f"  Check: was the tap within the first {SEARCH_WINDOW_SEC}s?"
        )

    best = np.argmax(proxy[global_indices])

    if len(global_indices) > 1:
        print(f"  [INFO] {len(global_indices)} peaks in [{label}] — picking LARGEST")
        for i, (gi, tp) in enumerate(zip(global_indices, peak_times)):
            marker = " <<<" if i == best else ""
            print(f"         peak {i+1}: t={tp:.3f}s  proxy={proxy[gi]:.4f}{marker}")

    return global_indices[best], float(peak_times[best])


def subframe_peak_time(proxy, peak_idx, t):
    """
    Parabolic interpolation for sub-sample tap timing.
    
    Uses the RAW proxy to find the true peak location between
    discrete samples. No half-frame offset — the peak of the
    proxy IS the event we want to time.
    """
    i = int(peak_idx)
    if i <= 0 or i >= len(proxy) - 1:
        return float(t[i])
    
    y0, y1, y2 = float(proxy[i-1]), float(proxy[i]), float(proxy[i+1])
    denom = y0 - 2.0*y1 + y2
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = np.clip((y0 - y2) / (2.0 * denom), -0.5, 0.5)
    
    # Interpolate time
    if delta >= 0:
        dt_step = float(t[i+1] - t[i])
    else:
        dt_step = float(t[i] - t[i-1])
    
    return float(t[i]) + delta * dt_step


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
        "protocol"     : "camera_valley_between_peaks__movesense_largest_peak",
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
ts_df_grid = pd.read_csv(BASE / "andreas_800_tshirt" / "timestamps.csv")
ts_df_grid = ts_df_grid.set_index("frame")
grid_timestamps = ts_df_grid.loc[grid_frames, "timestamp"].values
t_cam = (grid_timestamps - grid_timestamps[0]) / 1000.0

# Recompute FS_CAM from grid timestamps
intervals_grid = np.diff(t_cam)
FS_CAM = float(1.0 / np.median(intervals_grid[intervals_grid > 0]))
print(f"  Actual camera FPS (grid): {FS_CAM:.2f}")
print(f"    {X.shape[0]} grid_frames x {X.shape[1]} cells  |  {t_cam[-1]:.2f}s  at {FS_CAM:.2f}Hz")

cam_proxy = build_camera_proxy(X)

# DEBUG: top proxy values in search window
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
fs_ms = FS_MS_ACC

print(f"    {len(ms_proxy)} samples  |  {t_ms[-1]:.2f}s  at {fs_ms}Hz")

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
cam_tap_gi, cam_tap = select_tap_camera(cam_idx, cam_times, cam_proxy, t_cam, label="camera")
ms_tap_gi,  ms_tap  = select_tap_movesense(ms_idx, ms_times, ms_proxy, label="movesense")

# Camera contact is a VALLEY (minimum between peaks) — no parabolic peak refinement.
# The discrete frame timestamp is the best estimate.
cam_tap_refined = cam_tap
# Movesense: refine with parabolic interpolation on the RAW proxy
ms_tap_refined  = subframe_peak_time(ms_proxy, ms_tap_gi, t_ms)
dt = ms_tap_refined - cam_tap_refined

print(f"  Camera tap  : frame {grid_frames[cam_tap_gi]}  "
      f"t={t_cam[cam_tap_gi]:.4f}s  (refined: {cam_tap_refined:.4f}s)")
print(f"  Movesense tap: idx {ms_tap_gi}  "
      f"t={t_ms[ms_tap_gi]:.4f}s  (refined: {ms_tap_refined:.4f}s)")
print(f"  dt: {dt:+.4f}s")


if abs(dt) > 10.0:
    print("  WARNING: dt suspiciously large — likely wrong peak pair")
if abs(dt) < 0.01:
    print("  NOTE: dt very small — clocks nearly aligned")


print(f"\n{'─'*50}")
print(f"  Camera tap    : {cam_tap_refined:.4f}s  (frame {grid_frames[cam_tap_gi]})")
print(f"  Movesense tap : {ms_tap_refined:.4f}s")
print(f"  dt            : {dt:+.4f}s")
print(f"  Meaning       : t_camera + ({dt:+.4f}s) = t_movesense")
print(f"{'─'*50}")

save_tap_info(PATH_OUTPUT_TAP_JSON, cam_tap_refined, ms_tap_refined, dt,
              cam_dbg, ms_dbg, ms_start_ts)

# 4. Plots
if not PLOT_SYNC:
    raise SystemExit(0)

t_cam_aligned = t_cam + dt
a = ms_tap_refined - 1.5
b = ms_tap_refined + 3.0

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
ax.axvline(cam_tap_refined, color="steelblue",  linestyle="--", lw=2,
           label=f"Cam tap: {cam_tap_refined:.3f}s (fr{grid_frames[cam_tap_gi]})")
ax.axvline(ms_tap_refined,  color="darkorange", linestyle="--", lw=2,
           label=f"MS tap:  {ms_tap_refined:.3f}s")
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
ax.axvline(cam_tap_refined, color="steelblue",  linestyle="--", lw=2,
           label=f"Cam: {cam_tap_refined:.3f}s")
ax.axvline(ms_tap_refined,  color="darkorange", linestyle="--", lw=2,
           label=f"MS:  {ms_tap_refined:.3f}s")
ax.set_xlabel("Time (s) — own clock")
ax.set_ylabel("Normalized amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 3: After alignment — peaks must overlap
ax = axes[2]
ax.set_title(
    f"After alignment (camera {dt:+.4f}s)"
)
mc2 = (t_cam_aligned >= a) & (t_cam_aligned <= b)
mm2 = (t_ms          >= a) & (t_ms          <= b)
ax.plot(t_cam_aligned[mc2], normalize_zscore(cam_proxy_s[mc2]),
        color="steelblue",  label="Camera (aligned)", linewidth=1.5)
ax.plot(t_ms[mm2],          normalize_zscore(ms_proxy_s[mm2]),
        color="darkorange", label="Movesense", linewidth=1.5)
ax.axvline(ms_tap_refined, color="black", linestyle="--", lw=2,
           label=f"Common tap: {ms_tap_refined:.3f}s")
ax.set_xlabel("Time (s) — Movesense clock")
ax.set_ylabel("Normalized amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

out = PATH_OUTPUT_TAP_JSON.parent / f"{REC_ID}_tap_sync.png"
plt.savefig(out, dpi=150)
plt.show()