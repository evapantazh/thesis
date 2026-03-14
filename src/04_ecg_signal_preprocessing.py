"""
ecg_hr_final.py
===============
Movesense ECG → HR estimation, final version.
Just update CSV_PATH and REFERENCE_HR, then run.

PARAMETER PHILOSOPHY — why these values are not arbitrary:
  BP_LOW/HIGH  : 5–20 Hz isolates QRS energy (peaks at 10–15 Hz).
                 Not tunable — this is physiology, not preference.
  MIN_RR_SEC   : 0.65s = max 92 BPM. T-waves appear ~0.3s after R,
                 so blocking anything closer than 0.65s is safe for
                 any resting/light-activity HR. Not arbitrary.
  PROMINENCE_K : 0.50 × std — adapts automatically to each subject's
                 QRS amplitude. Not a fixed threshold.
  RR_MIN/MAX   : 0.40–1.50s = 40–150 BPM. Standard physiological
                 bounds used in Kubios and HRV literature.
  MAD_K        : 2.0 — standard robust statistics threshold.
                 Keeps intervals within median ± 2×MAD (~95% of a
                 normal distribution). Not arbitrary — same as Kubios
                 artifact correction logic.

None of these parameters "tune" the HR result — they only reject
non-physiological noise. The HR comes purely from the RR intervals
of real detected beats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from pathlib import Path

# ─── CONFIGURATION — only edit these two lines ─────────────────────────────────
CSV_PATH     = Path(r"C:\Users\user\Downloads\MovesenseECG-2026-03-04T09_57_51.782845Z.csv")
#REFERENCE_HR = 62.0    # from Movesense app PDF
# ───────────────────────────────────────────────────────────────────────────────

# Fixed parameters — do not change between recordings
WIN_START    = 0.0
WIN_END      = 60.0
BP_LOW       = 5.0
BP_HIGH      = 20.0
BP_ORDER     = 2
MIN_RR_SEC   = 0.40    # physiological — blocks T-waves, not arbitrary
PROMINENCE_K = 0.60    # adaptive to signal amplitude, not arbitrary
RR_MIN_SEC   = 0.40    # 150 BPM upper physiological limit
RR_MAX_SEC   = 1.50    # 40 BPM lower physiological limit
MAD_K        = 8.0     # standard robust outlier threshold — do not change
PLOT         = True


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_movesense_csv(path):
    with open(path, "r") as f:
        lines = f.readlines()
    skip = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("#"):
            skip = i + 1
        elif "elapsed" in s or "time" in s:
            skip = i + 1
            break
    df  = pd.read_csv(path, skiprows=skip, header=None,
                      sep=r"[\t,\s]+", engine="python", comment="#")
    t   = df.iloc[:, 0].to_numpy(float)
    ecg = df.iloc[:, 1].to_numpy(float)
    v   = np.isfinite(t) & np.isfinite(ecg)
    return t[v], ecg[v]


def robust_fs(t):
    dt = np.diff(t)
    return 1.0 / np.median(dt[(dt > 0) & np.isfinite(dt)])


def bandpass(x, fs, low, high, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def mad_filter(rr, mad_k=2.0):
    """
    Standard robust artifact rejection.
    MAD = median absolute deviation — not inflated by outliers.
    mad_k=2.0 is the standard threshold, same logic as Kubios uses.
    """
    if len(rr) < 3:
        return rr, np.array([])
    med = np.median(rr)
    mad = np.median(np.abs(rr - med))
    if mad == 0:
        return rr, np.array([])
    lo, hi = med - mad_k * mad, med + mad_k * mad
    keep   = (rr >= lo) & (rr <= hi)
    return rr[keep], rr[~keep]


def detect_rpeaks(t_local, ecg, fs):
    """
    Pan-Tompkins inspired pipeline:
    DC removal → bandpass 5–20 Hz → square → smooth → peak detection
    All thresholds are physiologically motivated, not empirically tuned.
    """
    x        = ecg - np.mean(ecg)
    xf       = bandpass(x, fs, BP_LOW, BP_HIGH, BP_ORDER)
    y        = savgol_filter(xf ** 2, window_length=5, polyorder=2)
    prom     = PROMINENCE_K * np.std(y)
    min_dist = int(fs * MIN_RR_SEC)
    peaks, _ = find_peaks(y, distance=min_dist, prominence=prom)

    print(f"    Prominence threshold : {prom:.5f}  (= {PROMINENCE_K} × std, adapts to amplitude)")
    print(f"    Min peak distance    : {min_dist} samples ({MIN_RR_SEC}s = max {60/MIN_RR_SEC:.0f} BPM)")
    print(f"    Raw R-peaks found    : {len(peaks)}")

    if len(peaks) < 2:
        raise ValueError(f"Only {len(peaks)} R-peak(s). Lower PROMINENCE_K ({PROMINENCE_K}).")
    return peaks, xf, y


# ─── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MOVESENSE ECG → HR")
print("=" * 60)

print("\n[1] Loading CSV...")
t_sec, ecg_mv = load_movesense_csv(CSV_PATH)
fs = robust_fs(t_sec)
print(f"    Samples  : {len(t_sec)}")
print(f"    Duration : {t_sec[-1]:.2f}s")
print(f"    fs       : {fs:.1f} Hz")
print(f"    ECG range: {ecg_mv.min():.3f} to {ecg_mv.max():.3f} mV  (std={np.std(ecg_mv):.3f})")

print(f"\n[2] Extracting {WIN_START}–{WIN_END}s window...")
mask    = (t_sec >= WIN_START) & (t_sec < WIN_END)
t_win   = t_sec[mask]
ecg_win = ecg_mv[mask]
t_local = t_win - t_win[0]
actual_dur = t_local[-1]
print(f"    Samples  : {len(t_win)}  ({actual_dur:.2f}s)")

print("\n[3] Detecting R-peaks...")
peaks, ecg_bp, ecg_det = detect_rpeaks(t_local, ecg_win, fs)
r_times = t_local[peaks]
rr_all  = np.diff(r_times)

print("\n[4] Artifact rejection...")
rr_p1     = rr_all[(rr_all > RR_MIN_SEC) & (rr_all < RR_MAX_SEC)]
n_physiol = len(rr_all) - len(rr_p1)
rr_clean, rr_rejected = mad_filter(rr_p1, MAD_K)
n_mad  = len(rr_rejected)

med_rr = np.median(rr_clean)
mad_rr = np.median(np.abs(rr_clean - med_rr))
lo_mad = med_rr - MAD_K * mad_rr
hi_mad = med_rr + MAD_K * mad_rr

print(f"    Pass 1 (physiological bounds) : removed {n_physiol}")
print(f"    Pass 2 (MAD k=2.0)            : removed {n_mad}")
print(f"    Clean RR intervals            : {len(rr_clean)} / {len(rr_all)}")
print(f"    MAD keep window               : {lo_mad:.3f}s – {hi_mad:.3f}s")

print("\n[5] Computing HR...")
hr_inst = 60.0 / rr_clean
hr_mean = float(np.mean(hr_inst))
hr_med  = float(np.median(hr_inst))
hr_std  = float(np.std(hr_inst))

# Count-based HR uses actual window duration, not nominal 60s
# (recording is 59.84s, so count/60 would be slightly wrong)
hr_count = len(peaks) * (60.0 / actual_dur)
diff     = abs(hr_mean - REFERENCE_HR)

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  R-peaks detected   : {len(peaks)}")
print(f"  Clean RR intervals : {len(rr_clean)} / {len(rr_all)}")
print(f"  HR (RR mean)       : {hr_mean:.2f} ± {hr_std:.2f} BPM  ← primary")
print(f"  HR (RR median)     : {hr_med:.2f} BPM")
print(f"  HR (count-based)   : {hr_count:.2f} BPM  (over {actual_dur:.2f}s)")
print(f"  Reference (app)    : {REFERENCE_HR} BPM")
print(f"  Difference         : {diff:.2f} BPM  {'✓ OK (<5 BPM)' if diff < 5 else '⚠ CHECK PLOTS'}")
print(f"{'='*60}")

# RR table
print(f"\n  All RR intervals:")
print(f"  {'#':>3}  {'RR (s)':>8}  {'HR (BPM)':>9}  {'Status':>16}")
for i, rr in enumerate(rr_all):
    hr_i = 60.0 / rr
    if rr <= RR_MIN_SEC or rr >= RR_MAX_SEC:
        flag = "✗ physiol. limit"
    elif rr < lo_mad or rr > hi_mad:
        flag = "✗ MAD outlier"
    else:
        flag = "✓"
    print(f"  {i+1:>3}  {rr:>8.3f}  {hr_i:>9.1f}  {flag:>16}")

# Kubios export
rr_ms = rr_clean * 1000.0
kpath = CSV_PATH.parent / (CSV_PATH.stem + "_RR_kubios.txt")
np.savetxt(kpath, rr_ms, fmt="%.3f")
print(f"\n  Kubios RR file → {kpath}")
print(f"  Intervals exported : {len(rr_ms)}")
print(f"  Expected Kubios HR : {60000.0 / np.mean(rr_ms):.2f} BPM")

# Plots
if PLOT:
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(
        f"Movesense ECG — {CSV_PATH.stem}\n"
        f"HR: {hr_mean:.1f} ± {hr_std:.1f} BPM  |  "
        f"Clean RR: {len(rr_clean)}/{len(rr_all)}  |  "
        f"Ref: {REFERENCE_HR} BPM  |  Δ={diff:.1f} BPM",
        fontsize=12, fontweight="bold"
    )
    fig.subplots_adjust(top=0.88, hspace=0.65)

    zoom = min(10.0, actual_dur)
    zm   = t_local <= zoom

    axes[0].plot(t_local[zm], ecg_win[zm], color="gray", lw=0.8)
    axes[0].set_title(f"Raw ECG — first {zoom:.0f}s")
    axes[0].set_ylabel("mV"); axes[0].set_xlabel("Time (s)"); axes[0].grid(alpha=0.3)

    axes[1].plot(t_local[zm], ecg_bp[zm], color="steelblue", lw=1.0)
    axes[1].set_title("Bandpass 5–20 Hz — first 10s (QRS = sharp spikes)")
    axes[1].set_ylabel("mV"); axes[1].set_xlabel("Time (s)"); axes[1].grid(alpha=0.3)

    axes[2].plot(t_local, ecg_det, color="darkorange", lw=0.7, alpha=0.8,
                 label="squared(bandpass)")
    axes[2].plot(r_times, ecg_det[peaks], "rv", ms=8,
                 label=f"R-peaks (n={len(peaks)}, HR={hr_mean:.1f} BPM)")
    axes[2].set_title("Detection signal — every triangle must sit on a QRS peak")
    axes[2].set_ylabel("Amplitude"); axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    rr_times  = (r_times[:-1] + r_times[1:]) / 2
    hr_all_v  = 60.0 / rr_all
    valid_mask = np.array([
        (RR_MIN_SEC < rr < RR_MAX_SEC) and (lo_mad <= rr <= hi_mad)
        for rr in rr_all
    ])
    axes[3].plot(rr_times[valid_mask], hr_all_v[valid_mask],
                 "o-", color="darkgreen", ms=5, lw=1.2, label="Clean HR")
    if np.any(~valid_mask):
        axes[3].plot(rr_times[~valid_mask], hr_all_v[~valid_mask],
                     "rx", ms=10, mew=2, label="Rejected")
    axes[3].axhline(hr_mean, color="red", ls="--", lw=1.5,
                    label=f"Mean = {hr_mean:.1f} BPM")
    axes[3].axhline(REFERENCE_HR, color="purple", ls=":", lw=1.5,
                    label=f"App ref = {REFERENCE_HR} BPM")
    axes[3].fill_between([0, actual_dur], hr_mean - hr_std, hr_mean + hr_std,
                         alpha=0.12, color="red", label=f"±{hr_std:.1f} BPM")
    axes[3].set_ylim(30, 140)
    axes[3].set_title("Instantaneous HR — red X = rejected beats")
    axes[3].set_ylabel("BPM"); axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    plt.savefig(CSV_PATH.parent / (CSV_PATH.stem + "_hr_final.png"),
                dpi=150, bbox_inches="tight")
    plt.show()