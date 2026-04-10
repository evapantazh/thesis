"""
ecg_diagnostic.py
=================
Examine raw ECG signal quality before attempting R-peak detection.
Identifies: signal inversion, amplitude collapse, motion artifacts.

NEEDS ADJUSTMENT TO NEW CODES!!!!!!!
OLD CODE KEEP FOR LATER TO CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json
from pathlib import Path

# ── same paths as 03_preprocess_ecg.py ───────────────────────
BASE     = Path(r"C:\Projects\thesis\data")
SUBJECT  = "Sub01"
DIST     = "800"
CLOTH    = "Tshirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"

PATH_MOVESENSE = BASE / f"Movesense_{REC_ID}.csv"
PATH_TAP_JSON  = BASE / "tap_info" / f"{REC_ID}_tap_info.json"

def robust_fs(t):
    t  = np.asarray(t, float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / np.median(dt)

# ── load ──────────────────────────────────────────────────────
with open(PATH_TAP_JSON) as f:
    tap_info = json.load(f)
ecg_tap_sec = tap_info["ecg_tap_sec"]

df_ms         = pd.read_csv(PATH_MOVESENSE, comment="#")
df_ms.columns = df_ms.columns.str.strip()
t_ms          = df_ms["Elapsed time"].values.astype(float)
ecg_raw       = df_ms["ECG"].values.astype(float)
fs            = robust_fs(t_ms)

# analysis window
win_start = ecg_tap_sec + 5.0
win_end   = win_start + 53.8
mask      = (t_ms >= win_start) & (t_ms <= win_end)
t_win     = t_ms[mask]
ecg_win   = ecg_raw[mask]
t_local   = t_win - t_win[0]

# bandpass
nyq = fs / 2
b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
ecg_dc = ecg_win - np.mean(ecg_win)
ecg_bp = filtfilt(b, a, ecg_dc)

# ── signal quality metrics ────────────────────────────────────
print("=" * 60)
print("ECG SIGNAL QUALITY DIAGNOSTIC")
print("=" * 60)

# Check if signal is inverted
# R-peaks should be positive. If more large-amplitude events are
# negative than positive, the signal is likely inverted.
pos_max = np.max(ecg_bp)
neg_min = np.min(ecg_bp)
print(f"\nAmplitude range : {neg_min:.2f} to {pos_max:.2f}")
print(f"Max positive    : {pos_max:.2f}")
print(f"Max negative    : {neg_min:.2f} (abs={abs(neg_min):.2f})")
if abs(neg_min) > pos_max * 1.5:
    print("  ⚠ Signal appears INVERTED — R-peaks point downward")
    print("    Fix: multiply signal by -1 before detection")
else:
    print("  ✓ Signal polarity appears normal")

# Check amplitude by 10s segments
print(f"\nAmplitude by 10s segment:")
print(f"  {'Segment':<15}  {'Max':>8}  {'Min':>8}  {'Std':>8}  {'Status'}")
print(f"  {'-'*55}")
for seg_start in range(0, int(t_local[-1]), 10):
    seg_end  = seg_start + 10
    seg_mask = (t_local >= seg_start) & (t_local < seg_end)
    if not np.any(seg_mask):
        continue
    seg      = ecg_bp[seg_mask]
    seg_std  = np.std(seg)
    overall_std = np.std(ecg_bp)
    status = ""
    if seg_std < overall_std * 0.3:
        status = "⚠ LOW AMPLITUDE (possible electrode loss)"
    elif seg_std > overall_std * 2.5:
        status = "⚠ HIGH AMPLITUDE (motion artifact)"
    else:
        status = "✓ Normal"
    print(f"  {seg_start:>4}s – {seg_end:<4}s   "
          f"{np.max(seg):>8.2f}  {np.min(seg):>8.2f}  "
          f"{seg_std:>8.2f}  {status}")

# ── plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 11))
fig.suptitle(f"ECG Signal Quality Diagnostic — {REC_ID}", fontsize=13, fontweight='bold')
fig.subplots_adjust(top=0.92, hspace=0.45)

# Plot 1: full bandpass signal with zero line
ax = axes[0]
ax.plot(t_local, ecg_bp, color='navy', linewidth=0.7, alpha=0.9)
ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title("Bandpass ECG — Full Window (look for inversion and amplitude collapse)")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (s)")
ax.grid(alpha=0.3)
# Shade low-amplitude regions
overall_std = np.std(ecg_bp)
for seg_start in range(0, int(t_local[-1]), 10):
    seg_end  = seg_start + 10
    seg_mask = (t_local >= seg_start) & (t_local < seg_end)
    if not np.any(seg_mask):
        continue
    if np.std(ecg_bp[seg_mask]) < overall_std * 0.3:
        ax.axvspan(seg_start, seg_end, alpha=0.2, color='red',
                   label='Low amplitude' if seg_start == 0 else '_')

# Plot 2: Zoom into first 10s — should see clear QRS complexes
ax = axes[1]
zoom_mask = t_local <= 10.0
ax.plot(t_local[zoom_mask], ecg_bp[zoom_mask], color='darkgreen', linewidth=1.2)
ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title("First 10s zoomed — R-peaks should be clear upward spikes")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (s)")
ax.grid(alpha=0.3)

# Plot 3: Inverted signal — try detecting on flipped version
ax = axes[2]
ecg_inv = -ecg_bp
ax.plot(t_local[zoom_mask], ecg_inv[zoom_mask],
        color='darkorange', linewidth=1.2, label='Inverted ECG (-1 × signal)')
ax.plot(t_local[zoom_mask], ecg_bp[zoom_mask],
        color='darkgreen', linewidth=1.2, alpha=0.5, label='Original ECG')
ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title("First 10s: Original vs Inverted — which one has upward R-peaks?")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (s)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

out = BASE / f"ecg_processed/{REC_ID}_ecg_diagnostic.png"
Path(out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved → {out}")
print("\nLook at Plot 3: whichever version has clear UPWARD sharp spikes = correct polarity.")
print("If inverted is better → set INVERT_ECG = True in 03_preprocess_ecg.py")