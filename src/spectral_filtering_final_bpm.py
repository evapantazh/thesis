"""
CAMERA-ONLY PIPELINE (from your Low-Rank X) + Adaptive Spectral Filtering (Eq. 6–7)

What this script does:
1) Loads the CLEAN low-rank matrix X (RPCA output already saved).
2) Eigen-decomposition of B = X^T X (paper Eq. 4).
3) Builds top-5 candidate pulse signals p_i = X v_i (paper Eq. 5).
4) Estimates HR for each candidate via FFT peak within 45–150 BPM.
5) Scores candidates using harmonic presence (2nd, 3rd).
6) Selects the best candidate within physiological range (50–150 BPM).
7) Applies Adaptive Spectral Filtering around fHR, 2fHR, 3fHR with omega=0.25 Hz (paper Eq. 6–7).
8) Re-estimates HR after filtering and plots before/after.

Notes:
- This is CAMERA-only. (No Movesense sync in this script.)
- If you want windowing after tap, apply a mask to frames BEFORE FFT steps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft
from scipy import signal

# ============================================================
# PARAMETERS
# ============================================================
path_clean = r"C:\Projects\thesis\data\Sub01_Matrix_X_LowRank.csv"

fs = 30.0                     # camera sampling rate
HR_MIN, HR_MAX = 45, 150      # bpm range for HR peak search
FREQ_MIN, FREQ_MAX = HR_MIN/60, HR_MAX/60

OMEGA = 0.25                  # Hz, paper suggests 0.25 Hz (~15 BPM window width)
PHYS_MIN, PHYS_MAX = 50, 150  # physiological selection range

# ============================================================
# HELPERS
# ============================================================

def estimate_fhr_hz(pc, fs, fmin=0.75, fmax=2.5):
    """
    Estimate fundamental HR frequency (Hz) by FFT peak search in [fmin,fmax].
    Uses rFFT for real signals. Includes detrend + Hann window.
    """
    pc = np.asarray(pc, float)
    pc = signal.detrend(pc)
    pc = pc * np.hanning(len(pc))

    F = np.abs(rfft(pc))
    f = rfftfreq(len(pc), d=1/fs)

    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        raise ValueError("No FFT bins in HR band. Check fs or segment length.")
    f_hr = f[band][np.argmax(F[band])]
    return float(f_hr), f, F

def adaptive_spectral_filter(pc, fs, f_hr, omega=0.25):
    """
    Paper Eq.(6)-(7):
    Υ(f)=1 in [k fHR - ω/2, k fHR + ω/2] for k=1,2,3 else 0
    pd(t)=iFFT( pc(f) * Υ(f) )
    """
    pc = np.asarray(pc, float)
    pc = signal.detrend(pc)

    Pc = rfft(pc)
    f = rfftfreq(len(pc), d=1/fs)

    mask = np.zeros_like(f, dtype=float)
    half = omega / 2.0
    for k in (1, 2, 3):
        fk = k * f_hr
        mask[(f >= fk - half) & (f <= fk + half)] = 1.0

    Pd = Pc * mask
    pd = irfft(Pd, n=len(pc))
    return pd, f, mask

# ============================================================
# 1) LOAD THE CLEAN Low-Rank matrix
# ============================================================
df = pd.read_csv(path_clean)
frames = df['frame'].values.astype(int)
X = df.drop(columns=['frame']).values

print(f"Loaded matrix X: {X.shape}")
print(f"  Frames: {X.shape[0]}")
print(f"  Cells:  {X.shape[1]}")

# ============================================================
# 2) Eigen-decomposition (paper Eq. 4)
# ============================================================
print("\n" + "="*60)
print("STEP 2: Eigenvalue Decomposition (B = X^T × X)")
print("="*60)

B = X.T @ X
vals, vecs = np.linalg.eigh(B)

idx = vals.argsort()[::-1]
vals_sorted = vals[idx]
vecs_sorted = vecs[:, idx]

total_variance = vals_sorted.sum()
print("\nVariance explained by top 5 eigenvectors:")
for i in range(5):
    variance_pct = (vals_sorted[i] / total_variance) * 100
    print(f"  v{i+1} (λ{i+1}): {variance_pct:.2f}%")

# ============================================================
# 3) Test top 5 eigenvectors (paper Eq. 5) + harmonics scoring
# ============================================================
print("\n" + "="*60)
print("STEP 3: Testing Top 5 Eigenvectors for Cardiac Signal")
print("="*60)

candidates = []

for i in range(5):
    # pulse signal candidate (Eq. 5)
    p = X @ vecs_sorted[:, i]

    # FFT analysis (your original method)
    T = len(p)
    xf = fftfreq(T, 1/fs)
    yf = np.abs(fft(p))

    # HR range only (positive freqs in band)
    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    xf_hr = xf[mask]
    yf_hr = yf[mask]

    peaks, properties = find_peaks(yf_hr, prominence=yf_hr.max() * 0.1)
    if len(peaks) == 0:
        fundamental_idx = np.argmax(yf_hr)
    else:
        fundamental_idx = peaks[np.argmax(yf_hr[peaks])]

    fundamental_freq = xf_hr[fundamental_idx]
    fundamental_hr = fundamental_freq * 60
    fundamental_power = yf_hr[fundamental_idx]

    # Harmonics scoring (same as you had)
    harmonic_2_freq = fundamental_freq * 2
    harmonic_3_freq = fundamental_freq * 3

    harmonic_score = fundamental_power

    harmonic_2_mask = (xf >= harmonic_2_freq - 0.1) & (xf <= harmonic_2_freq + 0.1)
    if harmonic_2_mask.any():
        harmonic_2_power = np.max(yf[harmonic_2_mask])
        harmonic_score += harmonic_2_power * 0.5

    harmonic_3_mask = (xf >= harmonic_3_freq - 0.1) & (xf <= harmonic_3_freq + 0.1)
    if harmonic_3_mask.any():
        harmonic_3_power = np.max(yf[harmonic_3_mask])
        harmonic_score += harmonic_3_power * 0.3

    candidates.append({
        'index': i + 1,
        'signal': p,
        'hr': fundamental_hr,
        'power': fundamental_power,
        'harmonic_score': harmonic_score,
        'xf_hr': xf_hr,
        'yf_hr': yf_hr
    })

    # classify
    if fundamental_hr < 40:
        signal_type = "TOO SLOW (noise)"
    elif fundamental_hr < 50:
        signal_type = "RESPIRATORY (breathing)"
    elif fundamental_hr > 100:
        signal_type = "POSSIBLE CARDIAC (high HR)"
    else:
        signal_type = "LIKELY CARDIAC"

    print(f"\nv{i+1} (p{i+1}):")
    print(f"  Fundamental HR: {fundamental_hr:.2f} BPM → {signal_type}")
    print(f"  Power: {fundamental_power:.0f}")
    print(f"  Harmonic Score: {harmonic_score:.0f}")

# ============================================================
# 4) Select best eigenvector (physiological filtering)
# ============================================================
print("\n" + "="*60)
print("STEP 4: Selecting Best Eigenvector (with physiological filtering)")
print("="*60)

valid_candidates = [c for c in candidates if PHYS_MIN < c['hr'] < PHYS_MAX]

print(f"\nValid cardiac candidates ({PHYS_MIN}-{PHYS_MAX} BPM): {len(valid_candidates)}")
for c in valid_candidates:
    print(f"  v{c['index']}: {c['hr']:.2f} BPM (score: {c['harmonic_score']:.0f})")

if len(valid_candidates) == 0:
    print("\n⚠️ WARNING: No valid cardiac candidates found!")
    print("   Falling back to p2 (second eigenvector)")
    best_candidate = candidates[1]
else:
    best_candidate = max(valid_candidates, key=lambda x: x['harmonic_score'])

best_idx = best_candidate['index']
best_hr = best_candidate['hr']
pc = best_candidate['signal']

print(f"\n✅ BEST CHOICE: v{best_idx} (p{best_idx})")
print(f"   Heart Rate (pre-filter): {best_hr:.2f} BPM")
print(f"   Harmonic Score: {best_candidate['harmonic_score']:.0f}")

# ============================================================
# 5) Adaptive Spectral Filtering (paper Eq. 6–7)
# ============================================================
print("\n" + "="*60)
print("STEP 5: Adaptive Spectral Filtering (Eq. 6–7)")
print("="*60)

f_hr_hz, f_full, F_full = estimate_fhr_hz(pc, fs, fmin=FREQ_MIN, fmax=FREQ_MAX)
print(f"Estimated fHR from pc(t): {f_hr_hz:.4f} Hz  ({f_hr_hz*60:.2f} BPM)")

pd_sig, f_spec, mask_spec = adaptive_spectral_filter(pc, fs, f_hr_hz, omega=OMEGA)

f_hr_post_hz, _, _ = estimate_fhr_hz(pd_sig, fs, fmin=FREQ_MIN, fmax=FREQ_MAX)
print(f"Estimated fHR after filtering: {f_hr_post_hz:.4f} Hz  ({f_hr_post_hz*60:.2f} BPM)")

# ============================================================
# 6) Visualizations
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 13))

# Plot 1: Top 5 pulse signals (first 10 seconds)
axes[0].set_title('Top 5 Pulse Signals (First 10 seconds)', fontsize=12, fontweight='bold')
colors = ['blue', 'red', 'green', 'orange', 'purple']
t10 = frames[:300] / fs

for i, c in enumerate(candidates):
    sig = c['signal'][:300]
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)
    alpha = 1.0 if c['index'] == best_idx else 0.4
    lw = 2 if c['index'] == best_idx else 1
    label = f"p{c['index']} ({c['hr']:.1f} BPM)" + (" ✓ BEST" if c['index'] == best_idx else "")
    axes[0].plot(t10, sig_norm + i*3, color=colors[i], alpha=alpha, linewidth=lw, label=label)

axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Normalized (offset)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Candidate spectra
axes[1].set_title('Frequency Spectra Comparison (HR band)', fontsize=12, fontweight='bold')
for i, c in enumerate(candidates):
    alpha = 1.0 if c['index'] == best_idx else 0.3
    lw = 2 if c['index'] == best_idx else 1
    label = f"p{c['index']}" + (" ✓" if c['index'] == best_idx else "")
    axes[1].plot(c['xf_hr'] * 60, c['yf_hr'] / (c['yf_hr'].max() + 1e-12),
                 color=colors[i], alpha=alpha, linewidth=lw, label=label)

axes[1].axvline(best_hr, color='red', linestyle='--', linewidth=2, alpha=0.6,
                label=f'Pre-filter: {best_hr:.1f} BPM')
axes[1].set_xlabel('Heart Rate (BPM)')
axes[1].set_ylabel('Normalized Power')
axes[1].set_xlim(HR_MIN, HR_MAX)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Before vs After filtering (time domain)
axes[2].set_title('Selected Signal: Before vs After Adaptive Spectral Filtering', fontsize=12, fontweight='bold')
t_full = frames / fs
Nview = min(600, len(pc))
pc_n = pc[:Nview] / (np.std(pc[:Nview]) + 1e-12)
pd_n = pd_sig[:Nview] / (np.std(pd_sig[:Nview]) + 1e-12)
axes[2].plot(t_full[:Nview], pc_n, label='pc(t) before', alpha=0.6)
axes[2].plot(t_full[:Nview], pd_n, label='pd(t) after', linewidth=2)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Normalized amplitude')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Mask visualization (frequency domain)
axes[3].set_title('Adaptive Spectral Mask Υ(f)', fontsize=12, fontweight='bold')
axes[3].plot(f_spec, mask_spec, linewidth=2)
axes[3].set_xlim(0, 5)  # show up to 5 Hz
axes[3].set_xlabel('Frequency (Hz)')
axes[3].set_ylabel('Mask value')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('camera_adaptive_spectral_filtering.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 7) Final Results + Save summary CSV
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Selected Eigenvector: v{best_idx}")
print(f"HR pre-filter:  {best_hr:.2f} BPM")
print(f"HR post-filter: {f_hr_post_hz*60:.2f} BPM")
print("\n✅ Saved plot:")
print("  - camera_adaptive_spectral_filtering.png")

results = {
    'method': ['Original (p2)', 'Paper Method (Best pre-filter)', 'Best post adaptive-filter'],
    'eigenvector': [2, best_idx, best_idx],
    'heart_rate_bpm': [candidates[1]['hr'], best_hr, f_hr_post_hz*60],
    'omega_hz': [np.nan, np.nan, OMEGA],
}

pd.DataFrame(results).to_csv('camera_adaptive_spectral_filtering_results.csv', index=False)
print("✅ Saved CSV:")
print("  - camera_adaptive_spectral_filtering_results.csv")
