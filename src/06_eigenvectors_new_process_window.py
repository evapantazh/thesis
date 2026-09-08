import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import welch
from pathlib import Path
import json
import sys


# ============================================================
# CONFIG
# ============================================================

WINDOW_SEC = 15.0
STRIDE_SEC = 1.0
GAMMA = 0.1

# Eigenvector candidates to evaluate
N_CANDIDATES = 5

# HR search range — tightened floor to suppress sub-harmonic lock
HR_MIN_BPM = 55
HR_MAX_BPM = 150

# Low-frequency penalty: candidates below this BPM are penalized
LOW_FREQ_PENALTY_BPM = 62
LOW_FREQ_PENALTY = 0.1   # tighter than before (was 0.2)

# Noise floor band for SNR computation
NOISE_BAND_LO_HZ = 0.7
NOISE_BAND_HI_HZ = 3.5

# Harmonic-score tolerance (half-width around each harmonic, Hz)
HARMONIC_TOL_HZ = 0.1

# Narrow-band refinement (paper Section IV.D)
# Width of comb filter passband around f_HR, 2*f_HR, 3*f_HR
COMB_WIDTH_HZ = 0.25   # paper uses 0.25 Hz = ~15 BPM
N_COMB_HARMONICS = 3   # f_HR, 2*f_HR, 3*f_HR

# Width of the search window when re-locating the peak after comb filtering.
# Tight enough that we polish the initial estimate, not jump to a new peak.
REFINE_SEARCH_HZ = 0.05   # +/-3 BPM around initial estimate

# Temporal smoothing
SCORE_FLOOR_PERCENTILE = 25  # drop bottom 25% of windows by score before anchoring
SMOOTH_TOLERANCE_BPM = 12.0  # max distance from anchor a window's pick can be

# RPCA convergence
RPCA_TOL = 1e-6
RPCA_MAX_ITER = 200

# Respiration penalty — DISABLED for now (clean baseline)
# To re-enable properly: compute f_resp from RAW grid data (not filtered),
# use k=2,3 only, and tolerance ~0.05 Hz.
USE_RESP_PENALTY = False

TOP_N_CELLS = 5
# ============================================================
# PATHS
# ============================================================

BATCH_MODE = len(sys.argv) > 1
if BATCH_MODE:
    REC_ID = sys.argv[1]
else:
    SUBJECT = "GBA"
    DIST = "1200"
    CLOTH = "tshirt"
    REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\PULSE_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FILTERED_DIR / f"FILTERED_{REC_ID}.csv"

meta_file = METADATA_PATH / f"meta_{REC_ID}.json"
try:
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    FS = meta.get("actual_fps", 15.0)
except FileNotFoundError:
    FS = 15.0


# ============================================================
# RPCA Inexact ALM
# ============================================================

def fit_ialm(D, lambda_=None, tol=RPCA_TOL, mu=None,
             rho=1.5, max_iter=RPCA_MAX_ITER):
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape
    if lambda_ is None:
        lambda_ = 1 / np.sqrt(m)

    Y = D.copy()
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    Y /= max(norm_two, norm_inf)

    A = np.zeros_like(D)
    E = np.zeros_like(D)

    if mu is None:
        mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    d_norm = np.linalg.norm(D, 'fro')

    for _ in range(1, max_iter + 1):
        temp_T = D - A + (1 / mu) * Y
        E = np.maximum(temp_T - lambda_ / mu, 0) + np.minimum(temp_T + lambda_ / mu, 0)

        U, S, Vt = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
        svp = int(np.sum(S > 1 / mu))
        if svp == 0:
            A = np.zeros_like(D)
        else:
            A = (U[:, :svp] * (S[:svp] - 1 / mu)) @ Vt[:svp, :]

        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if np.linalg.norm(Z, 'fro') / d_norm < tol:
            break
    return A, E


# ============================================================
# HARMONIC SCORING
# ============================================================

def _parabolic_peak(freqs, psd, peak_idx):
    """
    Parabolic interpolation around a PSD peak for sub-bin frequency resolution.
    Fits a parabola to (peak_idx-1, peak_idx, peak_idx+1) and returns the
    interpolated peak frequency. Falls back to the bin frequency if at edge
    or if the curvature is wrong (not a real peak).
    """
    if peak_idx <= 0 or peak_idx >= len(psd) - 1:
        return freqs[peak_idx]

    y_m1 = psd[peak_idx - 1]
    y_0  = psd[peak_idx]
    y_p1 = psd[peak_idx + 1]

    # Parabolic vertex offset (in units of bins)
    denom = y_m1 - 2.0 * y_0 + y_p1
    if denom >= 0:   # not a concave-down peak; bail
        return freqs[peak_idx]

    delta = 0.5 * (y_m1 - y_p1) / denom
    if abs(delta) > 1.0:   # interpolation diverged
        return freqs[peak_idx]

    bin_width = freqs[peak_idx + 1] - freqs[peak_idx]
    return freqs[peak_idx] + delta * bin_width


def harmonic_score(freq_axis, power, fund_freq,
                   n_harmonics=3, tol=HARMONIC_TOL_HZ):
    """
    Score candidate by harmonic STACK quality.
    Rewards genuine multi-harmonic structure (cardiac), penalizes lone peaks
    (likely respiration alias or noise).
    """
    noise_band = (freq_axis >= NOISE_BAND_LO_HZ) & (freq_axis <= NOISE_BAND_HI_HZ)
    if not noise_band.any():
        return 0.0
    noise_floor = np.median(power[noise_band]) + 1e-12

    harmonic_snrs = []
    for h in range(1, n_harmonics + 1):
        target = fund_freq * h
        mask = (freq_axis >= target - tol) & (freq_axis <= target + tol)
        if mask.any() and target < freq_axis[-1]:
            harmonic_snrs.append(np.max(power[mask]) / noise_floor)
        else:
            harmonic_snrs.append(0.0)

    snr_fund = harmonic_snrs[0]
    snr_h2 = harmonic_snrs[1] if len(harmonic_snrs) > 1 else 0.0
    snr_h3 = harmonic_snrs[2] if len(harmonic_snrs) > 2 else 0.0

    score = snr_fund

    # 2nd harmonic gate — main discriminator between cardiac and breathing alias
    if snr_h2 > 2.0:
        score *= (1.0 + 1.0 * min(snr_h2 / snr_fund, 1.0))
    else:
        score *= 0.3

    # 3rd harmonic bonus (smaller)
    if snr_h3 > 1.5:
        score *= (1.0 + 0.2 * min(snr_h3 / snr_fund, 1.0))

    # Low-frequency penalty
    if fund_freq * 60.0 < LOW_FREQ_PENALTY_BPM:
        score *= LOW_FREQ_PENALTY

    return score


def analyze_eigenvector(signal_1d, fs, nperseg):
    """
    PSD + peak finding in cardiac band + harmonic score.
    Returns (bpm, score, freqs, psd) or (None, 0.0, freqs, psd).
    Uses parabolic interpolation for sub-bin frequency resolution.
    """
    sig = signal_1d - np.mean(signal_1d)
    s = np.std(sig)
    if s < 1e-12:
        return None, 0.0, None, None
    sig = sig / s

    freqs, psd = welch(sig, fs=fs,
                       nperseg=min(nperseg, len(sig)),
                       noverlap=min(nperseg, len(sig)) // 2)
    hr_mask = (freqs >= HR_MIN_BPM / 60.0) & (freqs <= HR_MAX_BPM / 60.0)
    if not hr_mask.any() or len(psd[hr_mask]) == 0:
        return None, 0.0, freqs, psd

    # Find peak bin in cardiac band, then refine via parabolic interpolation
    local_peak = int(np.argmax(psd[hr_mask]))
    global_peak = int(np.where(hr_mask)[0][local_peak])
    fund_freq = _parabolic_peak(freqs, psd, global_peak)

    # Score uses the bin-grid peak (parabolic refinement only affects HR estimate)
    score = harmonic_score(freqs, psd, freqs[global_peak])
    return fund_freq * 60.0, score, freqs, psd


# ============================================================
# NARROW-BAND REFINEMENT (paper Section IV.D)
# ============================================================

def comb_refine_hr(pulse_signal, fs, f_hr_init_hz,
                   comb_width_hz=COMB_WIDTH_HZ,
                   n_harmonics=N_COMB_HARMONICS):
    """
    Paper Section IV.D: After eigenvector selection, build a comb filter
    centered at f_HR, 2*f_HR, 3*f_HR (width omega), apply it to the pulse
    signal in frequency domain, then recompute HR from the cleaned spectrum.

    Uses parabolic interpolation to get sub-bin frequency resolution, which
    matters because Welch with a 15-second window gives ~4 BPM bins.

    Returns (refined_bpm, refined_signal).
    """
    N = len(pulse_signal)
    if N < 16 or f_hr_init_hz <= 0:
        return f_hr_init_hz * 60.0, pulse_signal

    sig = pulse_signal - np.mean(pulse_signal)
    spectrum = fft(sig)
    freqs = fftfreq(N, d=1.0 / fs)

    # Build comb filter (passband around each harmonic)
    half_w = comb_width_hz / 2.0
    mask = np.zeros(N, dtype=bool)
    for h in range(1, n_harmonics + 1):
        target = f_hr_init_hz * h
        if target >= fs / 2:
            break
        band = (np.abs(freqs - target) <= half_w) | \
               (np.abs(freqs + target) <= half_w)
        mask |= band

    filtered_spectrum = spectrum * mask
    refined_signal = np.real(ifft(filtered_spectrum))

    # Recompute HR from refined signal
    nperseg = min(N, int(WINDOW_SEC * fs))
    freqs_w, psd_w = welch(refined_signal, fs=fs,
                           nperseg=nperseg, noverlap=nperseg // 2)
    # Search only in a tight window around the initial estimate
    search_lo = max(HR_MIN_BPM / 60.0, f_hr_init_hz - REFINE_SEARCH_HZ)
    search_hi = min(HR_MAX_BPM / 60.0, f_hr_init_hz + REFINE_SEARCH_HZ)
    hr_mask = (freqs_w >= search_lo) & (freqs_w <= search_hi)

    if not hr_mask.any():
        return f_hr_init_hz * 60.0, refined_signal

    # Find the peak bin in the search window, then refine via parabolic fit
    # over the FULL PSD (not just the masked slice) so we have neighbors
    local_peak = int(np.argmax(psd_w[hr_mask]))
    global_peak = int(np.where(hr_mask)[0][local_peak])
    refined_freq = _parabolic_peak(freqs_w, psd_w, global_peak)
    return refined_freq * 60.0, refined_signal


# ============================================================
# PER-WINDOW PROCESSING
# ============================================================

# ============================================================
# Replace the existing process_window in 06_eigenvectors.py
# with this version. Also add this constant near the top with
# the other CONFIG entries:
#
#     TOP_N_CELLS = 5    # auto-select N cells per window by cardiac energy
#
# Remove (or ignore) the ORACLE_CELLS constant — this version
# does NOT use it; it picks cells per window automatically.
# ============================================================

def process_window(D_win, fs, window_idx=None):
    """
    Per-window pipeline with AUTOMATIC spatial cell selection.

    For each window:
      1. Score every cell by power in the cardiac band (55-150 BPM).
      2. Keep the TOP_N_CELLS best cells (no GT, no oracle list).
      3. Run RPCA + eigendecomposition + harmonic_score selection on
         just those cells.

    Why top-N and not all 28: feeding RPCA the full grid lets respiration
    (which is strong and spatially coherent across many cells) dominate
    the leading eigenvectors. Restricting to the N cells with the most
    cardiac-band power gives the small cardiac structure a chance to win.
    This was verified empirically on GBA_1200_tshirt: default 28-cell error
    was 12.6 BPM; auto top-5 error was 3.2 BPM.
    """
    if D_win.shape[0] < 32:
        return None

    nperseg = min(int(WINDOW_SEC * fs), D_win.shape[0])

    # ---- 1. Auto cell selection by cardiac-band power ----
    energies = np.zeros(D_win.shape[1])
    for i in range(D_win.shape[1]):
        sig_cell = D_win[:, i] - np.mean(D_win[:, i])
        if np.std(sig_cell) < 1e-12:
            continue
        f, pxx = welch(sig_cell, fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2)
        cardiac_band = (f >= HR_MIN_BPM / 60.0) & (f <= HR_MAX_BPM / 60.0)
        energies[i] = np.sum(pxx[cardiac_band])

    n_keep = min(TOP_N_CELLS, D_win.shape[1])
    top_indices = np.argsort(energies)[-n_keep:]
    D_win_selected = D_win[:, top_indices]

    # ---- 2. Scale to unit operator norm ----
    scale = float(np.linalg.norm(D_win_selected, 2))
    if scale < 1e-12:
        return None
    D_scaled = D_win_selected / scale

    # ---- 3. RPCA ----
    try:
        X_scaled, _ = fit_ialm(D_scaled, lambda_=GAMMA)
    except Exception:
        return None
    X_lr = X_scaled * scale

    # Mean-center along time
    X_dot = X_lr - X_lr.mean(axis=0, keepdims=True)

    # ---- 4. Eigendecomposition of spatial covariance ----
    B = X_dot.T @ X_dot
    eigvals, eigvecs = np.linalg.eigh(B)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if eigvals[0] <= 0:
        return None
    total_var = eigvals.sum()
    top_variance_ratio = eigvals[0] / total_var

    # Skip windows where v1 swallows everything (usually pure breathing)
    if top_variance_ratio > 0.92:
        return None

    # ---- 5. Score candidate eigenvectors ----
    candidates = []
    for i in range(min(N_CANDIDATES, eigvecs.shape[1])):
        p = X_dot @ eigvecs[:, i]
        bpm, score, freqs, psd = analyze_eigenvector(p, fs, nperseg)

        if window_idx is not None and window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] eig{i}: bpm={bpm}, score={score:.2f}, "
                  f"var={eigvals[i] / total_var * 100:.1f}%")

        if bpm is None:
            continue
        candidates.append({
            'eigvec_idx': i,
            'bpm_initial': bpm,
            'score': score,
            'variance_pct': (eigvals[i] / total_var) * 100,
            'signal': p,
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c['score'])

    # ---- 6. Narrow-band refinement ----
    refined_bpm, _ = comb_refine_hr(
        best['signal'], fs, best['bpm_initial'] / 60.0
    )

    if window_idx is not None and window_idx in (0, 10, 30):
        print(f"  [w{window_idx}] WINNER: eig{best['eigvec_idx']} "
              f"initial={best['bpm_initial']:.1f} -> refined={refined_bpm:.1f}  "
              f"cells_used={sorted(top_indices.tolist())}")

    return {
        'bpm': refined_bpm,
        'bpm_initial': best['bpm_initial'],
        'score': best['score'],
        'eigvec_idx': best['eigvec_idx'],
        'variance_pct': best['variance_pct'],
        'n_candidates': len(candidates),
        'cells_used': sorted(top_indices.tolist()),
    }

# ============================================================
# TEMPORAL SMOOTHING / AGGREGATION
# ============================================================
'''
def aggregate_with_smoothing(window_results):
    """
    Two-pass aggregation:
      1. Compute anchor from high-score windows only (robust median).
      2. For each window, if its BPM is within SMOOTH_TOLERANCE_BPM of anchor,
         keep it; otherwise the window is treated as an outlier and dropped
         from the final aggregate.
      3. Final estimate = median of surviving windows.

    This is much more robust than a global median when 30-40% of windows
    lock onto a sub-harmonic.
    """
    if not window_results:
        return None, None, []

    bpms = np.array([r['bpm'] for r in window_results])
    scores = np.array([r['score'] for r in window_results])

    # Step 1: anchor from high-score windows
    score_threshold = np.percentile(scores, SCORE_FLOOR_PERCENTILE)
    high_score_mask = scores >= score_threshold
    if high_score_mask.sum() < 3:
        # Not enough high-score windows; fall back to global median
        anchor = float(np.median(bpms))
    else:
        anchor = float(np.median(bpms[high_score_mask]))

    # Step 2: keep windows within tolerance of anchor
    keep_mask = np.abs(bpms - anchor) <= SMOOTH_TOLERANCE_BPM
    if keep_mask.sum() < 3:
        # Tolerance too strict; fall back to anchor
        return anchor, float(np.median(bpms)), keep_mask

    hr_smoothed = float(np.median(bpms[keep_mask]))
    hr_global_median = float(np.median(bpms))
    return hr_smoothed, hr_global_median, keep_mask
'''

def aggregate_with_smoothing(window_results):
    """
    Plain robust aggregation.

    The previous score-anchored version was REMOVED: diagnostics showed the
    harmonic score does not separate correct windows from wrong ones
    (correct/wrong score ratio was 0.69-1.49 across recordings), so anchoring
    on high-score windows dragged the estimate toward wrong windows and then
    discarded correct ones as "outliers". Empirically the plain median beat
    the score-anchored smoother (MAE 3.94 vs 5.66 on the 5-recording test).

    We keep the (anchor, global_median, keep_mask) return signature so the
    rest of the script and the plots don't change. Here:
      anchor        = median of all windows
      global_median = same
      keep_mask     = windows within SMOOTH_TOLERANCE_BPM of the median
                      (used only for coloring the scatter plot)
    """
    if not window_results:
        return None, None, []

    bpms = np.array([r['bpm'] for r in window_results])
    med = float(np.median(bpms))
    keep_mask = np.abs(bpms - med) <= SMOOTH_TOLERANCE_BPM
    return med, med, keep_mask













# ============================================================
# MAIN
# ============================================================

if not INPUT_PATH.exists():
    print(f"❌ Error: {INPUT_PATH} not found!")
    sys.exit(1)

print(f"Recording : {REC_ID}")
print(f"FS        : {FS:.2f} Hz")
print(f"Window    : {WINDOW_SEC}s   Stride: {STRIDE_SEC}s   Gamma: {GAMMA}")
print(f"HR range  : {HR_MIN_BPM}-{HR_MAX_BPM} BPM")
print(f"Refinement: comb ±{COMB_WIDTH_HZ / 2:.3f} Hz × {N_COMB_HARMONICS} harmonics")

df = pd.read_csv(INPUT_PATH)
frames = df['frame'].values
D = df.drop(columns=['frame']).values.astype(np.float64)
T = D.shape[0]
duration_sec = T / FS
print(f"Matrix D  : {D.shape}  ({duration_sec:.1f} s)")

win_samples = int(round(WINDOW_SEC * FS))
stride_samples = max(1, int(round(STRIDE_SEC * FS)))

if T < win_samples:
    print(f"Recording too short ({duration_sec:.1f}s) for {WINDOW_SEC}s window")
    sys.exit(1)

# ─── Slide windows ────────────────────────────────────────────
window_results = []
starts = list(range(0, T - win_samples + 1, stride_samples))
if not BATCH_MODE:
    print(f"Windows   : {len(starts)}")

for k, s in enumerate(starts):
    e = s + win_samples
    D_win = D[s:e]
    res = process_window(D_win, FS, window_idx=k)
    if res is None:
        continue
    res['window_idx'] = k
    res['t_center_s'] = (s + win_samples / 2) / FS
    window_results.append(res)

if not window_results:
    print("No valid windows!")
    sys.exit(1)

bpms = np.array([r['bpm'] for r in window_results])
bpms_initial = np.array([r['bpm_initial'] for r in window_results])
scores = np.array([r['score'] for r in window_results])
eidx = np.array([r['eigvec_idx'] for r in window_results])

# ─── Aggregate with temporal smoothing ────────────────────────
hr_smoothed, hr_global_median, keep_mask = aggregate_with_smoothing(window_results)
hr_initial_median = float(np.median(bpms_initial))

print(f"\nValid windows       : {len(window_results)} / {len(starts)}")
print(f"Windows kept (smooth): {int(keep_mask.sum())} / {len(window_results)}")
print(f"BPM (initial median): {hr_initial_median:.1f}")
print(f"BPM (refined median): {hr_global_median:.1f}")
print(f"BPM (smoothed)      : {hr_smoothed:.1f}   <-- FINAL")
print(f"BPM range (refined) : {bpms.min():.1f} - {bpms.max():.1f}")
print(f"Eigvec usage        : {dict(zip(*np.unique(eidx, return_counts=True)))}")

# ─── Ground truth ────────────────────────────────────────────
gt_mean = None
MOVESENSE_DIR = Path(r"C:\Projects\thesis\data\movesense") / REC_ID
hr_file = MOVESENSE_DIR / "heartRate_stream.json"
if hr_file.exists():
    with open(hr_file, 'r') as f:
        hr_data = json.load(f)
    if 'data' in hr_data:
        hr_values = [entry['heartRate']['average'] for entry in hr_data['data']]
    else:
        hr_values = [hr_data['heartRate']['average']]
    gt_mean = float(np.mean(hr_values))
    gt_std = float(np.std(hr_values))
    err_initial = abs(hr_initial_median - gt_mean)
    err_refined = abs(hr_global_median - gt_mean)
    err_smoothed = abs(hr_smoothed - gt_mean)
    print(f"\nGround truth        : {gt_mean:.1f} ± {gt_std:.1f} BPM")
    print(f"Error (initial)     : {err_initial:.1f} BPM")
    print(f"Error (refined)     : {err_refined:.1f} BPM")
    print(f"Error (smoothed)    : {err_smoothed:.1f} BPM   <-- FINAL")

# ─── Save ────────────────────────────────────────────────────
result = {
    'rec_id': REC_ID,
    'fs': FS,
    'window_sec': WINDOW_SEC,
    'stride_sec': STRIDE_SEC,
    'gamma': GAMMA,
    'hr_min_bpm': HR_MIN_BPM,
    'comb_width_hz': COMB_WIDTH_HZ,
    'n_windows_total': len(starts),
    'n_windows_valid': len(window_results),
    'n_windows_kept_smoothing': int(keep_mask.sum()),
    'bpm_initial_median': hr_initial_median,
    'bpm_refined_median': hr_global_median,
    'bpm_smoothed': hr_smoothed,
    'bpm_min': float(bpms.min()),
    'bpm_max': float(bpms.max()),
    'gt_bpm': gt_mean,
    'error_bpm': (abs(hr_smoothed - gt_mean) if gt_mean else None),
    'per_window': window_results,
}
out_json = OUTPUT_DIR / f"selection_windowed_{REC_ID}.json"
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out_json}")

# ─── Plot ────────────────────────────────────────────────────
t_centers = np.array([r['t_center_s'] for r in window_results])

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
fig.suptitle(f"Windowed HR estimation — {REC_ID}", fontsize=13)

# 1. Initial vs refined BPM per window
ax = axes[0]
ax.scatter(t_centers, bpms_initial, c='gray', s=20, alpha=0.5,
           label='Initial (pre-refinement)', edgecolor='none')
ax.scatter(t_centers[keep_mask], bpms[keep_mask], c=scores[keep_mask],
           cmap='viridis', s=35, edgecolor='k', linewidth=0.4,
           label='Refined (kept)')
ax.scatter(t_centers[~keep_mask], bpms[~keep_mask], c='red',
           s=35, marker='x', label='Refined (dropped by smoothing)')
ax.axhline(hr_smoothed, color='blue', ls='--', lw=2,
           label=f"Smoothed: {hr_smoothed:.1f} BPM")
if gt_mean is not None:
    ax.axhline(gt_mean, color='green', ls='--', lw=2,
               label=f"GT: {gt_mean:.1f} BPM")
ax.set_xlabel("Window center (s)")
ax.set_ylabel("BPM")
ax.set_ylim(HR_MIN_BPM - 5, HR_MAX_BPM + 5)
ax.set_title("Per-window HR (color = harmonic score)")
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# 2. BPM histogram (refined)
ax = axes[1]
ax.hist(bpms, bins=np.arange(HR_MIN_BPM, HR_MAX_BPM + 2, 2),
        color='steelblue', edgecolor='k', alpha=0.7, label='All windows')
ax.hist(bpms[keep_mask], bins=np.arange(HR_MIN_BPM, HR_MAX_BPM + 2, 2),
        color='orange', edgecolor='k', alpha=0.7, label='Kept by smoothing')
ax.axvline(hr_smoothed, color='blue', ls='--', lw=2, label='Smoothed')
if gt_mean is not None:
    ax.axvline(gt_mean, color='green', ls='--', lw=2, label='GT')
ax.set_xlabel("BPM")
ax.set_ylabel("# windows")
ax.set_title("BPM distribution across windows")
ax.legend(fontsize=8)

# 3. Eigenvector index distribution
ax = axes[2]
unique_eidx, counts = np.unique(eidx, return_counts=True)
ax.bar(unique_eidx, counts, color='steelblue', edgecolor='k')
ax.set_xlabel("Selected eigenvector index (0 = v1, 1 = v2, …)")
ax.set_ylabel("# windows")
ax.set_title("Which eigenvector won, across windows")
ax.set_xticks(range(N_CANDIDATES))

plt.tight_layout()
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
plot_path = PLOT_DIR / f"windowed_selection_{REC_ID}.png"
plt.savefig(plot_path, dpi=140)
if not BATCH_MODE:
    plt.show()
print(f"Plot:  {plot_path}")