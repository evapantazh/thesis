"""
eigenvectors_comparison.py
===========================

Unified pipeline for comparing source-separation methods on the
heart-rate estimation task. Three modes:

    METHOD = "RPCA"  : current pipeline (RPCA -> PCA eigendecomp)
    METHOD = "PCA"   : skip RPCA, do PCA eigendecomp directly on D
    METHOD = "ICA"   : skip RPCA, do FastICA(n_components=5) on D

Everything downstream (variance gate, harmonic scoring, comb refinement,
SQI filter, median aggregation) is IDENTICAL across modes. This guarantees
the comparison is fair.

Usage:
    # Set METHOD at top, then:
    python eigenvectors_comparison.py                          # single rec (debug)
    python eigenvectors_comparison.py AGA_1200_tshirt          # one recording
    # Or call in batch from run_all.py (see end of file).

Outputs go to:
    PULSE_files_{METHOD}/selection_windowed_{REC_ID}.json
    PULSE_files_{METHOD}/plots/windowed_selection_{REC_ID}.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import welch
from sklearn.decomposition import FastICA   # NEW
from pathlib import Path
import json
import sys
import warnings


# ============================================================
# METHOD SWITCH  — set this before running
# ============================================================

METHOD = "RPCA"   # "RPCA" | "PCA" | "ICA"


# ============================================================
# CONFIG  -- THESE MUST MATCH YOUR LOCKED SQI=25 PRODUCTION VALUES
# ============================================================

WINDOW_SEC = 15.0
STRIDE_SEC = 1.0

# RPCA regularization. Only used when METHOD == "RPCA".
GAMMA = 0.02        # *** verify this matches your locked production value ***

# Eigenvector / component candidates to evaluate per window
N_CANDIDATES = 5

# HR search range
HR_MIN_BPM = 55
HR_MAX_BPM = 150

# Low-frequency penalty
LOW_FREQ_PENALTY_BPM = 62
LOW_FREQ_PENALTY = 0.1

# Noise floor band for SNR computation
NOISE_BAND_LO_HZ = 0.7
NOISE_BAND_HI_HZ = 3.5

# Harmonic-score tolerance
HARMONIC_TOL_HZ = 0.1

# Narrow-band refinement (comb filter)
COMB_WIDTH_HZ = 0.25
N_COMB_HARMONICS = 3
REFINE_SEARCH_HZ = 0.05

# SQI
SQI_PERCENTILE = 25  # *** verify this matches your locked production value ***
MIN_WINDOWS_AFTER_SQI = 5

# Temporal smoothing tolerance (used in aggregation)
SMOOTH_TOLERANCE_BPM = 12.0

# RPCA convergence
RPCA_TOL = 1e-6
RPCA_MAX_ITER = 200


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

# Method-specific output directory so PCA/ICA/RPCA results don't overwrite each other
OUTPUT_DIR = Path(rf"C:\Projects\thesis\data\PULSE_files_{METHOD}")
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
# RPCA Inexact ALM  -- unchanged from your script
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
# HARMONIC SCORING  -- unchanged from your script
# ============================================================

def _parabolic_peak(freqs, psd, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(psd) - 1:
        return freqs[peak_idx]
    y_m1 = psd[peak_idx - 1]
    y_0 = psd[peak_idx]
    y_p1 = psd[peak_idx + 1]
    denom = y_m1 - 2.0 * y_0 + y_p1
    if denom >= 0:
        return freqs[peak_idx]
    delta = 0.5 * (y_m1 - y_p1) / denom
    if abs(delta) > 1.0:
        return freqs[peak_idx]
    bin_width = freqs[peak_idx + 1] - freqs[peak_idx]
    return freqs[peak_idx] + delta * bin_width


def harmonic_score(freq_axis, power, fund_freq,
                   n_harmonics=3, tol=HARMONIC_TOL_HZ):
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
    if snr_h2 > 2.0:
        score *= (1.0 + 1.0 * min(snr_h2 / snr_fund, 1.0))
    else:
        score *= 0.3
    if snr_h3 > 1.5:
        score *= (1.0 + 0.2 * min(snr_h3 / snr_fund, 1.0))
    if fund_freq * 60.0 < LOW_FREQ_PENALTY_BPM:
        score *= LOW_FREQ_PENALTY
    return score


def analyze_signal(signal_1d, fs, nperseg):
    """PSD + peak finding + harmonic score for one candidate signal."""
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

    local_peak = int(np.argmax(psd[hr_mask]))
    global_peak = int(np.where(hr_mask)[0][local_peak])
    fund_freq = _parabolic_peak(freqs, psd, global_peak)
    score = harmonic_score(freqs, psd, freqs[global_peak])
    return fund_freq * 60.0, score, freqs, psd


# ============================================================
# NARROW-BAND REFINEMENT  -- unchanged from your script
# ============================================================

def comb_refine_hr(pulse_signal, fs, f_hr_init_hz,
                   comb_width_hz=COMB_WIDTH_HZ,
                   n_harmonics=N_COMB_HARMONICS):
    N = len(pulse_signal)
    if N < 16 or f_hr_init_hz <= 0:
        return f_hr_init_hz * 60.0, pulse_signal

    sig = pulse_signal - np.mean(pulse_signal)
    spectrum = fft(sig)
    freqs = fftfreq(N, d=1.0 / fs)

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

    nperseg = min(N, int(WINDOW_SEC * fs))
    freqs_w, psd_w = welch(refined_signal, fs=fs,
                           nperseg=nperseg, noverlap=nperseg // 2)
    search_lo = max(HR_MIN_BPM / 60.0, f_hr_init_hz - REFINE_SEARCH_HZ)
    search_hi = min(HR_MAX_BPM / 60.0, f_hr_init_hz + REFINE_SEARCH_HZ)
    hr_mask = (freqs_w >= search_lo) & (freqs_w <= search_hi)
    if not hr_mask.any():
        return f_hr_init_hz * 60.0, refined_signal

    local_peak = int(np.argmax(psd_w[hr_mask]))
    global_peak = int(np.where(hr_mask)[0][local_peak])
    refined_freq = _parabolic_peak(freqs_w, psd_w, global_peak)
    return refined_freq * 60.0, refined_signal


# ============================================================
# PER-WINDOW PROCESSING  -- modified for METHOD switch
# ============================================================

def process_window(D_win, fs, window_idx=None):
    """
    Three modes via METHOD global:
      RPCA: RPCA on D_win, eigendecomposition on L
      PCA : skip RPCA, eigendecomposition directly on D_win
      ICA : skip RPCA, FastICA(n_components=N_CANDIDATES) on D_win
    Everything downstream is identical.
    """
    if D_win.shape[0] < 32:
        return None

    # ---- Source separation step (THE ONLY THING THAT VARIES BY METHOD) ----
    candidate_signals = []  # list of 1-D arrays
    variance_ratios = []    # for diagnostics; only meaningful for PCA-based modes

    if METHOD == "RPCA":
        # Operator-norm scaling
        scale = float(np.linalg.norm(D_win, 2))
        if scale < 1e-12:
            return None
        D_scaled = D_win / scale
        try:
            X_scaled, _ = fit_ialm(D_scaled, lambda_=GAMMA)
        except Exception:
            return None
        X_lr = X_scaled * scale
        X_dot = X_lr - X_lr.mean(axis=0, keepdims=True)

        # Eigendecomposition on the low-rank matrix
        B = X_dot.T @ X_dot
        eigvals, eigvecs = np.linalg.eigh(B)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 0:
            return None
        total_var = eigvals.sum()
        # Variance gate
        if eigvals[0] / total_var > 0.92:
            return None
        for i in range(min(N_CANDIDATES, eigvecs.shape[1])):
            candidate_signals.append(X_dot @ eigvecs[:, i])
            variance_ratios.append(eigvals[i] / total_var)

    elif METHOD == "PCA":
        # No RPCA: eigendecomposition directly on the filtered matrix D_win
        D_dot = D_win - D_win.mean(axis=0, keepdims=True)
        B = D_dot.T @ D_dot
        eigvals, eigvecs = np.linalg.eigh(B)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 0:
            return None
        total_var = eigvals.sum()
        # Variance gate (same threshold)
        if eigvals[0] / total_var > 0.92:
            return None
        for i in range(min(N_CANDIDATES, eigvecs.shape[1])):
            candidate_signals.append(D_dot @ eigvecs[:, i])
            variance_ratios.append(eigvals[i] / total_var)

    elif METHOD == "ICA":
        # No RPCA: FastICA directly on the filtered matrix
        D_dot = D_win - D_win.mean(axis=0, keepdims=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # silence ICA convergence warnings
                ica = FastICA(
                    n_components=N_CANDIDATES,
                    random_state=42,
                    max_iter=500,
                    tol=1e-4,
                    whiten='unit-variance',
                )
                # FastICA expects (n_samples, n_features) -> rows=time, cols=cells
                S = ica.fit_transform(D_dot)   # shape: (T, N_CANDIDATES)
        except Exception:
            return None
        if S is None or S.shape[1] == 0:
            return None
        # Variance gate equivalent: skip if any single component carries >92% of total signal energy
        comp_energies = (S ** 2).sum(axis=0)
        total_energy = comp_energies.sum()
        if total_energy <= 0:
            return None
        if comp_energies.max() / total_energy > 0.92:
            return None
        # Sort components by energy descending so candidate ordering is stable
        order = np.argsort(-comp_energies)
        for j in order[:N_CANDIDATES]:
            candidate_signals.append(S[:, j])
            variance_ratios.append(comp_energies[j] / total_energy)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD!r}. Use 'RPCA', 'PCA', or 'ICA'.")

    # ---- Downstream: same for all three methods ----
    if not candidate_signals:
        return None

    nperseg = min(int(WINDOW_SEC * fs), D_win.shape[0])
    candidates = []
    for i, p in enumerate(candidate_signals):
        bpm, score, freqs, psd = analyze_signal(p, fs, nperseg)
        if bpm is None:
            continue
        candidates.append({
            'eigvec_idx': i,
            'bpm_initial': bpm,
            'score': score,
            'variance_pct': float(variance_ratios[i]) * 100.0,
            'signal': p,
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c['score'])
    refined_bpm, _ = comb_refine_hr(
        best['signal'], fs, best['bpm_initial'] / 60.0
    )

    return {
        'bpm': refined_bpm,
        'bpm_initial': best['bpm_initial'],
        'score': best['score'],
        'eigvec_idx': best['eigvec_idx'],
        'variance_pct': best['variance_pct'],
        'n_candidates': len(candidates),
    }


# ============================================================
# TEMPORAL SMOOTHING / AGGREGATION  -- unchanged
# ============================================================

def aggregate_with_smoothing(window_results):
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
    print(f"[ERROR] {INPUT_PATH} not found")
    sys.exit(1)

print(f"METHOD    : {METHOD}")
print(f"Recording : {REC_ID}")
print(f"FS        : {FS:.2f} Hz")
print(f"Window    : {WINDOW_SEC}s   Stride: {STRIDE_SEC}s")
if METHOD == "RPCA":
    print(f"Gamma     : {GAMMA}")
print(f"SQI pct   : {SQI_PERCENTILE}")

df = pd.read_csv(INPUT_PATH)
frames = df['frame'].values
D = df.drop(columns=['frame']).values.astype(np.float64)
T = D.shape[0]
duration_sec = T / FS
print(f"Matrix D  : {D.shape}  ({duration_sec:.1f} s)")

win_samples = int(round(WINDOW_SEC * FS))
stride_samples = max(1, int(round(STRIDE_SEC * FS)))

if T < win_samples:
    print(f"Recording too short ({duration_sec:.1f}s)")
    sys.exit(1)

# Slide windows
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
    print("[ERROR] No valid windows")
    sys.exit(1)

# SQI filter
n_before_sqi = len(window_results)
if SQI_PERCENTILE > 0 and n_before_sqi >= MIN_WINDOWS_AFTER_SQI * 2:
    scores_array = np.array([r['score'] for r in window_results])
    sqi_threshold = np.percentile(scores_array, SQI_PERCENTILE)
    filtered = [r for r in window_results if r['score'] >= sqi_threshold]
    if len(filtered) >= MIN_WINDOWS_AFTER_SQI:
        window_results = filtered
        sqi_applied = True
        sqi_threshold_value = float(sqi_threshold)
        print(f"SQI filter: kept {len(window_results)}/{n_before_sqi} "
              f"(threshold={sqi_threshold:.3f})")
    else:
        sqi_applied = False
        sqi_threshold_value = None
        print(f"SQI filter: skipped (only {len(filtered)} would survive)")
else:
    sqi_applied = False
    sqi_threshold_value = None
    print(f"SQI filter: disabled")

bpms = np.array([r['bpm'] for r in window_results])
bpms_initial = np.array([r['bpm_initial'] for r in window_results])
scores = np.array([r['score'] for r in window_results])
eidx = np.array([r['eigvec_idx'] for r in window_results])

hr_smoothed, hr_global_median, keep_mask = aggregate_with_smoothing(window_results)
hr_initial_median = float(np.median(bpms_initial))

print(f"\nValid windows: {len(window_results)} / {len(starts)}")
print(f"BPM (initial median) : {hr_initial_median:.1f}")
print(f"BPM (refined median) : {hr_global_median:.1f}")
print(f"BPM (smoothed)       : {hr_smoothed:.1f}   <-- FINAL")
print(f"BPM range            : {bpms.min():.1f} - {bpms.max():.1f}")
print(f"Component usage      : {dict(zip(*np.unique(eidx, return_counts=True)))}")

# Ground truth
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
    err_smoothed = abs(hr_smoothed - gt_mean)
    print(f"\nGround truth: {gt_mean:.1f} BPM")
    print(f"Error (smoothed): {err_smoothed:.1f} BPM   <-- FINAL")

# Save
result = {
    'rec_id': REC_ID,
    'method': METHOD,
    'fs': FS,
    'window_sec': WINDOW_SEC,
    'stride_sec': STRIDE_SEC,
    'gamma': GAMMA if METHOD == "RPCA" else None,
    'sqi_percentile': SQI_PERCENTILE,
    'sqi_applied': sqi_applied,
    'sqi_threshold_value': sqi_threshold_value,
    'n_windows_total': len(starts),
    'n_windows_before_sqi': n_before_sqi,
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

# Plot (only in interactive mode)
if not BATCH_MODE:
    t_centers = np.array([r['t_center_s'] for r in window_results])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{METHOD} — {REC_ID}", fontsize=13)

    ax = axes[0]
    ax.scatter(t_centers[keep_mask], bpms[keep_mask], c=scores[keep_mask],
               cmap='viridis', s=35, edgecolor='k', linewidth=0.4,
               label='Kept')
    ax.scatter(t_centers[~keep_mask], bpms[~keep_mask], c='red',
               s=35, marker='x', label='Dropped (smoothing)')
    ax.axhline(hr_smoothed, color='blue', ls='--', lw=2,
               label=f"Smoothed: {hr_smoothed:.1f}")
    if gt_mean is not None:
        ax.axhline(gt_mean, color='green', ls='--', lw=2,
                   label=f"GT: {gt_mean:.1f}")
    ax.set_xlabel("Window center (s)")
    ax.set_ylabel("BPM")
    ax.set_ylim(HR_MIN_BPM - 5, HR_MAX_BPM + 5)
    ax.set_title(f"Per-window HR ({METHOD})")
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    unique_eidx, counts = np.unique(eidx, return_counts=True)
    ax.bar(unique_eidx, counts, color='steelblue', edgecolor='k')
    ax.set_xlabel("Selected component index")
    ax.set_ylabel("# windows")
    ax.set_title(f"Component usage ({METHOD})")
    ax.set_xticks(range(N_CANDIDATES))

    plt.tight_layout()
    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"windowed_selection_{REC_ID}.png"
    plt.savefig(plot_path, dpi=140)
    plt.show()
    print(f"Plot: {plot_path}")