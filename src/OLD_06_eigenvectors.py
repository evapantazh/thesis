import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from pathlib import Path
import json
import sys


# -----------------------------------------------

WINDOW_SEC = 15.0
STRIDE_SEC = 1.0
GAMMA = 0.1 #paper

# How many eigenvectors to test
N_CANDIDATES = 5

# HR search range
HR_MIN_BPM = 50   #raise from 40 to have a stricter limit
HR_MAX_BPM = 150

LOW_FREQ_PENALTY_BPM = 65 #55  #below this bpm, they are penalised
LOW_FREQ_PENALTY =  0.2 #0.3

# noise floor band, cardiac search range
NOISE_BAND_LO_HZ = 0.7
NOISE_BAND_HI_HZ = 3.5

# Harmonic-score tolerance (half-width around each harmonic, Hz)
HARMONIC_TOL_HZ = 0.1

# RPCA convergence (loose for speed since we run it many times)
RPCA_TOL      = 1e-6
RPCA_MAX_ITER = 200

# -------------------------------------------


BATCH_MODE = len(sys.argv) > 1
if BATCH_MODE:
    REC_ID = sys.argv[1]
else:
    # --- 1. SETTINGS & PATHS ---
    SUBJECT = "AGE"
    DIST = "800"
    CLOTH = "tshirt"
    REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\PULSE_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FILTERED_DIR / f"FILTERED_{REC_ID}.csv"

# Load FPS from metadata
meta_file = METADATA_PATH / f"meta_{REC_ID}.json"
try:
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    FS = meta.get("actual_fps", 15.0)
except FileNotFoundError:
    FS = 15.0

'''
rpca_meta_file = RPCA_DIR / f"rpca_meta_{REC_ID}.json"
try:
    with open(rpca_meta_file, 'r') as f:
        rpca_meta = json.load(f)
    gamma_used = rpca_meta.get("gamma", 0.02)
except FileNotFoundError:
    gamma_used = 0.02
'''
# Find frequency of respiration
def estimate_resp_freq(D_win, fs):
    """
    Estimate respiration fundamental frequency from the mean depth signal
    of the window. Returns frequency in Hz, or None if estimation fails.
    """
    mean_sig = D_win.mean(axis=1)
    mean_sig = mean_sig - mean_sig.mean()
    if np.std(mean_sig) < 1e-12:
        return None

    nperseg = min(len(mean_sig), int(WINDOW_SEC * fs))
    freqs, psd = welch(mean_sig, fs=fs,
                        nperseg=nperseg,
                        noverlap=nperseg // 2)

    # Respiration is in [0.1, 0.5] Hz = [6, 30] BPM
    resp_mask = (freqs >= 0.10) & (freqs <= 0.50)
    if not resp_mask.any():
        return None

    resp_psd = psd[resp_mask]
    resp_freqs = freqs[resp_mask]
    return float(resp_freqs[np.argmax(resp_psd)])


# --- 1. RPCA Inexact ALM ---

#  same as previous existing implementation 05_rpca
#  but with looser convergence to keep windowed runs fast

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
 

# --- 2. HARMONIC SCORING ---
def harmonic_score(freq_axis, power, fund_freq, n_harmonics=3, tol=0.1, f_resp_hz =None, resp_harm_tol_hz =0.08):
    """
    Score candidate by harmonic STACK quality, not fundamental peak height.
    Heavily favor candidates where 2nd harmonic is genuinely present.
    """
    noise_band = (freq_axis >= NOISE_BAND_LO_HZ) & (freq_axis <= NOISE_BAND_HI_HZ)
    if not noise_band.any():
        return 0.0
    noise_floor = np.median(power[noise_band]) + 1e-12

    # Get per-harmonic SNR
    harmonic_snrs = []
    for h in range(1, n_harmonics + 1):
        target = fund_freq * h
        mask = (freq_axis >= target - tol) & (freq_axis <= target + tol)
        if mask.any() and target < freq_axis[-1]:
            harmonic_snrs.append(np.max(power[mask]) / noise_floor)
        else:
            harmonic_snrs.append(0.0)

    snr_fund = harmonic_snrs[0]
    snr_h2   = harmonic_snrs[1] if len(harmonic_snrs) > 1 else 0.0
    snr_h3   = harmonic_snrs[2] if len(harmonic_snrs) > 2 else 0.0

    # Base score: fundamental SNR
    score = snr_fund

    # Reward genuine 2nd harmonic presence (SNR > 2 means real peak)
    if snr_h2 > 2.0:
        score *= (1.0 + 1.0 * min(snr_h2 / snr_fund, 1.0)) #(1.0 + 0.5 * min(snr_h2 / snr_fund, 1.0))
    else:
        # 2nd harmonic absent or weak — likely breathing or noise
        score *= 0.3

    # Reward 3rd harmonic too (smaller bonus)
    if snr_h3 > 1.5:
        score *= (1.0 + 0.2 * min(snr_h3 / snr_fund, 1.0))

    if fund_freq * 60.0 < LOW_FREQ_PENALTY_BPM:
        score *= LOW_FREQ_PENALTY

    # Respiration harmonic
    if f_resp_hz is not None and f_resp_hz>0:
        for k in range(2,7):
            f_resp_harm = k*f_resp_hz
            if abs(fund_freq - f_resp_harm) < resp_harm_tol_hz:
                score*=0.85 #0.1
                break


    return score


def analyze_eigenvector(signal_1d, fs, nperseg, f_resp_hz =None):
    """
    PSD, peak finding in cardiac band, harmonic score.
    Returns (bpm, score, freqs, psd) or (None, 0.0, freqs, psd) if no peak.
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
    peak_idx = int(np.argmax(psd[hr_mask]))
    fund_freq = freqs[hr_mask][peak_idx]
    score = harmonic_score(freqs, psd, fund_freq, f_resp_hz=f_resp_hz)
    return fund_freq * 60.0, score, freqs, psd

#  Per-window processing
# ─────────────────────────────────────────────────────────────
def process_window(D_win, fs, window_idx = None):
    """
    Run RPCA, eigendecompose, score top eigenvectors.
    Returns dict with best candidate info, or None if degenerate.
    """
    if D_win.shape[0] < 32:   # at least ~2 s of data
        if window_idx is not None and window_idx == 0:
            print(f"  [w{window_idx}] FAIL: too few samples ({D_win.shape[0]})")
        return None
    
    # Check input signal energy
    if window_idx is not None and window_idx in (0, 10, 30):
        print(f"  [w{window_idx}] D_win stats: shape={D_win.shape}, "
              f"std={D_win.std():.6f}, max_abs={np.abs(D_win).max():.6f}")
        
    # NEW: estimate respiration frequency for this window
    f_resp_hz = estimate_resp_freq(D_win, fs)
    if window_idx is not None and window_idx in (0, 10, 30):
        if f_resp_hz:
            print(f"  [w{window_idx}] f_resp = {f_resp_hz:.3f} Hz "
                  f"({f_resp_hz*60:.1f} BPM)")

    # Scale to unit operator norm so IALM thresholds work in a sensible range
    scale = float(np.linalg.norm(D_win, 2))
    if scale < 1e-12:
        if window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] FAIL: zero-norm input")
        return None
    D_scaled = D_win / scale

    try:
        X_scaled, _ = fit_ialm(D_scaled, lambda_=GAMMA)
    except Exception as ex:
        if window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] FAIL: RPCA exception: {ex}")
        return None
    X_lr = X_scaled * scale   # undo scaling
        
    
    if window_idx is not None and window_idx in (0, 10, 30):
        print(f"  [w{window_idx}] D_win norm={scale:.3f}, "
              f"X_lr std={X_lr.std():.6f}, "
              f"frac_zero={(X_scaled == 0).mean():.3f}")


    # Mean-center along time
    X_dot = X_lr - X_lr.mean(axis=0, keepdims=True)
 
    # Eigendecomposition of spatial covariance
    B = X_dot.T @ X_dot
    eigvals, eigvecs = np.linalg.eigh(B)
    order = eigvals.argsort()[::-1]
    eigvals  = eigvals[order]
    eigvecs  = eigvecs[:, order]
 
    if eigvals[0] <= 0:
        if window_idx is not None and window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] FAIL: eigvals[0]={eigvals[0]:.2e}")
        return None
    total_var = eigvals.sum()
    
    # After eigendecomposition
    top_variance_ratio = eigvals[0] / total_var

    # Skip windows where one eigenvector swallows everything
    if top_variance_ratio > 0.92:
        if window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] SKIP: v1 dominates ({top_variance_ratio*100:.1f}%)")
        return None


    nperseg = min(int(WINDOW_SEC * fs), D_win.shape[0])
    candidates = []
    for i in range(min(N_CANDIDATES, eigvecs.shape[1])):
        p = X_dot @ eigvecs[:, i]
        bpm, score, freqs, psd = analyze_eigenvector(p, fs, nperseg, f_resp_hz)
        resp_flag = ""
        if bpm is not None and f_resp_hz is not None:
            for k in range(2, 7):
                if abs(bpm/60.0 - k * f_resp_hz) < 0.08:
                    resp_flag = f" [HIT resp×{k}]"
                    break
        if window_idx is not None and window_idx in (0, 10, 30):
             print(f"  [w{window_idx}] eig{i}: bpm={bpm}, score={score:.2f}, "
                f"var={eigvals[i]/total_var*100:.1f}%{resp_flag}") 
        
        if bpm is None:
            continue
        candidates.append({
            'eigvec_idx'   : i,
            'bpm'          : bpm,
            'score'        : score,
            'variance_pct' : (eigvals[i] / total_var) * 100,
        })
 
    if not candidates:
        if window_idx is not None and window_idx in (0, 10, 30):
            print(f"  [w{window_idx}] FAIL: no candidates passed")
        return None
 
    best = max(candidates, key=lambda c: c['score'])
    return {
        'bpm'         : best['bpm'],
        'score'       : best['score'],
        'eigvec_idx'  : best['eigvec_idx'],
        'variance_pct': best['variance_pct'],
        'n_candidates': len(candidates),
    }
 

# --- MAIN. EXECUTION ---
if not INPUT_PATH.exists():
    print(f"❌ Error: {INPUT_PATH} not found!")
    sys.exit(1)


print(f"Recording : {REC_ID}")
print(f"FS        : {FS:.2f} Hz")
print(f"Window    : {WINDOW_SEC}s   Stride: {STRIDE_SEC}s   Gamma: {GAMMA}")

df = pd.read_csv(INPUT_PATH)
frames = df['frame'].values

D = df.drop(columns=['frame']).values.astype(np.float64)
T = D.shape[0]
duration_sec = T / FS
print(f"Matrix D  : {D.shape}  ({duration_sec:.1f} s)")
    
win_samples    = int(round(WINDOW_SEC * FS))
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
    res = process_window(D_win, FS, window_idx =k)
    if res is None:
        continue
    res['window_idx']  = k
    res['t_center_s']  = (s + win_samples / 2) / FS
    window_results.append(res)
 
if not window_results:
    print("No valid windows!")
    sys.exit(1)
 
bpms   = np.array([r['bpm']          for r in window_results])
scores = np.array([r['score']        for r in window_results])
eidx   = np.array([r['eigvec_idx']   for r in window_results])
 
# ─── Aggregate ────────────────────────────────────────────────
# Median is robust to a few bad windows; we also report the
# score-weighted median for comparison.
hr_median        = float(np.median(bpms))
hr_weighted_med  = float(np.median(bpms[scores > np.median(scores)]))  # top-half by score
# Αν είμαστε σε BATCH_MODE, τυπώνουμε μόνο ΜΙΑ μαζεμένη γραμμή με το τελικό αποτέλεσμα

print(f"\nValid windows  : {len(window_results)} / {len(starts)}")
print(f"BPM median     : {hr_median:.1f}")
print(f"BPM (top-half) : {hr_weighted_med:.1f}")
print(f"BPM range      : {bpms.min():.1f} – {bpms.max():.1f}")
print(f"Eigvec usage   : {dict(zip(*np.unique(eidx, return_counts=True)))}")
    
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
    gt_std  = float(np.std(hr_values))
    err_med = abs(hr_median - gt_mean)
    err_top = abs(hr_weighted_med - gt_mean)
    print(f"\nGround truth   : {gt_mean:.1f} ± {gt_std:.1f} BPM")
    print(f"Error (median) : {err_med:.1f} BPM")
    print(f"Error (top-half): {err_top:.1f} BPM")
 
# ─── Save ────────────────────────────────────────────────────
result = {
    'rec_id'              : REC_ID,
    'fs'                  : FS,
    'window_sec'          : WINDOW_SEC,
    'stride_sec'          : STRIDE_SEC,
    'gamma'               : GAMMA,
    'n_windows_total'     : len(starts),
    'n_windows_valid'     : len(window_results),
    'bpm_median'          : hr_median,
    'bpm_top_half_median' : hr_weighted_med,
    'bpm_min'             : float(bpms.min()),
    'bpm_max'             : float(bpms.max()),
    'gt_bpm'              : gt_mean,
    'error_bpm'           : (abs(hr_median - gt_mean) if gt_mean else None),
    'per_window'          : window_results,
}
out_json = OUTPUT_DIR / f"selection_windowed_{REC_ID}.json"
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out_json}")
 
# ─── Plot ────────────────────────────────────────────────────
t_centers = np.array([r['t_center_s'] for r in window_results])
 
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
fig.suptitle(f"Windowed HR estimation — {REC_ID}", fontsize=13)
 
# 1. BPM over time
ax = axes[0]
ax.scatter(t_centers, bpms, c=scores, cmap='viridis', s=30,
           edgecolor='k', linewidth=0.3)
ax.axhline(hr_median, color='red', ls='--', lw=2,
           label=f"Median: {hr_median:.1f} BPM")
if gt_mean is not None:
    ax.axhline(gt_mean, color='green', ls='--', lw=2,
               label=f"GT (Movesense): {gt_mean:.1f} BPM")
ax.set_xlabel("Window center (s)")
ax.set_ylabel("BPM")
ax.set_ylim(HR_MIN_BPM - 5, HR_MAX_BPM + 5)
ax.set_title("Per-window HR estimate (color = harmonic score)")
ax.legend()
ax.grid(alpha=0.3)
 
# 2. Eigenvector index distribution
ax = axes[1]
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


