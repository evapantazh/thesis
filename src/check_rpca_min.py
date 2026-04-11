import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from pathlib import Path
import json

# --- 1. SETTINGS & PATHS ---
SUBJECT = "AVE"
DIST = "800"
CLOTH = "tshirt"
REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

RPCA_DIR = Path(r"C:\Projects\thesis\data\RPCA_files")
METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\PULSE_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = RPCA_DIR / f"LOWRANK_{REC_ID}.csv"

# Load FPS from metadata
meta_file = METADATA_PATH / f"meta_{REC_ID}.json"
try:
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    FS = meta.get("actual_fps", 15.0)
except FileNotFoundError:
    FS = 15.0

# HR search range
HR_MIN_BPM = 40
HR_MAX_BPM = 150
FREQ_MIN = HR_MIN_BPM / 60.0
FREQ_MAX = HR_MAX_BPM / 60.0

# How many eigenvectors to test
N_CANDIDATES = 5


# --- 2. HARMONIC SCORING ---
def harmonic_score(freq_axis, power, fund_freq, n_harmonics=3, tol=0.1):
    """
    Score a candidate fundamental frequency by summing power
    at the fundamental + 2nd and 3rd harmonics.
    """
    score = 0.0
    weights = [1.0, 0.5, 0.3]  # fundamental, 2nd, 3rd harmonic

    for h, w in zip(range(1, n_harmonics + 1), weights):
        target = fund_freq * h
        mask = (freq_axis >= target - tol) & (freq_axis <= target + tol)
        if mask.any():
            score += np.max(power[mask]) * w

    return score


# --- 3. EXECUTION ---
if not INPUT_PATH.exists():
    print(f"❌ Error: {INPUT_PATH} not found!")
else:
    df = pd.read_csv(INPUT_PATH)
    frames = df['frame'].values
    X = df.drop(columns=['frame']).values

    print(f"Low-rank matrix X: {X.shape} (frames × cells)")
    print(f"FS = {FS:.2f} Hz")

    # Mean-center X (paper: "mean-centered spatial temporal matrix X_dot")
    X_centered = X - X.mean(axis=0)

    # Eigendecomposition: B = X^T X
    B = X_centered.T @ X_centered
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    total_var = eigenvalues.sum()
    print(f"\nVariance explained by top {N_CANDIDATES} eigenvectors:")
    for i in range(N_CANDIDATES):
        pct = (eigenvalues[i] / total_var) * 100
        print(f"  v{i+1}: {pct:.1f}%")

    # Project onto each eigenvector and score
    candidates = []
    for i in range(N_CANDIDATES):
        # Projection (equation 5 from paper)
        p = X_centered @ eigenvectors[:, i]

        # Welch PSD for frequency analysis
        freqs, psd = welch(p, fs=FS, nperseg=min(256, len(p)), noverlap=128)

        # Find peak in cardiac range
        hr_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
        freqs_hr = freqs[hr_mask]
        psd_hr = psd[hr_mask]

        if len(psd_hr) == 0:
            continue

        peak_idx = np.argmax(psd_hr)
        fund_freq = freqs_hr[peak_idx]
        fund_bpm = fund_freq * 60.0

        # Score using harmonics
        score = harmonic_score(freqs, psd, fund_freq)

        candidates.append({
            'index': i,
            'label': f'v{i+1}',
            'bpm': fund_bpm,
            'fund_freq': fund_freq,
            'score': score,
            'signal': p,
            'freqs': freqs,
            'psd': psd,
            'variance_pct': (eigenvalues[i] / total_var) * 100
        })

        print(f"  v{i+1}: {fund_bpm:.1f} BPM, score={score:.2f}, var={candidates[-1]['variance_pct']:.1f}%")

    # Select best candidate
    valid = [c for c in candidates if HR_MIN_BPM < c['bpm'] < HR_MAX_BPM]
    if not valid:
        print("⚠️ No valid candidates found — using highest score overall")
        best = max(candidates, key=lambda c: c['score'])
    else:
        best = max(valid, key=lambda c: c['score'])

    print(f"\n✅ SELECTED: {best['label']} — {best['bpm']:.1f} BPM (score={best['score']:.2f})")

    # Save pulse signal
    pulse_df = pd.DataFrame({
        'frame': frames,
        'pulse': best['signal']
    })
    pulse_path = OUTPUT_DIR / f"PULSE_{REC_ID}.csv"
    pulse_df.to_csv(pulse_path, index=False)
    print(f"💓 Pulse signal saved: {pulse_path}")

    # Save selection metadata
    selection_info = {
        'rec_id': REC_ID,
        'selected_eigenvector': best['label'],
        'estimated_bpm': round(best['bpm'], 1),
        'harmonic_score': round(best['score'], 4),
        'variance_explained_pct': round(best['variance_pct'], 1),
        'gamma_used': 0.02,
        'fs': FS,
        'all_candidates': [
            {'label': c['label'], 'bpm': round(c['bpm'], 1),
             'score': round(c['score'], 4), 'var_pct': round(c['variance_pct'], 1)}
            for c in candidates
        ]
    }
    info_path = OUTPUT_DIR / f"selection_{REC_ID}.json"
    with open(info_path, 'w') as f:
        json.dump(selection_info, f, indent=2)

    # --- 4. VISUAL CHECK ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Eigenvector Selection — {REC_ID}", fontsize=14)

    # Panel 1: All candidate spectra
    for c in candidates:
        alpha = 1.0 if c == best else 0.3
        lw = 1.5 if c == best else 0.8
        axes[0].plot(c['freqs'] * 60, c['psd'], label=f"{c['label']} ({c['bpm']:.0f} BPM)",
                     alpha=alpha, linewidth=lw)
    axes[0].set_xlim(HR_MIN_BPM, HR_MAX_BPM)
    axes[0].set_xlabel("BPM")
    axes[0].set_ylabel("PSD")
    axes[0].set_title("Welch PSD of top 5 eigenvector projections")
    axes[0].legend(fontsize=9)

    # Panel 2: Selected pulse signal (time domain)
    time_axis = np.arange(len(best['signal'])) / FS
    axes[1].plot(time_axis, best['signal'], linewidth=0.6, color="tab:red")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Selected pulse signal — {best['label']} ({best['bpm']:.1f} BPM)")
    axes[1].set_xlabel("Time (s)")

    # Panel 3: Eigenvalue spectrum (scree plot)
    axes[2].bar(range(1, len(eigenvalues) + 1), eigenvalues / total_var * 100, color="steelblue")
    axes[2].set_xlim(0.5, min(15, len(eigenvalues)) + 0.5)
    axes[2].set_xlabel("Eigenvector index")
    axes[2].set_ylabel("Variance explained (%)")
    axes[2].set_title("Eigenvalue spectrum (scree plot)")

    plt.tight_layout()
    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"eigenvector_selection_{REC_ID}.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"📊 Plot saved: {plot_path}")

    # --- 5. COMPARE WITH GROUND TRUTH ---
    MOVESENSE_DIR = Path(r"C:\Projects\thesis\data\movesense") / REC_ID
    hr_file = MOVESENSE_DIR / "heartRate_stream.json"

    if hr_file.exists():
        with open(hr_file, 'r') as f:
            hr_data = json.load(f)

        # Extract all average HR values
        if 'data' in hr_data:
            hr_values = [entry['heartRate']['average'] for entry in hr_data['data']]
        else:
            hr_values = [hr_data['heartRate']['average']]

        hr_array = np.array(hr_values)
        gt_mean = np.mean(hr_array)
        gt_std = np.std(hr_array)

        error = abs(best['bpm'] - gt_mean)

        print(f"\n--- Ground Truth Comparison ---")
        print(f"Camera estimate:  {best['bpm']:.1f} BPM")
        print(f"Movesense mean:   {gt_mean:.1f} ± {gt_std:.1f} BPM")
        print(f"Error:            {error:.1f} BPM")
    else:
        print(f"\n⚠️ No heartRate.json found at {hr_file}")