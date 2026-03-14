import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

print("="*70)
print("FINAL HR EXTRACTION FROM SPECTRALLY FILTERED MATRIX")
print("="*70)

# ============================================================
# LOAD FILTERED MATRIX
# ============================================================

print("\n[Step 1] Loading spectrally filtered matrix...")

# Load BOTH matrices for comparison
df_lowrank = pd.read_csv(r"C:\Projects\thesis\data\csv\Sub01_Matrix_X_LowRank.csv")
df_filtered = pd.read_csv(r"C:\Projects\thesis\data\csv\Sub01_Matrix_X_Filtered.csv")

frames = df_filtered['frame'].values
X_lowrank = df_lowrank.drop(columns=['frame']).values
X_filtered = df_filtered.drop(columns=['frame']).values

fs = 30.0  # Hz

print(f"✅ Matrix loaded: {X_filtered.shape}")
print(f"   Frames: {len(frames)}")
print(f"   Cells: {X_filtered.shape[1]}")
print(f"   Duration: {len(frames) / fs:.2f}s")

# ============================================================
# EIGENVECTOR EXTRACTION (PCA)
# ============================================================

print("\n" + "="*70)
print("[Step 2] Eigenvector Extraction (PCA)")
print("="*70)

def extract_eigenvector(X, idx=1):
    """Extract eigenvector from matrix"""
    B = X.T @ X
    vals, vecs = np.linalg.eigh(B)
    vecs_sorted = vecs[:, vals.argsort()[::-1]]
    p = X @ vecs_sorted[:, idx]
    
    # Variance explained
    total_var = vals.sum()
    var_explained = vals[vals.argsort()[::-1]]
    
    return p, var_explained, total_var

# Extract p2 from BOTH matrices
print("\nA) RPCA Only (baseline):")
p2_lowrank, var_lowrank, total_var_lowrank = extract_eigenvector(X_lowrank, idx=1)
print(f"   p2 extracted (second eigenvector)")
print(f"   Variance explained by top 5:")
for i in range(5):
    pct = (var_lowrank[i] / total_var_lowrank) * 100
    print(f"     v{i+1}: {pct:.2f}%")

print("\nB) RPCA + Spectral Filter:")
p2_filtered, var_filtered, total_var_filtered = extract_eigenvector(X_filtered, idx=1)
print(f"   p2 extracted (second eigenvector)")
print(f"   Variance explained by top 5:")
for i in range(5):
    pct = (var_filtered[i] / total_var_filtered) * 100
    print(f"     v{i+1}: {pct:.2f}%")

# ============================================================
# HR ESTIMATION VIA FFT
# ============================================================

print("\n" + "="*70)
print("[Step 3] Heart Rate Estimation (FFT)")
print("="*70)

def estimate_hr_fft(signal, fs, hr_range=(45, 150)):
    """Estimate HR from signal using FFT"""
    # FFT
    N = len(signal)
    freqs = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(signal))
    
    # Focus on cardiac range
    hr_min_hz = hr_range[0] / 60.0
    hr_max_hz = hr_range[1] / 60.0
    
    mask = (freqs >= hr_min_hz) & (freqs <= hr_max_hz)
    freqs_cardiac = freqs[mask]
    fft_cardiac = fft_vals[mask]
    
    # Find peak
    if len(fft_cardiac) > 0:
        peak_idx = np.argmax(fft_cardiac)
        hr_hz = freqs_cardiac[peak_idx]
        hr_bpm = hr_hz * 60.0
        
        # Find 2nd and 3rd harmonics for validation
        harmonic_2_mask = (freqs >= 1.9*hr_hz) & (freqs <= 2.1*hr_hz)
        harmonic_3_mask = (freqs >= 2.9*hr_hz) & (freqs <= 3.1*hr_hz)
        
        harmonic_2_power = np.max(fft_vals[harmonic_2_mask]) if harmonic_2_mask.any() else 0
        harmonic_3_power = np.max(fft_vals[harmonic_3_mask]) if harmonic_3_mask.any() else 0
        
        return hr_bpm, hr_hz, freqs_cardiac, fft_cardiac, harmonic_2_power, harmonic_3_power
    else:
        return 60.0, 1.0, freqs_cardiac, fft_cardiac, 0, 0

# Estimate HR from both signals
print("\nA) RPCA Only:")
hr_lowrank, hz_lowrank, f_lr, fft_lr, h2_lr, h3_lr = estimate_hr_fft(p2_lowrank, fs)
print(f"   Heart Rate: {hr_lowrank:.2f} BPM ({hz_lowrank:.3f} Hz)")
print(f"   2nd harmonic power: {h2_lr:.1f}")
print(f"   3rd harmonic power: {h3_lr:.1f}")

print("\nB) RPCA + Spectral Filter:")
hr_filtered, hz_filtered, f_filt, fft_filt, h2_filt, h3_filt = estimate_hr_fft(p2_filtered, fs)
print(f"   Heart Rate: {hr_filtered:.2f} BPM ({hz_filtered:.3f} Hz)")
print(f"   2nd harmonic power: {h2_filt:.1f}")
print(f"   3rd harmonic power: {h3_filt:.1f}")

# ============================================================
# COMPARISON WITH GROUND TRUTH
# ============================================================

print("\n" + "="*70)
print("[Step 4] Comparison with Ground Truth")
print("="*70)

movesense_hr = 62.45  # From your synchronized comparison

print(f"\nGround Truth (Movesense): {movesense_hr:.2f} BPM")
print(f"\nCamera Estimates:")
print(f"  RPCA Only:              {hr_lowrank:.2f} BPM")
print(f"  RPCA + Spectral Filter: {hr_filtered:.2f} BPM")

error_lowrank = abs(hr_lowrank - movesense_hr)
error_filtered = abs(hr_filtered - movesense_hr)

print(f"\nAbsolute Errors:")
print(f"  RPCA Only:              {error_lowrank:.2f} BPM")
print(f"  RPCA + Spectral Filter: {error_filtered:.2f} BPM")

improvement = error_lowrank - error_filtered
print(f"\nImprovement: {improvement:+.2f} BPM", end="")
if improvement > 0:
    print(f" ✅ (Spectral filter helped!)")
elif improvement < -0.5:
    print(f" ⚠️  (Spectral filter made it worse)")
else:
    print(f" (Minimal change)")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "="*70)
print("[Step 5] Generating Visualizations")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Time domain signals
t = frames / fs

# Left: RPCA only
axes[0, 0].plot(t[:600], p2_lowrank[:600], 'b-', linewidth=1)
axes[0, 0].set_title('A) RPCA Only - Time Domain (First 20s)', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(alpha=0.3)

# Right: RPCA + Filter
axes[0, 1].plot(t[:600], p2_filtered[:600], 'r-', linewidth=1)
axes[0, 1].set_title('B) RPCA + Spectral Filter - Time Domain (First 20s)', fontweight='bold')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(alpha=0.3)

# Plot 2: Frequency domain (full spectrum)
# Left: RPCA only
axes[1, 0].plot(f_lr * 60, fft_lr / np.max(fft_lr), 'b-', linewidth=1.5)
axes[1, 0].axvline(hr_lowrank, color='r', linestyle='--', linewidth=2, 
                   label=f'{hr_lowrank:.2f} BPM')
axes[1, 0].axvline(movesense_hr, color='g', linestyle='--', linewidth=2, 
                   label=f'Ground Truth: {movesense_hr:.2f} BPM')
axes[1, 0].set_title('A) RPCA Only - Frequency Spectrum', fontweight='bold')
axes[1, 0].set_xlabel('Heart Rate (BPM)')
axes[1, 0].set_ylabel('Normalized Power')
axes[1, 0].set_xlim(45, 150)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Right: RPCA + Filter
axes[1, 1].plot(f_filt * 60, fft_filt / np.max(fft_filt), 'r-', linewidth=1.5)
axes[1, 1].axvline(hr_filtered, color='r', linestyle='--', linewidth=2, 
                   label=f'{hr_filtered:.2f} BPM')
axes[1, 1].axvline(movesense_hr, color='g', linestyle='--', linewidth=2, 
                   label=f'Ground Truth: {movesense_hr:.2f} BPM')
axes[1, 1].set_title('B) RPCA + Spectral Filter - Frequency Spectrum', fontweight='bold')
axes[1, 1].set_xlabel('Heart Rate (BPM)')
axes[1, 1].set_ylabel('Normalized Power')
axes[1, 1].set_xlim(45, 150)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 3: Harmonics analysis
# Left: RPCA only
freqs_full_lr = fftfreq(len(p2_lowrank), 1/fs)
fft_full_lr = np.abs(fft(p2_lowrank))
axes[2, 0].plot(freqs_full_lr[:len(freqs_full_lr)//2] * 60, 
                fft_full_lr[:len(fft_full_lr)//2] / np.max(fft_full_lr), 'b-', linewidth=1)
axes[2, 0].axvline(hr_lowrank, color='r', linestyle='--', alpha=0.7, label='Fundamental')
axes[2, 0].axvline(2*hr_lowrank, color='orange', linestyle='--', alpha=0.7, label='2nd Harmonic')
axes[2, 0].axvline(3*hr_lowrank, color='purple', linestyle='--', alpha=0.7, label='3rd Harmonic')
axes[2, 0].set_title('A) RPCA Only - Harmonics', fontweight='bold')
axes[2, 0].set_xlabel('Frequency (BPM)')
axes[2, 0].set_ylabel('Normalized Power')
axes[2, 0].set_xlim(0, 250)
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(alpha=0.3)

# Right: RPCA + Filter
freqs_full_filt = fftfreq(len(p2_filtered), 1/fs)
fft_full_filt = np.abs(fft(p2_filtered))
axes[2, 1].plot(freqs_full_filt[:len(freqs_full_filt)//2] * 60, 
                fft_full_filt[:len(fft_full_filt)//2] / np.max(fft_full_filt), 'r-', linewidth=1)
axes[2, 1].axvline(hr_filtered, color='r', linestyle='--', alpha=0.7, label='Fundamental')
axes[2, 1].axvline(2*hr_filtered, color='orange', linestyle='--', alpha=0.7, label='2nd Harmonic')
axes[2, 1].axvline(3*hr_filtered, color='purple', linestyle='--', alpha=0.7, label='3rd Harmonic')
axes[2, 1].set_title('B) RPCA + Spectral Filter - Harmonics (Cleaned!)', fontweight='bold')
axes[2, 1].set_xlabel('Frequency (BPM)')
axes[2, 1].set_ylabel('Normalized Power')
axes[2, 1].set_xlim(0, 250)
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rpca_vs_spectral_filtering_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: rpca_vs_spectral_filtering_comparison.png")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "="*70)
print("[Step 6] Saving Results")
print("="*70)

results = {
    'Method': [
        'RPCA Only',
        'RPCA + Spectral Filter',
        'Ground Truth (Movesense)'
    ],
    'Heart Rate (BPM)': [
        hr_lowrank,
        hr_filtered,
        movesense_hr
    ],
    'Absolute Error (BPM)': [
        error_lowrank,
        error_filtered,
        0.0
    ],
    'Relative Error (%)': [
        (error_lowrank / movesense_hr) * 100,
        (error_filtered / movesense_hr) * 100,
        0.0
    ]
}

df_results = pd.DataFrame(results)
df_results.to_csv('final_hr_comparison.csv', index=False)
print("✅ Saved: final_hr_comparison.csv")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\n📊 Results:")
print(f"   Ground Truth:           {movesense_hr:.2f} BPM")
print(f"   RPCA Only:              {hr_lowrank:.2f} BPM (error: {error_lowrank:.2f} BPM)")
print(f"   RPCA + Spectral Filter: {hr_filtered:.2f} BPM (error: {error_filtered:.2f} BPM)")

if improvement > 0:
    print(f"\n✅ Spectral filtering IMPROVED accuracy by {improvement:.2f} BPM!")
elif improvement < -0.5:
    print(f"\n⚠️  Spectral filtering made it slightly worse by {-improvement:.2f} BPM")
    print("   This can happen if the initial HR estimate was off")
else:
    print(f"\n➡️  Spectral filtering had minimal impact ({improvement:+.2f} BPM)")
    print("   Signal was already clean after RPCA")

print("\n✅ Analysis complete!")
print("="*70)

plt.show()