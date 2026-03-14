import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# 1. Load the CLEAN Low-Rank matrix
path_clean = r"C:\Projects\thesis\data\Sub01_Matrix_X_LowRank.csv"
df = pd.read_csv(path_clean)
frames = df['frame'].values
X = df.drop(columns=['frame']).values

print(f"Loaded matrix X: {X.shape}")
print(f"  Frames: {X.shape[0]}")
print(f"  Cells: {X.shape[1]}")

# 2. Eigen-decomposition (Following paper's equation 4)
print("\n" + "="*60)
print("STEP 2: Eigenvalue Decomposition (B = X^T × X)")
print("="*60)

B = X.T @ X  # Spatial-temporal covariance matrix
vals, vecs = np.linalg.eigh(B)

# Sort by eigenvalue (largest first)
idx = vals.argsort()[::-1]
vals_sorted = vals[idx]
vecs_sorted = vecs[:, idx]

# Show variance explained by each eigenvector
total_variance = vals_sorted.sum()
print("\nVariance explained by top 5 eigenvectors:")
for i in range(5):
    variance_pct = (vals_sorted[i] / total_variance) * 100
    print(f"  v{i+1} (λ{i+1}): {variance_pct:.2f}%")

# 3. Test all top 5 eigenvectors (Following paper's method)
print("\n" + "="*60)
print("STEP 3: Testing Top 5 Eigenvectors for Cardiac Signal")
print("="*60)

fs = 30.0  # Sampling rate
HR_MIN, HR_MAX = 45, 150  # BPM range
FREQ_MIN, FREQ_MAX = HR_MIN/60, HR_MAX/60  # Hz range

candidates = []

for i in range(5):
    # Extract pulse signal (Equation 5 from paper)
    p = X @ vecs_sorted[:, i]
    
    # FFT analysis
    T = len(p)
    xf = fftfreq(T, 1/fs)
    yf = np.abs(fft(p))
    
    # Focus on HR range
    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    xf_hr = xf[mask]
    yf_hr = yf[mask]
    
    # Find fundamental frequency peak
    peaks, properties = find_peaks(yf_hr, prominence=yf_hr.max()*0.1)
    
    if len(peaks) == 0:
        # No clear peak, use max
        fundamental_idx = np.argmax(yf_hr)
    else:
        # Use highest peak
        fundamental_idx = peaks[np.argmax(yf_hr[peaks])]
    
    fundamental_freq = xf_hr[fundamental_idx]
    fundamental_hr = fundamental_freq * 60
    fundamental_power = yf_hr[fundamental_idx]
    
    # Check for harmonics (2nd and 3rd order)
    harmonic_2_freq = fundamental_freq * 2
    harmonic_3_freq = fundamental_freq * 3
    
    # Score based on fundamental strength + harmonics presence
    harmonic_score = fundamental_power
    
    # Check if 2nd harmonic exists
    harmonic_2_mask = (xf >= harmonic_2_freq - 0.1) & (xf <= harmonic_2_freq + 0.1)
    if harmonic_2_mask.any():
        harmonic_2_power = np.max(yf[harmonic_2_mask])
        harmonic_score += harmonic_2_power * 0.5
    
    # Check if 3rd harmonic exists
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
        'xf': xf_hr,
        'yf': yf_hr
    })
    
    # Classify signal type
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

# 4. Select best eigenvector - FILTER BY PHYSIOLOGICAL RANGE FIRST
print("\n" + "="*60)
print("STEP 4: Selecting Best Eigenvector (with physiological filtering)")
print("="*60)

# Filter candidates to cardiac range (exclude breathing)
valid_candidates = [c for c in candidates if 50 < c['hr'] < 150]

print(f"\nValid cardiac candidates (50-150 BPM): {len(valid_candidates)}")
for c in valid_candidates:
    print(f"  v{c['index']}: {c['hr']:.2f} BPM (score: {c['harmonic_score']:.0f})")

if len(valid_candidates) == 0:
    print("\n⚠️ WARNING: No valid cardiac candidates found!")
    print("   Falling back to p2 (second eigenvector)")
    best_candidate = candidates[1]
else:
    # Among valid candidates, pick strongest harmonic score
    best_candidate = max(valid_candidates, key=lambda x: x['harmonic_score'])
best_idx = best_candidate['index']
best_hr = best_candidate['hr']
best_signal = best_candidate['signal']

print(f"\n✅ BEST CHOICE: v{best_idx} (p{best_idx})")
print(f"   Heart Rate: {best_hr:.2f} BPM")
print(f"   Harmonic Score: {best_candidate['harmonic_score']:.0f}")

# 5. Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Top 5 pulse signals
axes[0].set_title('Top 5 Pulse Signals (First 10 seconds)', fontsize=12, fontweight='bold')
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, candidate in enumerate(candidates):
    time = frames[:300] / fs
    signal_norm = (candidate['signal'][:300] - np.mean(candidate['signal'])) / np.std(candidate['signal'])
    
    alpha = 1.0 if candidate['index'] == best_idx else 0.4
    linewidth = 2 if candidate['index'] == best_idx else 1
    label = f"p{candidate['index']} ({candidate['hr']:.1f} BPM)" + (" ✓ BEST" if candidate['index'] == best_idx else "")
    
    axes[0].plot(time, signal_norm + i*3, color=colors[i], alpha=alpha, 
                linewidth=linewidth, label=label)

axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Normalized Signal (offset for clarity)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Frequency spectra of all candidates
axes[1].set_title('Frequency Spectra Comparison', fontsize=12, fontweight='bold')
for i, candidate in enumerate(candidates):
    alpha = 1.0 if candidate['index'] == best_idx else 0.3
    linewidth = 2 if candidate['index'] == best_idx else 1
    label = f"p{candidate['index']}" + (" ✓" if candidate['index'] == best_idx else "")
    
    axes[1].plot(candidate['xf'] * 60, candidate['yf'] / candidate['yf'].max(), 
                color=colors[i], alpha=alpha, linewidth=linewidth, label=label)

axes[1].axvline(best_hr, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Selected: {best_hr:.1f} BPM')
axes[1].set_xlabel('Heart Rate (BPM)')
axes[1].set_ylabel('Normalized Power')
axes[1].set_xlim(45, 150)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Best signal with detailed view
axes[2].set_title(f'Selected Cardiac Signal: p{best_idx} ({best_hr:.2f} BPM)', fontsize=12, fontweight='bold')
time_full = frames / fs
axes[2].plot(time_full[:600], best_signal[:600], color='red', linewidth=1.5)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eigenvector_selection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Final Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Selected Eigenvector: v{best_idx}")
print(f"Heart Rate: {best_hr:.2f} BPM")
print(f"\nComparison with your original method (always p2):")
print(f"  Your method: {candidates[1]['hr']:.2f} BPM (using p2)")
print(f"  Paper method: {best_hr:.2f} BPM (using p{best_idx})")
print(f"  Difference: {abs(candidates[1]['hr'] - best_hr):.2f} BPM")

# Save results
results = {
    'method': ['Original (p2)', 'Paper Method (Best)'],
    'eigenvector': [2, best_idx],
    'heart_rate_bpm': [candidates[1]['hr'], best_hr],
    'harmonic_score': [candidates[1]['harmonic_score'], best_candidate['harmonic_score']]
}

df_results = pd.DataFrame(results)
df_results.to_csv('eigenvector_selection_comparison.csv', index=False)

print("\n✅ Analysis saved:")
print("  - eigenvector_selection_analysis.png")
print("  - eigenvector_selection_comparison.csv")