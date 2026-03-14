import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt

# Load data
path_clean = r"C:\Projects\thesis\data\Sub01_Matrix_X_LowRank.csv"
df = pd.read_csv(path_clean)
frames = df['frame'].values
X = df.drop(columns=['frame']).values

# Eigendecomposition
B = X.T @ X
vals, vecs = np.linalg.eigh(B)
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

fs = 30.0
HR_MIN, HR_MAX = 45, 150
FREQ_MIN, FREQ_MAX = HR_MIN/60, HR_MAX/60

# Function to extract HR from signal
def extract_hr(signal, fs):
    """Extract heart rate from signal using FFT."""
    T = len(signal)
    xf = fftfreq(T, 1/fs)
    yf = np.abs(fft(signal))
    
    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    peak_idx = np.argmax(yf[mask])
    peak_freq = xf[mask][peak_idx]
    hr = peak_freq * 60
    power = yf[mask][peak_idx]
    
    return hr, power, xf, yf

# Function to measure cardiac band power
def cardiac_band_power(signal, fs):
    """Measure power in cardiac frequency band (0.75-2.5 Hz)."""
    # Bandpass filter for cardiac range
    sos = butter(4, [0.75, 2.5], btype='band', fs=fs, output='sos')
    filtered = sosfilt(sos, signal)
    
    # Calculate power (variance of filtered signal)
    power = np.var(filtered)
    return power

print("="*70)
print("TESTING DIFFERENT EIGENVECTOR COMBINATION STRATEGIES")
print("="*70)

# Strategy 1: Single best eigenvector (Paper's method)
print("\n1️⃣ STRATEGY 1: Single Eigenvector (v2)")
print("-" * 70)
p2 = X @ vecs[:, 1]
hr_single, power_single, xf_single, yf_single = extract_hr(p2, fs)
print(f"   HR: {hr_single:.2f} BPM")
print(f"   Power: {power_single:.0f}")

# Strategy 2: Combine v2 (fundamental) + v5 (2nd harmonic)
print("\n2️⃣ STRATEGY 2: Combine v2 + v5 (Fundamental + 2nd Harmonic)")
print("-" * 70)
p2_component = X @ vecs[:, 1]
p5_component = X @ vecs[:, 4]

# Test different weightings
best_combination = None
best_hr_diff = float('inf')

for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    beta = 1 - alpha
    p_combined = alpha * p2_component + beta * p5_component
    hr_combined, power_combined, _, _ = extract_hr(p_combined, fs)
    
    # Prefer combinations closer to expected range (55-75 BPM)
    hr_diff = abs(hr_combined - 65)
    
    if hr_diff < best_hr_diff and 50 < hr_combined < 150:
        best_hr_diff = hr_diff
        best_combination = {
            'alpha': alpha,
            'beta': beta,
            'signal': p_combined,
            'hr': hr_combined,
            'power': power_combined
        }

if best_combination:
    print(f"   Best weights: α={best_combination['alpha']:.1f} (v2), β={best_combination['beta']:.1f} (v5)")
    print(f"   HR: {best_combination['hr']:.2f} BPM")
    print(f"   Power: {best_combination['power']:.0f}")
    p_combined_best = best_combination['signal']
    hr_combined_best = best_combination['hr']
else:
    print("   ⚠️ No valid combination found")
    p_combined_best = p2
    hr_combined_best = hr_single

# Strategy 3: Weighted combination of all cardiac eigenvectors
print("\n3️⃣ STRATEGY 3: Weighted Combination (Power-based)")
print("-" * 70)

# Calculate cardiac band power for each eigenvector
cardiac_powers = []
for i in range(5):
    p = X @ vecs[:, i]
    power = cardiac_band_power(p, fs)
    cardiac_powers.append(power)
    print(f"   v{i+1} cardiac power: {power:.2f}")

# Normalize to get weights
total_power = sum(cardiac_powers)
weights = np.array(cardiac_powers) / total_power

print(f"\n   Normalized weights:")
for i in range(5):
    print(f"     v{i+1}: {weights[i]:.3f}")

# Create weighted combination
p_weighted = np.zeros(X.shape[0])
for i in range(5):
    p_weighted += weights[i] * (X @ vecs[:, i])

hr_weighted, power_weighted, xf_weighted, yf_weighted = extract_hr(p_weighted, fs)
print(f"\n   Combined HR: {hr_weighted:.2f} BPM")
print(f"   Power: {power_weighted:.0f}")

# Strategy 4: Only combine cardiac-range eigenvectors
print("\n4️⃣ STRATEGY 4: Selective Combination (Only Cardiac Range)")
print("-" * 70)

cardiac_indices = []
cardiac_weights_selective = []

for i in range(5):
    p = X @ vecs[:, i]
    hr, _, _, _ = extract_hr(p, fs)
    power = cardiac_band_power(p, fs)
    
    # Only include if HR is in cardiac range OR is a harmonic
    if (50 < hr < 150) or (hr > 100 and hr < 140):  # 2nd harmonic range
        cardiac_indices.append(i)
        cardiac_weights_selective.append(power)
        print(f"   Including v{i+1}: HR={hr:.1f} BPM, Power={power:.2f}")

if len(cardiac_indices) > 0:
    # Normalize weights
    total = sum(cardiac_weights_selective)
    cardiac_weights_selective = [w/total for w in cardiac_weights_selective]
    
    # Combine
    p_selective = np.zeros(X.shape[0])
    for i, weight in zip(cardiac_indices, cardiac_weights_selective):
        p_selective += weight * (X @ vecs[:, i])
    
    hr_selective, power_selective, xf_selective, yf_selective = extract_hr(p_selective, fs)
    print(f"\n   Combined HR: {hr_selective:.2f} BPM")
    print(f"   Power: {power_selective:.0f}")
else:
    print("   ⚠️ No cardiac eigenvectors found")
    p_selective = p2
    hr_selective = hr_single

# ============================================================
# COMPARISON AND VISUALIZATION
# ============================================================

print("\n" + "="*70)
print("COMPARISON OF ALL STRATEGIES")
print("="*70)

strategies = {
    'Single (v2)': {'hr': hr_single, 'signal': p2},
    'v2+v5 Combined': {'hr': hr_combined_best, 'signal': p_combined_best},
    'Power-Weighted': {'hr': hr_weighted, 'signal': p_weighted},
    'Selective': {'hr': hr_selective, 'signal': p_selective}
}

# Assuming Movesense ground truth is 66.21 BPM (from your previous sync)
movesense_hr = 66.21

print(f"\nGround Truth (Movesense): {movesense_hr:.2f} BPM\n")

for name, data in strategies.items():
    error = abs(data['hr'] - movesense_hr)
    error_pct = (error / movesense_hr) * 100
    print(f"{name:20s}: {data['hr']:6.2f} BPM | Error: {error:4.2f} BPM ({error_pct:4.2f}%)")

# Find best strategy
best_strategy = min(strategies.items(), key=lambda x: abs(x[1]['hr'] - movesense_hr))
print(f"\n🏆 BEST STRATEGY: {best_strategy[0]}")
print(f"   HR: {best_strategy[1]['hr']:.2f} BPM")
print(f"   Error: {abs(best_strategy[1]['hr'] - movesense_hr):.2f} BPM ({abs(best_strategy[1]['hr'] - movesense_hr)/movesense_hr*100:.2f}%)")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot time signals
time = frames[:600] / fs
for idx, (name, data) in enumerate(strategies.items()):
    row, col = idx // 2, idx % 2
    signal = data['signal'][:600]
    signal_norm = (signal - np.mean(signal)) / np.std(signal)
    
    axes[row, col].plot(time, signal_norm, linewidth=1.5, color='blue')
    axes[row, col].set_title(f"{name}\nHR: {data['hr']:.2f} BPM (Error: {abs(data['hr']-movesense_hr):.2f} BPM)", 
                            fontweight='bold')
    axes[row, col].set_xlabel('Time (s)')
    axes[row, col].set_ylabel('Normalized Amplitude')
    axes[row, col].grid(True, alpha=0.3)

# Frequency comparison plot
axes[2, 0].remove()
axes[2, 1].remove()
ax_freq = fig.add_subplot(3, 1, 3)

colors = ['blue', 'red', 'green', 'orange']
for (name, data), color in zip(strategies.items(), colors):
    _, _, xf, yf = extract_hr(data['signal'], fs)
    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    ax_freq.plot(xf[mask] * 60, yf[mask] / yf[mask].max(), 
                label=f"{name}: {data['hr']:.1f} BPM", 
                linewidth=2, alpha=0.7, color=color)

ax_freq.axvline(movesense_hr, color='black', linestyle='--', linewidth=2, 
               label=f'Ground Truth: {movesense_hr:.1f} BPM')
ax_freq.set_xlabel('Heart Rate (BPM)', fontweight='bold')
ax_freq.set_ylabel('Normalized Power', fontweight='bold')
ax_freq.set_title('Frequency Spectrum Comparison', fontweight='bold', fontsize=12)
ax_freq.legend(loc='upper right')
ax_freq.grid(True, alpha=0.3)
ax_freq.set_xlim(45, 150)

plt.tight_layout()
plt.savefig('eigenvector_combination_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Analysis saved: eigenvector_combination_comparison.png")

# Save results
results_df = pd.DataFrame([
    {'Strategy': name, 'HR_BPM': data['hr'], 
     'Error_BPM': abs(data['hr'] - movesense_hr),
     'Error_Percent': abs(data['hr'] - movesense_hr) / movesense_hr * 100}
    for name, data in strategies.items()
])
results_df.to_csv('eigenvector_strategies_comparison.csv', index=False)
print("✅ Results saved: eigenvector_strategies_comparison.csv")