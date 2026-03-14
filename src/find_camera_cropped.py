import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION (From your Sync Results)
# ==========================================
TAP_TIME_CAMERA = 1.333  # <--- YOUR EXACT RESULT
START_OFFSET = 5.0       # Seconds to skip after tap
DURATION = 60.0          # Seconds to analyze
FS = 30.0                # Camera sampling rate

# 1. Load the CLEAN Low-Rank matrix
path_clean = r"C:\Projects\thesis\data\Sub01_Matrix_X_LowRank.csv"
df = pd.read_csv(path_clean)
frames_all = df['frame'].values
X_all = df.drop(columns=['frame']).values

print(f"Original Matrix: {X_all.shape} (Includes Tap Artifact)")

# ==========================================
# STEP 1.5: SLICE THE WINDOW (Crucial Step)
# ==========================================
print("\n" + "="*60)
print(f"STEP 1.5: Slicing Window ({START_OFFSET}s after tap)")
print("="*60)

# Calculate frame indices
# We start exactly 5 seconds after the tap to ensure body stability
start_time = TAP_TIME_CAMERA + START_OFFSET
end_time = start_time + DURATION

start_frame_idx = int(start_time * FS)
end_frame_idx = int(end_time * FS)

# Safety check: Ensure we don't go past the end of the file
if end_frame_idx > len(X_all):
    print(f"⚠️ Warning: Requested end frame {end_frame_idx} is beyond data length {len(X_all)}.")
    end_frame_idx = len(X_all)

# OVERWRITE X with the clean slice
X = X_all[start_frame_idx:end_frame_idx, :]
frames = frames_all[start_frame_idx:end_frame_idx]

print(f"Analysis Window: {start_time:.2f}s to {end_time:.2f}s")
print(f"Sliced Matrix X: {X.shape} (Clean Data)")

# 2. Eigen-decomposition (Same as before, but on Clean X)
print("\n" + "="*60)
print("STEP 2: Eigenvalue Decomposition (B = X^T * X)")
print("="*60)

B = X.T @ X 
vals, vecs = np.linalg.eigh(B)

# Sort by eigenvalue (largest first)
idx = vals.argsort()[::-1]
vals_sorted = vals[idx]
vecs_sorted = vecs[:, idx]

# Show variance
total_variance = vals_sorted.sum()
print("\nVariance explained by top 5 eigenvectors:")
for i in range(5):
    variance_pct = (vals_sorted[i] / total_variance) * 100
    print(f"  v{i+1}: {variance_pct:.2f}%")

# 3. Test top 5 eigenvectors (Same logic)
print("\n" + "="*60)
print("STEP 3: Testing Top 5 Eigenvectors")
print("="*60)

HR_MIN, HR_MAX = 45, 150
FREQ_MIN, FREQ_MAX = HR_MIN/60, HR_MAX/60

candidates = []

for i in range(5):
    p = X @ vecs_sorted[:, i]
    
    # FFT
    T = len(p)
    xf = fftfreq(T, 1/FS)
    yf = np.abs(fft(p))
    
    mask = (xf >= FREQ_MIN) & (xf <= FREQ_MAX)
    xf_hr = xf[mask]
    yf_hr = yf[mask]
    
    # Peak finding
    peaks, _ = find_peaks(yf_hr, prominence=yf_hr.max()*0.1)
    
    if len(peaks) > 0:
        fundamental_idx = peaks[np.argmax(yf_hr[peaks])]
    else:
        fundamental_idx = np.argmax(yf_hr)
    
    fundamental_freq = xf_hr[fundamental_idx]
    fundamental_hr = fundamental_freq * 60
    fundamental_power = yf_hr[fundamental_idx]
    
    # Harmonic Scoring
    harmonic_score = fundamental_power
    
    # Check 2nd Harmonic
    h2 = fundamental_freq * 2
    mask2 = (xf >= h2 - 0.1) & (xf <= h2 + 0.1)
    if mask2.any(): harmonic_score += np.max(yf[mask2]) * 0.5
    
    # Check 3rd Harmonic
    h3 = fundamental_freq * 3
    mask3 = (xf >= h3 - 0.1) & (xf <= h3 + 0.1)
    if mask3.any(): harmonic_score += np.max(yf[mask3]) * 0.3
    
    candidates.append({
        'index': i+1,
        'hr': fundamental_hr,
        'score': harmonic_score,
        'signal': p,
        'xf': xf_hr,
        'yf': yf_hr
    })
    
    print(f"v{i+1}: {fundamental_hr:.2f} BPM (Score: {harmonic_score:.0f})")

# 4. Select Best
print("\n" + "="*60)
print("STEP 4: Selection")
print("="*60)

# Filter for physiological range
valid = [c for c in candidates if 50 < c['hr'] < 150]

if not valid:
    print("⚠️ No valid candidates found. Using v2.")
    best = candidates[1]
else:
    best = max(valid, key=lambda x: x['score'])

print(f"✅ BEST CHOICE: v{best['index']} ({best['hr']:.2f} BPM)")

# 5. Visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Candidates
for c in candidates:
    alpha = 1.0 if c == best else 0.3
    axes[0].plot(c['xf']*60, c['yf'], label=f"v{c['index']}", alpha=alpha)

axes[0].set_title("Frequency Spectra of Top 5 Eigenvectors")
axes[0].set_xlabel("BPM")
axes[0].set_xlim(40, 160)
axes[0].legend()

# Plot Best Signal (Time Domain)
time_axis = np.arange(len(best['signal'])) / FS
axes[1].plot(time_axis, best['signal'], 'r', label=f"v{best['index']} (Best)")
axes[1].set_title(f"Selected Pulse Signal (Start: {start_time:.1f}s, End: {end_time:.1f}s)")
axes[1].set_xlabel("Time (s within window)")
axes[1].legend()

plt.tight_layout()
plt.show()