import numpy as np
import pandas as pd
import os
from scipy.fft import fft, ifft, fftfreq

# ============================================================
# IALM RPCA ENGINE
# ============================================================

def fit_ialm(D, lambda_=None, epsilon1=1e-7, epsilon2=1e-5, mu=None, rho=1.6, max_iter=1000, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier method for Robust PCA
    Decomposes D = A + E where:
    - A: Low-rank matrix (cardiac signal)
    - E: Sparse matrix (noise/outliers)
    """
    D = np.asarray(D)
    m, n = D.shape
    
    if lambda_ is None:
        lambda_ = 1 / np.sqrt(m)

    D = np.float64(D)
    Y = np.copy(D)
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm

    A = np.zeros_like(D)
    E = np.zeros_like(D)
    d_norm = np.linalg.norm(D, 'fro')
    tol_proj = epsilon2 * d_norm

    if mu is None:
        mu = 1.25 / norm_two

    iter_ = 0
    converged = False

    while not converged:
        iter_ += 1

        # Step 1: Soft-thresholding (Sparse component E)
        temp_T = D - A + (1 / mu) * Y
        temp_E = (np.maximum(temp_T - lambda_ / mu, 0) + np.minimum(temp_T + lambda_ / mu, 0))

        # Step 2: Singular Value Thresholding (Low-rank component A)
        U, S, V = np.linalg.svd(D - temp_E + (1 / mu) * Y, full_matrices=False)
        svp = np.count_nonzero(S > 1 / mu)
        A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])

        # Step 3: Update Lagrange Multiplier
        Z = D - A - temp_E
        Y += mu * Z
        
        if mu * np.linalg.norm(E - temp_E, 'fro') < tol_proj:
            mu *= rho
        
        E = temp_E
        stop_criterion = np.linalg.norm(Z, 'fro') / d_norm
        
        if verbose:
            print(f"Iter: {iter_} | Rank(A): {np.linalg.matrix_rank(A)} | Stop: {stop_criterion:.2e}")

        if stop_criterion < epsilon1:
            converged = True

        if not converged and iter_ >= max_iter:
            print('Maximum iterations reached.')
            break

    return A, E

# ============================================================
# ADAPTIVE SPECTRAL FILTERING (Equation 6 & 7 from paper)
# ============================================================

def estimate_hr_from_signal(signal, fs, hr_range=(45, 150)):
    """
    Estimate heart rate from signal using FFT
    
    Args:
        signal: Time-domain signal (1D array)
        fs: Sampling frequency (Hz)
        hr_range: Valid HR range in BPM (tuple)
    
    Returns:
        estimated_hr_hz: Estimated HR in Hz
        estimated_hr_bpm: Estimated HR in BPM
    """
    # FFT
    N = len(signal)
    freq = fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(signal))
    
    # Focus on positive frequencies in cardiac range
    hr_min_hz = hr_range[0] / 60.0
    hr_max_hz = hr_range[1] / 60.0
    
    mask = (freq >= hr_min_hz) & (freq <= hr_max_hz)
    freq_cardiac = freq[mask]
    fft_cardiac = fft_vals[mask]
    
    # Find peak (fundamental frequency)
    if len(fft_cardiac) > 0:
        peak_idx = np.argmax(fft_cardiac)
        estimated_hr_hz = freq_cardiac[peak_idx]
        estimated_hr_bpm = estimated_hr_hz * 60.0
    else:
        # Fallback
        estimated_hr_hz = 1.0  # 60 BPM
        estimated_hr_bpm = 60.0
    
    return estimated_hr_hz, estimated_hr_bpm

def adaptive_spectral_filter(signal, fs, omega=0.25, hr_range=(45, 150), verbose=True):
    """
    Adaptive Spectral Filter as described in the paper (Equation 6 & 7)
    
    Keeps only:
    - Fundamental HR frequency ± ω/2
    - 2nd harmonic (2×HR) ± ω/2
    - 3rd harmonic (3×HR) ± ω/2
    
    Args:
        signal: Time-domain pulse signal pc(t) after RPCA
        fs: Sampling frequency (Hz)
        omega: Sub-band window size (Hz), default 0.25 Hz = 15 BPM
        hr_range: Valid HR range in BPM
        verbose: Print debug info
    
    Returns:
        pd: Denoised pulse signal (time-domain)
        Y: Frequency-domain filter mask
        estimated_hr_bpm: Estimated HR in BPM
    """
    N = len(signal)
    
    # Step 1: Estimate HR from signal
    f_hr, hr_bpm = estimate_hr_from_signal(signal, fs, hr_range)
    
    if verbose:
        print(f"\n[Adaptive Spectral Filter]")
        print(f"  Estimated HR: {hr_bpm:.2f} BPM ({f_hr:.3f} Hz)")
        print(f"  Window size ω: {omega:.3f} Hz ({omega*60:.1f} BPM)")
    
    # Step 2: Compute FFT
    freq = fftfreq(N, 1/fs)
    pc_f = fft(signal)  # pc(f) in the paper
    
    # Step 3: Create filter mask Υ(f) - Equation 6
    Y = np.zeros(N, dtype=float)
    
    # Band 1: Fundamental frequency (f_HR ± ω/2)
    mask1 = (np.abs(freq - f_hr) <= omega/2) | (np.abs(freq + f_hr) <= omega/2)
    
    # Band 2: 2nd harmonic (2×f_HR ± ω/2)
    mask2 = (np.abs(freq - 2*f_hr) <= omega/2) | (np.abs(freq + 2*f_hr) <= omega/2)
    
    # Band 3: 3rd harmonic (3×f_HR ± ω/2)
    mask3 = (np.abs(freq - 3*f_hr) <= omega/2) | (np.abs(freq + 3*f_hr) <= omega/2)
    
    # Combine masks (Υ(f) = 1 in these bands, 0 elsewhere)
    Y[mask1 | mask2 | mask3] = 1.0
    
    if verbose:
        print(f"  Fundamental band: {f_hr - omega/2:.3f} - {f_hr + omega/2:.3f} Hz")
        print(f"  2nd harmonic band: {2*f_hr - omega/2:.3f} - {2*f_hr + omega/2:.3f} Hz")
        print(f"  3rd harmonic band: {3*f_hr - omega/2:.3f} - {3*f_hr + omega/2:.3f} Hz")
        print(f"  Kept frequencies: {np.sum(Y > 0)} / {N} bins")
    
    # Step 4: Apply filter in frequency domain (Equation 7)
    # pd(t) = iFFT[pc(f) ⊙ Υ(f)]
    pd_f = pc_f * Y  # Element-wise multiplication
    pd_t = np.real(ifft(pd_f))  # Inverse FFT
    
    return pd_t, Y, hr_bpm

def apply_spectral_filtering_to_matrix(X_rpca, fs, omega=0.25, hr_range=(45, 150)):
    """
    Apply adaptive spectral filtering to each column (cell) of RPCA output
    
    Args:
        X_rpca: Low-rank matrix after RPCA (time × cells)
        fs: Sampling frequency
        omega: Sub-band window size
        hr_range: Valid HR range
    
    Returns:
        X_filtered: Spectrally filtered matrix
        hr_estimates: Estimated HR for each cell
    """
    n_frames, n_cells = X_rpca.shape
    X_filtered = np.zeros_like(X_rpca)
    hr_estimates = np.zeros(n_cells)
    
    print("\n" + "="*70)
    print("ADAPTIVE SPECTRAL FILTERING")
    print("="*70)
    print(f"Processing {n_cells} cells...")
    
    for i in range(n_cells):
        signal = X_rpca[:, i]
        
        # Apply filter
        signal_filtered, filter_mask, hr_est = adaptive_spectral_filter(
            signal, fs, omega=omega, hr_range=hr_range, verbose=(i == 0)  # Verbose for first cell
        )
        
        X_filtered[:, i] = signal_filtered
        hr_estimates[i] = hr_est
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{n_cells} cells", end='\r')
    
    print(f"\n✅ Spectral filtering complete!")
    print(f"  HR estimates: {hr_estimates.min():.1f} - {hr_estimates.max():.1f} BPM")
    print(f"  Mean HR: {hr_estimates.mean():.1f} BPM")
    print(f"  Std HR:  {hr_estimates.std():.1f} BPM")
    
    return X_filtered, hr_estimates


from scipy.fft import rfft, rfftfreq, irfft
from scipy import signal
import numpy as np

def estimate_fhr_hz(pc, fs, fmin=0.75, fmax=2.5):
    """
    Estimate fundamental HR frequency (Hz) by FFT peak search in [fmin, fmax].
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
    Υ(f)=1 in [k*fHR - ω/2, k*fHR + ω/2] for k=1,2,3 else 0
    pd(t) = iFFT( pc(f) * Υ(f) )
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
    pd_t = irfft(Pc * mask, n=len(pc))
    return pd_t, f, mask












# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("="*70)
    print("RPCA + ADAPTIVE SPECTRAL FILTERING PIPELINE")
    print("="*70)
    
    # Configuration
    INPUT_FILE = r"C:\Projects\thesis\data\csv\Sub01_Matrix_D_Filtered.csv"
    OUTPUT_RPCA = r"C:\Projects\thesis\data\csv\Sub01_Matrix_X_LowRank.csv"
    OUTPUT_FILTERED = r"C:\Projects\thesis\data\csv\Sub01_Matrix_X_Filtered.csv"
    
    FS = 30.0  # Sampling rate (Hz)
    OMEGA = 0.25  # Sub-band window (Hz) - 15 BPM as per paper
    HR_RANGE = (45, 150)  # Valid HR range (BPM)
    
    # ============================================================
    # STEP 1: Load filtered matrix D
    # ============================================================
    print("\n[Step 1] Loading filtered matrix D...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        return
    
    df = pd.read_csv(INPUT_FILE)
    frames = df['frame'].values
    D_matrix = df.drop(columns=['frame']).values
    
    print(f"✅ Matrix loaded: {D_matrix.shape} (frames × cells)")
    
    # ============================================================
    # STEP 2: RPCA decomposition
    # ============================================================
    print("\n[Step 2] Running RPCA...")
    print("This separates cardiac signal (low-rank A) from noise (sparse E)")
    
    A_lowrank, E_sparse = fit_ialm(D_matrix, verbose=True)
    
    print(f"\n✅ RPCA complete!")
    print(f"  Low-rank matrix A (cardiac): {A_lowrank.shape}")
    print(f"  Sparse matrix E (noise): {E_sparse.shape}")
    print(f"  Matrix rank: {np.linalg.matrix_rank(A_lowrank)}")
    
    # Save RPCA result
    df_lowrank = pd.DataFrame(A_lowrank, columns=df.columns[1:])
    df_lowrank.insert(0, 'frame', frames)
    df_lowrank.to_csv(OUTPUT_RPCA, index=False)
    print(f"  Saved: {OUTPUT_RPCA}")
    
    # ============================================================
    # STEP 3: Adaptive Spectral Filtering
    # ============================================================
    print("\n[Step 3] Applying Adaptive Spectral Filtering...")
    print("This keeps only HR fundamental + 2nd/3rd harmonics")
    
    X_filtered, hr_estimates = apply_spectral_filtering_to_matrix(
        A_lowrank, FS, omega=OMEGA, hr_range=HR_RANGE
    )
    
    # ============================================================
    # STEP 4: Save filtered result
    # ============================================================
    print("\n[Step 4] Saving results...")
    
    df_filtered = pd.DataFrame(X_filtered, columns=df.columns[1:])
    df_filtered.insert(0, 'frame', frames)
    df_filtered.to_csv(OUTPUT_FILTERED, index=False)
    
    print(f"✅ Saved: {OUTPUT_FILTERED}")
    
    # ============================================================
    # STEP 5: Summary comparison
    # ============================================================
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    print("\nSignal Quality Comparison:")
    print(f"  Original D variance:     {np.var(D_matrix):.4f}")
    print(f"  After RPCA A variance:   {np.var(A_lowrank):.4f}")
    print(f"  After Filtering variance: {np.var(X_filtered):.4f}")
    print(f"  Noise E variance:        {np.var(E_sparse):.4f}")
    
    print("\nPer-cell HR estimates (from spectral filtering):")
    for i in range(min(10, len(hr_estimates))):
        print(f"  Cell {i}: {hr_estimates[i]:.2f} BPM")
    if len(hr_estimates) > 10:
        print(f"  ... and {len(hr_estimates) - 10} more cells")
    
    print("\n✅ Pipeline complete! Ready for eigenvector extraction.")
    print("\nNext steps:")
    print("  1. Use Sub01_Matrix_X_Filtered.csv for PCA")
    print("  2. Extract eigenvector p2 (second component)")
    print("  3. Compute final HR estimate")
    print("="*70)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()