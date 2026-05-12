import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from scipy import signal


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