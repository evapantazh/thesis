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

def estimate_fhr_hz_via_2nd_harmonic(pc, fs, fmin_fund=0.9, fmax_fund=2.5):
    """
    Βρίσκει τον παλμό ψάχνοντας τη 2η αρμονική (για να αποφύγει την αναπνοή)
    και στο τέλος διαιρεί δια 2.
    """
    pc = np.asarray(pc, float)
    pc = signal.detrend(pc)
    pc = pc * np.hanning(len(pc))
    
    F = np.abs(rfft(pc))
    f = rfftfreq(len(pc), d=1/fs)
    
    # 1. Ορίζουμε το ψάξιμο στη 2Η ΑΡΜΟΝΙΚΗ! 
    # (Αν η καρδιά είναι 55-150 BPM, η 2η αρμονική της είναι 110-300 BPM)
    fmin_2nd = fmin_fund * 2.0  # π.χ. 1.8 Hz
    fmax_2nd = fmax_fund * 2.0  # π.χ. 5.0 Hz
    
    band = (f >= fmin_2nd) & (f <= fmax_2nd)
    
    if not np.any(band):
        raise ValueError("No FFT bins in HR band.")
        
    # 2. Βρίσκουμε το peak σε αυτές τις ΥΨΗΛΕΣ συχνότητες
    f_peak_2nd = f[band][np.argmax(F[band])]
    
    # 3. Ο ΠΡΑΓΜΑΤΙΚΟΣ παλμός είναι το μισό αυτού που βρήκαμε!
    f_hr = f_peak_2nd / 2.0
    
    return float(f_hr), f, F