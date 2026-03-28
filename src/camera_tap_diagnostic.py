import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  SETTINGS (Άλλαξέ τα ανάλογα με το τεστ σου)
# ─────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Projects\thesis\recordings") # Ή C:\Projects\thesis\data
SUBJECT  = "Sub01"
DIST     = "800"
CLOTH    = "Hoodie"
REC_ID   = "grid2_test" # Το όνομα του CSV που έβγαλες πριν

PATH_CSV = BASE / f"{REC_ID}.csv"
FS_CAM   = 30.0

# Παράμετροι ανίχνευσης
SEARCH_WINDOW_SEC = 5.0  # Ψάξε το tap στα πρώτα 5 δευτερόλεπτα
PROMINENCE_SIGMA  = 2.0  # Πόσο "ψηλό" πρέπει να είναι το peak

def build_camera_proxy(X):
    """
    Υπολογίζει το σήμα του tap: Ενέργεια x Συνοχή.
    Υψηλή τιμή σημαίνει ότι όλα τα κελιά κουνήθηκαν μαζί απότομα.
    """
    X = np.asarray(X, float).copy()
    # Fill zeros
    for j in range(X.shape[1]):
        bad = (X[:, j] == 0)
        if np.any(bad):
            X[bad, j] = np.median(X[~bad, j]) if np.any(~bad) else 0

    dX = np.gradient(X, axis=0)
    energy = np.sum(np.abs(dX), axis=1)    # Συνολική κίνηση
    coherence = np.abs(np.mean(dX, axis=1)) # Συνοχή κατεύθυνσης
    return energy * coherence

# ─────────────────────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────────────────────
print(f"--- Diagnostic for: {REC_ID} ---")

df = pd.read_csv(PATH_CSV)
frames = df["frame"].values
X = df.drop(columns=["frame"]).values
t = (frames - frames[0]) / FS_CAM

proxy = build_camera_proxy(X)

# Εντοπισμός Peak
mask = t <= SEARCH_WINDOW_SEC
x_search = proxy[mask]
threshold = np.std(x_search) * PROMINENCE_SIGMA
peaks, _ = signal.find_peaks(x_search, prominence=threshold, distance=10)

# ─────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(t, proxy, color='steelblue', label='Camera Tap Proxy')

if len(peaks) > 0:
    tap_idx = peaks[0]
    plt.axvline(t[tap_idx], color='red', linestyle='--', label=f'Detected Tap: {t[tap_idx]:.3f}s')
    plt.scatter(t[peaks], proxy[peaks], color='red')
    print(f"SUCCESS: Tap detected at frame {frames[tap_idx]} ({t[tap_idx]:.3f}s)")
else:
    print("FAILED: No tap detected. Try hitting harder or lowering PROMINENCE_SIGMA.")

plt.title(f"Tap Diagnostic: {SUBJECT} | {CLOTH} | {DIST}mm")
plt.xlabel("Time (seconds)")
plt.ylabel("Proxy Strength (Energy * Coherence)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()