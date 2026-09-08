#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# --- 1. CONFIG & PATHS ---
MOVESENSE_DIR = Path(r"C:\Projects\thesis\data\movesense")
OUT_PLOTDIR  = Path(r"C:\Projects\thesis\thesis_latex\images")
OUT_PLOTDIR.mkdir(parents=True, exist_ok=True)

ECG_FS = 125.0

# Τα 3 subjects που επιλέξαμε για να δείξουμε το εύρος των παλμών (Low, Medium, High HR)
RECORDINGS = [
    ("GVA_1200_tshirt", "Subject: GVA (~61 BPM)"),
    ("MPA_1200_tshirt", "Subject: MPA (~75 BPM)"),
    ("EST_1200_tshirt", "Subject: EST (~96 BPM)")
]

# Θα δείξουμε 10 δευτερόλεπτα για να φαίνεται καθαρά η διαφορά στην πυκνότητα των R-peaks
ZOOM = (10.0, 20.0) 

# --- 2. ΣΤΥΛ & ΧΡΩΜΑΤΑ (Academic) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_ECG   = "#2e2e2e"   # Σκούρο γκρι/μαύρο - ECG trace
C_RPEAK = "#a01818"   # Σκούρο κόκκινο - R-peaks


# --- 3. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ (Από τον κώδικά σου) ---
def load_ecg(rec_id):
    ecg_file = MOVESENSE_DIR / rec_id / "ecg_stream.json"
    if not ecg_file.exists():
        print(f"❌ Δεν βρέθηκε: {ecg_file}")
        return None
    d = json.load(open(ecg_file))["data"]
    samples, ts = [], []
    for p in d:
        e = p["ecg"]
        samples.append(np.asarray(e["Samples"], dtype=float))
        ts.append(e["Timestamp"])
    sig = np.concatenate(samples)

    ts = np.asarray(ts, dtype=float)
    spp = len(d[0]["ecg"]["Samples"])
    if len(ts) > 1:
        dt = np.median(np.diff(ts)) / 1000.0
        fs = spp / dt if dt > 0 else ECG_FS
    else:
        fs = ECG_FS
    t = np.arange(len(sig)) / fs
    return t, sig, fs


# --- 4. ΣΧΕΔΙΑΣΜΟΣ (Stacked Plot) ---
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for ax, (rec_id, label) in zip(axes, RECORDINGS):
    ecg_data = load_ecg(rec_id)
    if ecg_data is None:
        continue
    
    t, ecg, fs = ecg_data
    
    # Απομόνωση του παραθύρου ZOOM
    m = (t >= ZOOM[0]) & (t <= ZOOM[1])
    tp, sp = t[m], ecg[m]
    
    # Εύρεση R-peaks (όπως ακριβώς το είχες)
    s_full = ecg - np.median(ecg)
    height = np.percentile(s_full, 98) * 0.5
    peaks_full, _ = find_peaks(s_full, height=height, distance=int(0.4 * fs))
    
    # Κρατάμε μόνο τα peaks που πέφτουν μέσα στο ZOOM
    pk = peaks_full[(t[peaks_full] >= ZOOM[0]) & (t[peaks_full] <= ZOOM[1])]
    
    # Σχεδίαση
    ax.plot(tp, sp, color=C_ECG, lw=0.8)
    # Βάζουμε label μόνο στο πρώτο plot για να μη γεμίσει το υπόμνημα
    peak_label = "Detected R-peak" if ax == axes[0] else None 
    ax.plot(t[pk], ecg[pk], "v", color=C_RPEAK, markersize=5,
            markeredgecolor="black", markeredgewidth=0.3, label=peak_label)
    
    # Καλλωπισμός κάθε υπο-γραφήματος
    ax.set_ylabel("ECG (a.u.)")
    ax.set_title(label, loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Αφαίρεση του κάτω άξονα στα πάνω γραφήματα για να "κολλήσουν"
    if ax != axes[-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    
    if ax == axes[0]:
        ax.legend(frameon=False, loc="upper right")

# Ο κάτω-κάτω άξονας κρατάει τον χρόνο
axes[-1].set_xlabel("Time (s)")

fig.subplots_adjust(hspace=0.25)
fig.tight_layout()

# Αποθήκευση
plot_path = OUT_PLOTDIR / "ecg_hr_range_stacked.png"
fig.savefig(plot_path, dpi=300)
fig.savefig(plot_path.with_suffix(".pdf"))
plt.show()

print(f"\n✅ Το Stacked ECG plot αποθηκεύτηκε στο:\n{plot_path}")