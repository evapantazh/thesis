import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIG & PATHS ---
PULSE_DIR = Path(r"C:\Projects\thesis\data\PULSE_files_SQI")
IMAGE_DIR = Path(r"C:\Projects\thesis\thesis_latex\images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Στυλ (ίδιο με τα υπόλοιπα γραφήματά σου)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})

# --- 2. ΦΟΡΤΩΣΗ ΠΡΑΓΜΑΤΙΚΩΝ ΔΕΔΟΜΕΝΩΝ ---
hr_cam_list = []
hr_gt_list = []

# Διαβάζουμε όλα τα JSON αρχεία
for json_path in PULSE_DIR.glob("selection_windowed_*.json"):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Εξάγουμε τις δύο τιμές που χρειαζόμαστε
            hr_cam = data.get("bpm_smoothed")
            hr_gt = data.get("gt_bpm")
            
            if hr_cam is not None and hr_gt is not None:
                hr_cam_list.append(hr_cam)
                hr_gt_list.append(hr_gt)
    except Exception as e:
        print(f"Σφάλμα ανάγνωσης στο {json_path.name}: {e}")

# Μετατροπή σε numpy arrays για ευκολία στις πράξεις
hr_cam = np.array(hr_cam_list)
hr_gt = np.array(hr_gt_list)

print(f"Διαβάστηκαν επιτυχώς {len(hr_gt)} καταγραφές.")

# --- 3. ΥΠΟΛΟΓΙΣΜΟΙ BLAND-ALTMAN ---
# Στην κλινική πράξη, όταν το ένα μέσο είναι το "Απόλυτο Gold Standard" (Movesense),
# βάζουμε αυτό στον άξονα Χ, όχι τον μέσο όρο των δύο.
mean_hr = hr_gt  
diff = hr_cam - hr_gt  # Το σφάλμα (Camera - Ground Truth)

md = np.mean(diff)                   # Mean Bias
sd = np.std(diff, axis=0)            # Standard Deviation
loa_up = md + 1.96 * sd              # Upper Limit of Agreement
loa_down = md - 1.96 * sd            # Lower Limit of Agreement

# --- 4. ΣΧΕΔΙΑΣΜΟΣ ΓΡΑΦΗΜΑΤΟΣ ---
fig, ax = plt.subplots(figsize=(8, 5))

# Κουκκίδες (Μπλε χρώμα από τα υπόλοιπα plots σου)
ax.scatter(mean_hr, diff, alpha=0.75, color="#5b7c99", edgecolors='k', linewidth=0.5, s=45)

# Γραμμή του 0 (Τέλεια εκτίμηση)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, zorder=0)

# Γραμμή Mean Bias (Κόκκινη)
ax.axhline(md, color='#c44e52', linestyle='-', linewidth=2, label=f'Mean Bias: {md:.2f} BPM')

# Όρια Συμφωνίας (Limits of Agreement)
ax.axhline(loa_up, color='black', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {loa_up:.2f} BPM')
ax.axhline(loa_down, color='black', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {loa_down:.2f} BPM')

# Περιοχή 95% (Σκιασμένη περιοχή)
# Υπολογίζουμε λίγο "αέρα" δεξιά-αριστερά για τον άξονα Χ
x_margin = (max(mean_hr) - min(mean_hr)) * 0.05
ax.fill_between([min(mean_hr) - x_margin, max(mean_hr) + x_margin], 
                loa_down, loa_up, color='#5b7c99', alpha=0.1, zorder=0)

# Καλλωπισμός του γραφήματος
ax.set_xlabel("Movesense ECG Heart Rate (BPM)")
ax.set_ylabel("Error: Camera - ECG (BPM)")
ax.set_title("Bland-Altman Plot (All 76 Recordings)", pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

# Σφίγγουμε τα όρια του Χ για να μην υπάρχει περιττός κενός χώρος
ax.set_xlim(min(mean_hr) - x_margin, max(mean_hr) + x_margin)

plt.tight_layout()

# --- 5. ΑΠΟΘΗΚΕΥΣΗ ---
plot_path = IMAGE_DIR / 'bland_altman_aggregate.png'
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ Το Bland-Altman Plot αποθηκεύτηκε επιτυχώς στο:\n{plot_path}")