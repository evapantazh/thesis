import json
from pathlib import Path
import pandas as pd

PULSE_DIR = Path(r"C:\Projects\thesis\data\PULSE_files")
OUTPUT_FILE = Path(r"C:\Projects\thesis\final_results_summary.csv")

results = []

# Διαβάζουμε όλα τα json που έχουν αποτελέσματα
for json_file in PULSE_DIR.glob("selection_windowed_*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        # Υπολογίζουμε το σφάλμα μόνο αν υπάρχει Ground Truth
        gt = data.get("gt_bpm")
        bpm_top = data.get("bpm_top_half_median")
        error = abs(bpm_top - gt) if gt is not None else None
        
        results.append({
            "Recording": data["rec_id"],
            "BPM_Median": data["bpm_median"],
            "BPM_TopHalf": bpm_top,
            "GT": gt,
            "Error": round(error, 1) if error is not None else None
        })

# Δημιουργία DataFrame και ταξινόμηση
df = pd.DataFrame(results)
df = df.sort_values(by="Recording")

# Αποθήκευση σε CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"Ο πίνακας αποθηκεύτηκε στο: {OUTPUT_FILE}")
print(df.head())