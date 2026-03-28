import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# --- 1. CONFIGURATION & PATHS ---
BASE = Path(r"C:\Projects\thesis\data")
SUBJECT, DIST, CLOTH = "Sub01", "500", "Tshirt"
REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

PATH_FILTERED_CAM = BASE / f"FILTERED_{REC_ID}.csv"
PATH_MOVESENSE_ECG = BASE / f"Movesense_{REC_ID}" / f"{REC_ID}_ecg_stream.json"
PATH_TAP_INFO = BASE / "tap_info" / f"Tap_info_{REC_ID}.json"

# --- 2. LOAD DATA ---
# Load Synchronization Info
with open(PATH_TAP_INFO, 'r') as f:
    tap_info = json.load(f)
dt = tap_info['offset_sec']

# Load Filtered Camera Data
df_cam = pd.read_csv(PATH_FILTERED_CAM)
t_cam = df_cam['frame'].values / 30.0  # Camera time axis
t_cam_aligned = t_cam + dt             # Align camera to Movesense clock

# --- 2. LOAD DATA (FIXED FOR YOUR JSON) ---
with open(PATH_MOVESENSE_ECG, 'r') as f:
    raw_json = json.load(f)
    ecg_data = raw_json['data']

ecg_samples = []
ecg_ts = []
start_ts = ecg_data[0]['ecg']['Timestamp']

for entry in ecg_data:
    if 'ecg' in entry and 'Samples' in entry['ecg']:
        ts = entry['ecg']['Timestamp']
        samples = entry['ecg']['Samples']
        
        for i, s in enumerate(samples):
            ecg_samples.append(s)
            # 125Hz = 1 sample κάθε 8ms
            ecg_ts.append((ts - start_ts + (i * 8.0)) / 1000.0)

print(f"✅ Loaded {len(ecg_samples)} ECG samples.")

# --- 3. PLOT ALIGNED SIGNALS ---
plt.figure(figsize=(15, 8))

# Zoom window: 10 to 20 seconds after start
# Adjusted to see the heartbeats clearly
T_START, T_END = 15.0, 25.0 

# Plot ECG (Ground Truth)
ax1 = plt.subplot(2, 1, 1)
mask_ecg = (np.array(ecg_ts) >= T_START) & (np.array(ecg_ts) <= T_END)
plt.plot(np.array(ecg_ts)[mask_ecg], np.array(ecg_samples)[mask_ecg], color='black', label="Movesense ECG")
plt.title(f"Aligned ECG vs Camera Pulse ({REC_ID})")
plt.ylabel("uV")
plt.legend()
plt.grid(alpha=0.3)

# Plot Camera (Aligned)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
mask_cam = (t_cam_aligned >= T_START) & (t_cam_aligned <= T_END)
# Using cell_14 as your main pulse signal
plt.plot(t_cam_aligned[mask_cam], df_cam['cell_14'].values[mask_cam], color='red', label="Camera BCG (Aligned)")
plt.ylabel("Displacement (mm)")
plt.xlabel("Time (s) - Movesense Clock")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()