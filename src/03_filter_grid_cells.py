import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
import json

# --- 1. SETTINGS & PATHS ---
# Update this to your exact path
REC_ID = "andreas_800_tshirt"


GRID_DIR   = Path(r"C:\Projects\thesis\data\GRID_files")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH  = GRID_DIR   / f"GRID_{REC_ID}.csv"
OUTPUT_PATH = OUTPUT_DIR / f"FILTERED_{REC_ID}.csv"

METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")

meta_file = METADATA_PATH/f"meta_{REC_ID}.json"

# Expanded temporal filtering
LOWCUT = 0.5     # 30 BPM
HIGHCUT = 4.0    # 240 BPM
ORDER = 4        # Butterworth Order
# Sampling frequency FPS from timestamps

try:
    with open(meta_file, 'r') as file:
        meta = json.load(file)
    FS = meta.get("actual_fps", 15.0)  # the get() function never crushes because if it doesnt find the actual fps fiels, then it uses the fallback 15.0
    print("FPS of recording found through timestamps in metada json file =", FS)
    
except FileNotFoundError:
    FS = 15.0 
    print("Error: The file metadata was not found. Using default FPS = 15.0")


# --- 2. FILTER DEFINITION ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- 3. EXECUTION ---
if not INPUT_PATH.exists():
    print(f"❌ Error: Could not find {INPUT_PATH}. Check your folder path!")
else:
    # Load Data
    df = pd.read_csv(INPUT_PATH)
    filtered_df = pd.DataFrame()
    filtered_df['frame'] = df['frame']

    print(f"🚀 Processing 28 cells from {INPUT_PATH}...")

    # Loop through all 28 cells
    for i in range(28):
        cell_name = f"cell_{i}"
        if cell_name in df.columns:
            # Fix zero values
            raw_signal = df[cell_name].values.copy()

            # Skip cell if mostly invalid
            if (raw_signal > 0).sum() < 10:
                filtered_df[cell_name] = 0.0
                continue

            # Replace zeros with cell mean to avoid filter artifacts
            valid_mean = raw_signal[raw_signal > 0].mean()
            raw_signal[raw_signal == 0] = valid_mean


            # Apply Filter
            filt_signal = butter_bandpass_filter(raw_signal, LOWCUT, HIGHCUT, FS, order=ORDER)
            
            # Mean-Centering (Final step for Matrix D)
            centered_signal = filt_signal - np.mean(filt_signal)
            
            filtered_df[cell_name] = centered_signal

    # Save Output
    filtered_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Filtered matrix saved to: {OUTPUT_PATH}")