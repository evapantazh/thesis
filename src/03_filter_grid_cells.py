import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

# --- 1. SETTINGS & PATHS ---
# Update this to your exact path
INPUT_PATH = r"C:\Projects\thesis\data\GRID_Sub01_500_Tshirt.csv"
OUTPUT_PATH = r"C:\Projects\thesis\data\FILTERED_Sub01_500_Tshirt.csv"

FS = 30.0        # Sampling Rate
LOWCUT = 0.5     # 30 BPM
HIGHCUT = 4.0    # 240 BPM
ORDER = 4        # Butterworth Order

# --- 2. FILTER DEFINITION ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- 3. EXECUTION ---
if not os.path.exists(INPUT_PATH):
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
            # Apply Filter
            raw_signal = df[cell_name].values
            filt_signal = butter_bandpass_filter(raw_signal, LOWCUT, HIGHCUT, FS, order=ORDER)
            
            # Mean-Centering (Final step for Matrix D)
            centered_signal = filt_signal - np.mean(filt_signal)
            
            filtered_df[cell_name] = centered_signal

    # Save Output
    filtered_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Filtered matrix saved to: {OUTPUT_PATH}")