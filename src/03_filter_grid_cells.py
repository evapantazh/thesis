import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
import json
import sys
from scipy.signal import detrend

# ADD DETRENDING LATER??????
# KEEP IT FOR NOW AS IS
# ADDED


# --- 1. SETTINGS & PATHS ---
# Update this to your exact path

BATCH_MODE = len(sys.argv) > 1

if BATCH_MODE:
    REC_ID = sys.argv[1]
else:
    # Default for manual runs
    SUBJECT = "GAX"
    DIST = "800"
    CLOTH = "tshirt"
    REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

GRID_DIR   = Path(r"C:\Projects\thesis\data\GRID_files")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH  = GRID_DIR   / f"GRID_{REC_ID}.csv"
OUTPUT_PATH = OUTPUT_DIR / f"FILTERED_DETRENDED_{REC_ID}.csv"

METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")

meta_file = METADATA_PATH/f"meta_{REC_ID}.json"

# --- LOAD TAP INFO ---
TAP_DIR = Path(r"C:\Projects\thesis\data\tap_info")
tap_file = TAP_DIR / "json"/ f"Tap_info_{REC_ID}.json"


SETTLE_SEC = 5.0  # seconds after tap to start usable signal

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

try:
    with open(tap_file, 'r') as f:
        tap_info = json.load(f)
    CAM_TAP_SEC = tap_info["manual_tap_sec"]
    print(f"Camera Chest tap at {CAM_TAP_SEC:.3f}s (camera time)")
except FileNotFoundError:
    CAM_TAP_SEC = None
    print("⚠️ No tap info found — cannot auto-trim")

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

    # Compute trim point from tap info
    time_axis_full = df['frame'].values  # frame numbers
    # We need time in seconds — use frame index / FS as approximation
    #time_sec = np.arange(len(df)) / FS

    #  |
    #  |
    #  v

    # We don't just assume every frame is exactly 1/15 seconds apart. There might be a frame drop for example. So

    # new — real timestamps
    ts_df_filter = pd.read_csv(Path(r"D:\recordings") / REC_ID / "timestamps.csv").set_index("frame")
    depth_col = "depth_timestamp" if "depth_timestamp" in ts_df_filter.columns else "timestamp"
    grid_frames = df["frame"].values
    timestamps = ts_df_filter.loc[grid_frames, depth_col].values
    time_sec = (timestamps - timestamps[0]) / 1000.0 


    if CAM_TAP_SEC is not None:
        trim_sec = CAM_TAP_SEC + SETTLE_SEC
        trim_idx = np.searchsorted(time_sec, trim_sec)
        print(f"Trimming first {trim_sec:.1f}s ({trim_idx} frames) — tap + settling")
    else:
        trim_idx = int(15.0 * FS)  # fallback: 15s
        trim_sec = 15.0
        print(f"Fallback trim: first {trim_sec:.1f}s ({trim_idx} frames)")
    
    # Trim the dataframe before filtering
    df_trimmed = df.iloc[trim_idx:].reset_index(drop=True)
    
    # --- CHECK ZEROS PER CELL ---
    print("\nZero counts per cell:")
    for i in range(28):
        zeros = (df_trimmed[f'cell_{i}'] == 0).sum()
        if zeros > 0:
            print(f"  cell_{i}: {zeros} zeros")
    print()

    # --- CHECK FRAME CONTINUITY ---
    frames = df_trimmed['frame'].values
    diffs = np.diff(frames)
    gaps = np.where(diffs > 1)[0]
    if len(gaps) > 0:
        print(f"⚠️ {len(gaps)} frame gaps found (non-contiguous):")
        for g in gaps[:10]:  # show first 10
            print(f"  frame {frames[g]} → {frames[g+1]} (skipped {diffs[g]-1})")
    else:
        print("✅ All frames contiguous — no gaps")

    filtered_df = pd.DataFrame()
    filtered_df['frame'] = df_trimmed['frame']

    print(f"🚀 Processing 28 cells from {INPUT_PATH} ({len(df_trimmed)} frames after trim)...")


    # Loop through all 28 cells
    for i in range(28):
        cell_name = f"cell_{i}"
        if cell_name in df_trimmed.columns:
            # Fix zero values
            raw_signal = df_trimmed[cell_name].values.copy()

            # Skip cell if mostly invalid
            if (raw_signal > 0).sum() < 10:
                filtered_df[cell_name] = 0.0
                continue

            # Replace zeros with cell mean to avoid filter artifacts
            #valid_mean = raw_signal[raw_signal > 0].mean()
            #raw_signal[raw_signal == 0] = valid_mean

            # Interpolate over zero values to avoid filter artifacts
            mask = raw_signal == 0
            if mask.any():
                indices = np.arange(len(raw_signal))
                good = ~mask
                if good.sum() >= 2:
                    raw_signal[mask] = np.interp(indices[mask], indices[good], raw_signal[good])
                else:
                    filtered_df[cell_name] = 0.0
                    continue


            # Clip sudden jumps (steps larger than N*std of diff)
            diffs = np.diff(raw_signal, prepend=raw_signal[0])
            jump_threshold = 3 * np.std(diffs)
            jump_mask = np.abs(diffs) > jump_threshold
            if jump_mask.any():
                indices = np.arange(len(raw_signal))
                good = ~jump_mask
                raw_signal[jump_mask] = np.interp(indices[jump_mask], indices[good], raw_signal[good])

            # Then detrend
            raw_signal = detrend(raw_signal, type='linear')    

            # Apply Filter
            filt_signal = butter_bandpass_filter(raw_signal, LOWCUT, HIGHCUT, FS, order=ORDER)
            
            # Mean-Centering (Final step for Matrix D)
            centered_signal = filt_signal - np.mean(filt_signal)
            
            filtered_df[cell_name] = centered_signal

    # Save Output
    filtered_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Filtered matrix saved to: {OUTPUT_PATH}")

    # --- VISUAL CHECK ---
    import matplotlib.pyplot as plt

    # Pick a cell to inspect (try a few: 0, 10, 14, 20)
    CELL_TO_PLOT = 14

    cell_name = f"cell_{CELL_TO_PLOT}"
    raw_signal = df_trimmed[f"cell_{CELL_TO_PLOT}"].values.copy()

    # Reproduce zero handling (same as your current code)
    #valid_mean = raw_signal[raw_signal > 0].mean()
    #raw_signal[raw_signal == 0] = valid_mean

    # Reproduce zero handling (interpolation)
    mask = raw_signal == 0
    if mask.any():
        indices = np.arange(len(raw_signal))
        good = ~mask
        raw_signal[mask] = np.interp(indices[mask], indices[good], raw_signal[good])

    # ADD: match the jump clipping from the main loop
    diffs = np.diff(raw_signal, prepend=raw_signal[0])
    jump_threshold = 3 * np.std(diffs)
    jump_mask = np.abs(diffs) > jump_threshold
    if jump_mask.any():
        indices = np.arange(len(raw_signal))
        good = ~jump_mask
        raw_signal[jump_mask] = np.interp(indices[jump_mask], indices[good], raw_signal[good])

    # ADD: match the detrend from the main loop
    raw_signal = detrend(raw_signal, type='linear')


    time_axis = time_sec[trim_idx:]  # np.arange(len(raw_signal)) / FS + trim_sec  # seconds

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{REC_ID} — {cell_name}", fontsize=14)

    # 1. Raw signal (after zero fill)
    axes[0].plot(time_axis, raw_signal, linewidth=0.6)
    axes[0].set_ylabel("Depth (mm)")
    axes[0].set_title("Raw signal interpolated + jump clipped + detrended")

    # 2. Filtered signal
    filt_signal = filtered_df[cell_name].values
    axes[1].plot(time_axis, filt_signal, linewidth=0.6, color="tab:orange")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Bandpass filtered [{LOWCUT}–{HIGHCUT} Hz]")

    # 3. Overlay: raw detrended vs filtered (to see what the filter kept)
    detrended = raw_signal - np.mean(raw_signal)
    axes[2].plot(time_axis, detrended, linewidth=0.5, alpha=0.5, label="Raw (preprocessed and mean-removed)")
    axes[2].plot(time_axis, filt_signal, linewidth=0.7, label="Filtered")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Overlay comparison")
    axes[2].legend()

    plt.tight_layout()

    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"filter_check_detrended_{REC_ID}_{cell_name}.png"
    plt.savefig(plot_path, dpi=150)
    if not BATCH_MODE:
        plt.show()
    print(f"📊 Plot saved to: {plot_path}")