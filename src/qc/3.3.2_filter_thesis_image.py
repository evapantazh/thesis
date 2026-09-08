import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend
import json
import matplotlib.pyplot as plt

# --- 1. THESIS STYLING BLOCK ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# --- 2. CONFIGURATION & PATHS ---
REC_ID = "GVA_1800_tshirt"
CELL_TO_PLOT = 14
cell_name = f"cell_{CELL_TO_PLOT}"

# Adjust these base paths if your drive mapping differs
GRID_DIR      = Path(r"C:\Projects\thesis\data\GRID_files")
METADATA_PATH = Path(r"C:\Projects\thesis\data\metadata")
TAP_DIR       = Path(r"C:\Projects\thesis\data\tap_info")
OUTPUT_DIR    = Path(r"C:\Projects\thesis\thesis_latex\images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = GRID_DIR / f"GRID_{REC_ID}.csv"
meta_file  = METADATA_PATH / f"meta_{REC_ID}.json"
tap_file   = TAP_DIR / "json" / f"Tap_info_{REC_ID}.json"

SETTLE_SEC = 5.0
LOWCUT = 0.5     # 30 BPM
HIGHCUT = 4.0    # 240 BPM
ORDER = 4        

# --- 3. LOAD METADATA & PARSE FS ---
try:
    with open(meta_file, 'r') as file:
        meta = json.load(file)
    FS = meta.get("actual_fps", 15.0)
except FileNotFoundError:
    FS = 15.0 

try:
    with open(tap_file, 'r') as f:
        tap_info = json.load(f)
    CAM_TAP_SEC = tap_info["manual_tap_sec"]
except FileNotFoundError:
    CAM_TAP_SEC = None

# --- 4. SIGNAL PROCESSING ---
if not INPUT_PATH.exists():
    print(f"❌ Error: Could not find {INPUT_PATH}.")
else:
    df = pd.read_csv(INPUT_PATH)
    
    # Resolve exact time axis via depth timestamps
    ts_path = Path(r"D:\recordings") / REC_ID / "timestamps.csv"
    if not ts_path.exists():
        ts_path = Path(r"C:\Projects\thesis\data\timestamps_backup") / REC_ID / "timestamps.csv"
    
    ts_df_filter = pd.read_csv(ts_path).set_index("frame")
    depth_col = "depth_timestamp" if "depth_timestamp" in ts_df_filter.columns else "timestamp"
    grid_frames = df["frame"].values
    timestamps = ts_df_filter.loc[grid_frames, depth_col].values
    time_sec = (timestamps - timestamps[0]) / 1000.0 

    # Calculate Trim Index
    if CAM_TAP_SEC is not None:
        trim_sec = CAM_TAP_SEC + SETTLE_SEC
        trim_idx = np.searchsorted(time_sec, trim_sec)
    else:
        trim_idx = int(15.0 * FS)

    df_trimmed = df.iloc[trim_idx:].reset_index(drop=True)
    time_axis = time_sec[trim_idx:]

    # Isolate targeting cell array
    raw_signal = df_trimmed[cell_name].values.copy()

    # Handle missing raw details via linear interpolation
    mask = raw_signal == 0
    if mask.any():
        indices = np.arange(len(raw_signal))
        good = ~mask
        raw_signal[mask] = np.interp(indices[mask], indices[good], raw_signal[good])

    # Step-Jump removal step
    diffs = np.diff(raw_signal, prepend=raw_signal[0])
    jump_threshold = 3 * np.std(diffs)
    jump_mask = np.abs(diffs) > jump_threshold
    if jump_mask.any():
        indices = np.arange(len(raw_signal))
        good = ~jump_mask
        raw_signal[jump_mask] = np.interp(indices[jump_mask], indices[good], raw_signal[good])

    # De-trending before filtering phase
    raw_signal = detrend(raw_signal, type='linear')

    # Butterworth implementation
    nyq = 0.5 * FS
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(ORDER, [low, high], btype='band')
    filt_signal = filtfilt(b, a, raw_signal)
    centered_filt_signal = filt_signal - np.mean(filt_signal)

    # --- 5. GENERATE CLEAN PLOT ---
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)

    # Subplot A: Preprocessed Signal
    axes[0].plot(time_axis, raw_signal, linewidth=0.6, color="#2b5c8f")
    axes[0].set_ylabel("Depth (mm)")
    axes[0].set_title("Raw chest displacement (interpolated, jump-clipped, and detrended)")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Subplot B: Bandpass Trace
    axes[1].plot(time_axis, centered_filt_signal, linewidth=0.6, color="#e07a5f")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Bandpass filtered signal [{LOWCUT}–{HIGHCUT} Hz]")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Subplot C: Concise Legend Overlay
    mean_removed_raw = raw_signal - np.mean(raw_signal)
    axes[2].plot(time_axis, mean_removed_raw, linewidth=0.5, color="#2b5c8f", alpha=0.4, label="Raw")
    axes[2].plot(time_axis, centered_filt_signal, linewidth=0.7, color="#e07a5f", label="Filtered")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Overlay comparison")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Save to your LaTeX image directory
    png_out = OUTPUT_DIR / f"standalone_filter_check_{REC_ID}.png"
    pdf_out = OUTPUT_DIR / f"standalone_filter_check_{REC_ID}.pdf"
    
    plt.savefig(png_out)
    plt.savefig(pdf_out)
    plt.close()
    
    print(f"📊 Success! Polished standalone thesis figure saved to:\n   -> {png_out}\n   -> {pdf_out}")