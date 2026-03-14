import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- 1. FILTER DEFINITION ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- 2. LOAD DATA ---
file_path = r"C:\Projects\thesis\data\csv\chrys_500_tshirt_grid_FILTERED.csv"
OUPUT_DIR_FIGURE = r"C:\Projects\thesis\data\figures\chrys_500_tshirt_plot_FILTER"

df = pd.read_csv(file_path)
fs = 30.0  # Azure Kinect frequency
lowcut = 0.5   # 30 BPM
highcut = 4.0  # 240 BPM

# Choose a cell (e.g., cell_14 is roughly center-chest)
target_cell = "cell_14"
raw_signal = df[target_cell].values

# --- 3. APPLY FILTER ---
# We subtract the mean first (DC Offset removal)
clean_signal = raw_signal - np.mean(raw_signal)
filtered_signal = butter_bandpass_filter(clean_signal, lowcut, highcut, fs, order=4)

# --- 4. PLOT RESULTS ---

# --- 4.1 CROP FOR VISUALIZATION ---
crop_start = 150   # remove tap artifact (~5 seconds at 30 Hz)
crop_end = 900     # optional: stop earlier to zoom

raw_plot = raw_signal[crop_start:crop_end]
filtered_plot = filtered_signal[crop_start:crop_end]
frames = np.arange(crop_start, crop_end)
time = frames / fs

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, raw_plot, color='blue')
plt.title(f"Raw Depth Signal ({target_cell})")
plt.ylabel("Depth (mm)")

plt.subplot(2, 1, 2)
plt.plot(time, filtered_plot, color='red')
plt.title(f"Filtered Band-Pass Signal (30-240 BPM range)")
plt.ylabel("Vibration Amplitude")
plt.xlabel("Time(s)")

# --- Add grid to the plot ----
for ax in plt.gcf().axes:
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])

plt.tight_layout()
# -- Save figure --
plt.savefig(OUPUT_DIR_FIGURE,
            dpi=300,
            bbox_inches='tight')
plt.show()