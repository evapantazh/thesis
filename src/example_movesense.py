import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# --- SIGNAL PROCESSING FUNCTIONS ---

def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=125, order=2):
    """Applies a Butterworth bandpass filter to remove baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def find_sync_point(acc_file):
    """Locates the chest tap (t1) using the peak magnitude of accelerometer data."""
    with open(acc_file, 'r') as f:
        data = json.load(f)['data']
    
    start_ts = data[0]['acc']['Timestamp']
    all_timestamps = []
    magnitudes = []

    for entry in data:
        ts = entry['acc']['Timestamp']
        samples = entry['acc']['ArrayAcc']
        for i, s in enumerate(samples):
            # Calculate 3D magnitude to find the sharp mechanical impact
            mag = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            magnitudes.append(mag)
            # Movesense packet internal timing interpolation
            all_timestamps.append(ts + (i * 5)) 

    peak_idx = np.argmax(magnitudes)
    t1_timestamp = all_timestamps[peak_idx]
    t1_seconds = (t1_timestamp - start_ts) / 1000.0
    return t1_timestamp, t1_seconds

def extract_ecg_signal(ecg_file):
    """Parses raw ECG samples and assigns timestamps at 125Hz (8ms intervals)."""
    with open(ecg_file, 'r') as f:
        data = json.load(f)['data']
    
    ecg_raw = []
    ecg_times = []
    for entry in data:
        base_ts = entry['ecg']['Timestamp']
        samples = entry['ecg']['Samples']
        for i, val in enumerate(samples):
            ecg_raw.append(val)
            ecg_times.append(base_ts + (i * 8)) # 1000ms / 125Hz = 8ms
            
    return np.array(ecg_times), np.array(ecg_raw)

def calculate_instantaneous_hr(times, signal):
    """Detects R-peaks using prominence to filter noise and calculates beat-to-beat HR."""
    # Prominence measures how much a peak stands out from the baseline
    # distance=80 (640ms) prevents double-counting the same heartbeat
    peaks, _ = find_peaks(signal, prominence=1000, distance=80)
    
    peak_times = times[peaks]
    rr_intervals = np.diff(peak_times) / 1000.0 # Convert ms to seconds
    
    # Mathematical filter: ignore RR intervals that are physically impossible (e.g. noise)
    valid_mask = (rr_intervals > 0.3) & (rr_intervals < 1.5)
    
    hr_sequence = 60.0 / rr_intervals[valid_mask]
    hr_timestamps = peak_times[:-1][valid_mask] + (np.diff(peak_times)[valid_mask] / 2)
    
    return hr_timestamps, hr_sequence, peaks

def get_exact_10s_hr(hr_times, hr_values, sync_time):
    win_start = sync_time + 2000 # Skip the first 2s of tap noise
    win_end = sync_time + 10000 
    
    mask = (hr_times >= win_start) & (hr_times < win_end)
    window_hr = hr_values[mask]
    
    # NEW: Outlier Rejection
    # We use the Median because it ignores those 40 BPM and 87 BPM errors
    median_hr = np.median(window_hr)
    
    # Filter: Keep only beats within 10% of the median
    clean_hr = window_hr[(window_hr > median_hr * 0.8) & (window_hr < median_hr * 1.2)]
    
    print(f"--- Refined 10s Analysis ---")
    print(f"Raw Sequence: {window_hr}")
    print(f"Cleaned Average HR: {np.mean(clean_hr):.2f} BPM")
    print(f"Confidence: {len(clean_hr)}/{len(window_hr)} beats were valid.")
# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Paths to your specific Movesense recording files
    acc_json = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_acc_stream.json"
    ecg_json = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_ecg_stream.json"

    try:
        # 2. Synchronize (Find t1)
        t1_ts, t1_sec = find_sync_point(acc_json)
        print(f"--- Synchronization ---")
        print(f"Chest Tap (t1) identified at: {t1_sec:.3f}s (TS: {t1_ts})")

        # 3. Load and Clean ECG Data
        ecg_t, ecg_v_raw = extract_ecg_signal(ecg_json)
        # Apply the bandpass filter to remove the 'wavy' baseline wander
        ecg_v_clean = bandpass_filter(ecg_v_raw)

        # 4. Calculate HR using the CLEANED signal
        hr_t, hr_v, r_peaks = calculate_instantaneous_hr(ecg_t, ecg_v_clean)
        
        # 5. Extract Ground Truth for your comparison window
        get_exact_10s_hr(hr_t, hr_v, t1_ts)

        # 6. Visual Verification Plot
        plt.figure(figsize=(15, 6))
        # Plotting the Cleaned signal shows why peaks were picked
        plt.plot(ecg_t, ecg_v_clean, label='Filtered ECG (5-15Hz)', alpha=0.8)
        plt.plot(ecg_t[r_peaks], ecg_v_clean[r_peaks], 'ro', markersize=4, label='Detected R-peaks')
        plt.axvline(x=t1_ts, color='green', linestyle='--', label='Synchronization Tap (t1)')
        
        plt.title("ECG Ground Truth Analysis: Filtered Signal & Peak Detection")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (Filtered Units)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find the file. Check your path:\n{e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")