import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# --- SIGNAL PROCESSING FUNCTIONS ---

def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=125, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def find_sync_point(acc_file):
    with open(acc_file, 'r') as f:
        data = json.load(f)['data']
    start_ts = data[0]['acc']['Timestamp']
    all_timestamps, magnitudes = [], []
    for entry in data:
        ts = entry['acc']['Timestamp']
        samples = entry['acc']['ArrayAcc']
        for i, s in enumerate(samples):
            mag = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
            magnitudes.append(mag)
            all_timestamps.append(ts + (i * 5)) 
    peak_idx = np.argmax(magnitudes)
    return all_timestamps[peak_idx], (all_timestamps[peak_idx] - start_ts) / 1000.0

def extract_ecg_signal(ecg_file):
    with open(ecg_file, 'r') as f:
        data = json.load(f)['data']
    ecg_raw, ecg_times = [], []
    for entry in data:
        base_ts = entry['ecg']['Timestamp']
        samples = entry['ecg']['Samples']
        for i, val in enumerate(samples):
            ecg_raw.append(val)
            ecg_times.append(base_ts + (i * 8))
    return np.array(ecg_times), np.array(ecg_raw)

def calculate_instantaneous_hr(times, signal):
    peaks, _ = find_peaks(signal, prominence=1000, distance=80)
    peak_times = times[peaks]
    rr_intervals = np.diff(peak_times) / 1000.0
    valid_mask = (rr_intervals > 0.3) & (rr_intervals < 1.5)
    hr_sequence = 60.0 / rr_intervals[valid_mask]
    hr_timestamps = peak_times[:-1][valid_mask] + (np.diff(peak_times)[valid_mask] / 2)
    return hr_timestamps, hr_sequence, peaks, peak_times

def export_peak_timestamps(peak_times, sync_time, filename="ecg_peaks_alignment.csv"):
    """Saves timestamps relative to the chest tap for video alignment."""
    # Calculate time of each beat relative to the sync tap (t1 = 0)
    relative_times_ms = peak_times - sync_time
    relative_times_sec = relative_times_ms / 1000.0
    
    df = pd.DataFrame({
        'Absolute_Timestamp_ms': peak_times,
        'Relative_to_Tap_sec': relative_times_sec
    })
    
    # Filter to only keep peaks that happen AFTER the tap
    df = df[df['Relative_to_Tap_sec'] >= 0].reset_index(drop=True)
    df.to_csv(filename, index=False)
    print(f"\n[SUCCESS] Exported {len(df)} peaks to {filename}")
    return df

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    acc_json = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_acc_stream.json"
    ecg_json = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_ecg_stream.json"

    try:
        t1_ts, t1_sec = find_sync_point(acc_json)
        ecg_t, ecg_v_raw = extract_ecg_signal(ecg_json)
        ecg_v_clean = bandpass_filter(ecg_v_raw)
        hr_t, hr_v, r_peaks, all_peak_times = calculate_instantaneous_hr(ecg_t, ecg_v_clean)
        
        # EXPORT DATA
        peak_df = export_peak_timestamps(all_peak_times, t1_ts)
        
        print(f"First 5 beats after tap (seconds):")
        print(peak_df['Relative_to_Tap_sec'].head())

        # Plot for verification
        plt.figure(figsize=(12, 5))
        plt.plot(ecg_t, ecg_v_clean, label='Filtered ECG')
        plt.plot(ecg_t[r_peaks], ecg_v_clean[r_peaks], 'ro', label='R-peaks')
        plt.axvline(x=t1_ts, color='g', linestyle='--', label='Tap (t1)')
        plt.title("ECG Alignment Verification")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")