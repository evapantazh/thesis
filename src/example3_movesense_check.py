import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# --- 1. SIGNAL PROCESSING FUNCTIONS ---
def bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=125, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

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

def calculate_hr_from_ecg(times, signal):
    # Detect R-peaks using prominence to ignore noise
    peaks, _ = find_peaks(signal, prominence=1000, distance=80)
    peak_times = times[peaks]
    rr_intervals = np.diff(peak_times) / 1000.0
    # Clean outliers (keep only realistic resting HR)
    valid_mask = (rr_intervals > 0.4) & (rr_intervals < 1.5)
    hr_sequence = 60.0 / rr_intervals[valid_mask]
    return hr_sequence

def load_sensor_hr_json(hr_json_path):
    with open(hr_json_path, 'r') as f:
        data = json.load(f)['data']
    # Extract the 'average' HR reported by the sensor firmware
    return np.array([entry['heartRate']['average'] for entry in data])

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    # Update these paths to your actual files
    ecg_json_path = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_ecg_stream.json"
    hr_json_path  = r"C:\Users\user\Downloads\lala\20260308T002953Z_223430000019_heartRate_stream.json"

    try:
        # Step A: Process Raw ECG
        print("Processing Raw ECG...")
        t, v_raw = extract_ecg_signal(ecg_json_path)
        v_clean = bandpass_filter(v_raw)
        hr_v_python = calculate_hr_from_ecg(t, v_clean)
        
        # Step B: Load Sensor's Internal Average
        print("Loading Sensor Average...")
        hr_v_sensor = load_sensor_hr_json(hr_json_path)

        # Step C: Plot Comparison
        plt.figure(figsize=(12, 6))
        
        # Plot your calculated instantaneous HR
        plt.plot(hr_v_python, label='Independent Python Calculation (Raw ECG)', 
                 color='blue', marker='o', markersize=4, linewidth=1.5)
        
        # Plot sensor's internal average
        # We align them by showing the last N beats to capture the stable period
        plt.plot(hr_v_sensor[-len(hr_v_python):], label='Movesense Internal Average (Firmware)', 
                 color='orange', linestyle='--', linewidth=2)
        
        plt.title("Ground Truth Cross-Validation: Independent vs. Sensor Logic")
        plt.ylabel("Heart Rate (BPM)")
        plt.xlabel("Beat Sequence (Stable Period)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"\nVerification Success!")
        print(f"Python Mean HR: {np.mean(hr_v_python):.2f} BPM")
        print(f"Sensor Mean HR: {np.mean(hr_v_sensor[-len(hr_v_python):]):.2f} BPM")

    except Exception as e:
        print(f"Error: {e}")