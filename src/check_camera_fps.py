import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Projects\thesis\data")
REC_ID = "Sub03_800_Tshirt"

ts_df = pd.read_csv(BASE / f"Camera_{REC_ID}" / "timestamps.csv")
timestamps = ts_df["timestamp"].values

# Frame intervals in ms
intervals = np.diff(timestamps)
print(f"Mean interval:   {np.mean(intervals):.1f} ms")
print(f"Median interval: {np.median(intervals):.1f} ms")
print(f"Std interval:    {np.std(intervals):.1f} ms")
print(f"Min interval:    {np.min(intervals):.1f} ms")
print(f"Max interval:    {np.max(intervals):.1f} ms")

actual_fps = 1000.0 / np.median(intervals)
print(f"\nActual FPS: {actual_fps:.2f}")

total_duration = (timestamps[-1] - timestamps[0]) / 1000.0
print(f"Total duration: {total_duration:.2f} seconds")
print(f"Total frames: {len(timestamps)}")
print(f"Expected frames at 30fps: {total_duration * 30:.0f}")