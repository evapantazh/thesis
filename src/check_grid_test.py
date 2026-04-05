import glob
import os
from pathlib import Path

BASE = Path(r"C:\Projects\thesis\data")
COLOR_DIR = BASE / "Camera_Sub03_800_Tshirt" / "color"

color_files = sorted(glob.glob(os.path.join(COLOR_DIR, "frame_*.jpg")))
print(f"Python sees: {len(color_files)} color files")
print(f"First 5: {color_files[:5]}")
print(f"Last 5:  {color_files[-5:]}")