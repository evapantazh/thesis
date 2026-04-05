import subprocess
import sys
from pathlib import Path

RECORDINGS_DIR = Path(r"D:\recordings")
GRID_SCRIPT    = Path(r"C:\Projects\thesis\src\01_extract_grid.py")
OUTPUT_DIR     = Path(r"C:\Projects\thesis\data\GRID_files")

# Find all recording folders
recording_folders = sorted([
    f for f in RECORDINGS_DIR.iterdir()
    if f.is_dir()
    and (f / "depth").exists()
    and not f.name.startswith("Camera_")
])

print(f"Found {len(recording_folders)} recordings to process\n")

for folder in recording_folders:
    rec_id     = folder.name
    output_csv = OUTPUT_DIR / f"GRID_{rec_id}.csv"

    # Skip if already processed
    if output_csv.exists():
        print(f"Skipping {rec_id} — already done")
        continue

    print(f"Processing: {rec_id} ...")

    # Call the grid script with the recording path as argument
    result = subprocess.run(
        [sys.executable, str(GRID_SCRIPT), str(folder)],
        
    )

    if result.returncode == 0:
        print(f"Done: {rec_id}")
    else:
        print(f"[!] FAILED: {rec_id}")

print("\nBatch complete.")