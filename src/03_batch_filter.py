import subprocess
import sys
from pathlib import Path

GRID_DIR       = Path(r"C:\Projects\thesis\data\GRID_files")
FILTER_SCRIPT  = Path(r"C:\Projects\thesis\src\03_filter_grid_cells.py")
OUTPUT_DIR     = Path(r"C:\Projects\thesis\data\FILTERED_files")

# Find all GRID CSVs
grid_files = sorted(GRID_DIR.glob("GRID_*.csv"))
print(f"Found {len(grid_files)} GRID files\n")

for grid_file in grid_files:
    rec_id = grid_file.stem.replace("GRID_", "")
    output_csv = OUTPUT_DIR / f"FILTERED_{rec_id}.csv"

    # Skip if already processed
    if output_csv.exists():
        print(f"Skipping {rec_id} — already done")
        continue

    print(f"Processing: {rec_id} ...")

    result = subprocess.run(
        [sys.executable, str(FILTER_SCRIPT), rec_id]
    )

    if result.returncode == 0:
        print(f"✅ Done: {rec_id}\n")
    else:
        print(f"❌ FAILED: {rec_id}\n")

print("\nBatch complete.")