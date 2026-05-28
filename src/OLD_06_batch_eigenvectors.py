import subprocess
import sys
import json
from pathlib import Path

FILTERED_DIR  = Path(r"C:\Projects\thesis\data\FILTERED_files")
EIGEN_SCRIPT  = Path(r"C:\Projects\thesis\src\06_eigenvectors.py")
OUTPUT_DIR    = Path(r"C:\Projects\thesis\data\PULSE_files")

# Discover recordings from FILTERED files (these are what 06 actually reads)
filtered_files = sorted(FILTERED_DIR.glob("FILTERED_*.csv"))
print(f"Found {len(filtered_files)} FILTERED files\n")

# --- RUN BATCH ---
for fpath in filtered_files:
    rec_id = fpath.stem.replace("FILTERED_", "")
    output_json = OUTPUT_DIR / f"selection_windowed_{rec_id}.json"

    #if output_json.exists():
        #print(f"Skipping {rec_id} — already done")
        #continue

    print(f"Processing: {rec_id} ...")
    result = subprocess.run([sys.executable, str(EIGEN_SCRIPT), rec_id])

    if result.returncode == 0:
        print(f"OK: {rec_id}\n")
    else:
        print(f"FAILED: {rec_id}\n")

print("\nBatch complete.")

# --- SUMMARY ---
print("\n" + "=" * 90)
print("WINDOWED EIGENVECTOR SELECTION SUMMARY")
print("=" * 90)
print(f"{'Recording':30s} | {'BPM med':>7} | {'BPM top':>7} | {'Range':>13} | "
      f"{'Valid':>7} | {'GT':>6} | {'Error':>6}")
print("-" * 90)

# IMPORTANT: glob specifically for selection_windowed_*.json so we don't
# mix in old-format files from the previous eigenvector script
results = []
for sel_file in sorted(OUTPUT_DIR.glob("selection_windowed_*.json")):
    rec_id = sel_file.stem.replace("selection_windowed_", "")
    with open(sel_file) as f:
        s = json.load(f)

    bpm_med = s.get("bpm_median")
    bpm_top = s.get("bpm_top_half_median")
    bpm_min = s.get("bpm_min")
    bpm_max = s.get("bpm_max")
    n_valid = s.get("n_windows_valid")
    n_total = s.get("n_windows_total")
    gt      = s.get("gt_bpm")
    err     = s.get("error_bpm")

    gt_str  = f"{gt:>6.1f}"  if gt  is not None else "   N/A"
    err_str = f"{err:>6.1f}" if err is not None else "   N/A"
    range_str = f"{bpm_min:>5.1f}-{bpm_max:<5.1f}" if bpm_min is not None else "    N/A"
    valid_str = f"{n_valid:>3}/{n_total:<3}"      if n_valid is not None else "  N/A"

    print(f"{rec_id:30s} | {bpm_med:>7.1f} | {bpm_top:>7.1f} | {range_str:>13s} | "
          f"{valid_str:>7s} | {gt_str} | {err_str}")
    results.append((rec_id, bpm_med, gt, err))

print("=" * 90)

# Aggregate stats
valid_errors = [r[3] for r in results if r[3] is not None]
if valid_errors:
    import statistics
    print(f"\nTotal recordings: {len(results)}")
    print(f"With ground truth: {len(valid_errors)}")
    print(f"Mean abs error:   {statistics.mean(valid_errors):.2f} BPM")
    print(f"Median abs error: {statistics.median(valid_errors):.2f} BPM")
    print(f"Errors < 5 BPM:   {sum(1 for e in valid_errors if e < 5)} / {len(valid_errors)}")
    print(f"Errors < 10 BPM:  {sum(1 for e in valid_errors if e < 10)} / {len(valid_errors)}")
    print(f"Errors < 15 BPM:  {sum(1 for e in valid_errors if e < 15)} / {len(valid_errors)}")