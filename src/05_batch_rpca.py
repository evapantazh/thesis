import subprocess
import sys
import json
from pathlib import Path

FILTERED_DIR   = Path(r"C:\Projects\thesis\data\FILTERED_files")
RPCA_SCRIPT    = Path(r"C:\Projects\thesis\src\rpca_minimization.py")
OUTPUT_DIR     = Path(r"C:\Projects\thesis\data\RPCA_files")

# Find all FILTERED CSVs
filtered_files = sorted(FILTERED_DIR.glob("FILTERED_*.csv"))
print(f"Found {len(filtered_files)} FILTERED files\n")

for filtered_file in filtered_files:
    rec_id = filtered_file.stem.replace("FILTERED_", "")
    output_csv = OUTPUT_DIR / f"LOWRANK_{rec_id}.csv"

    # Skip if already processed
    if output_csv.exists():
        print(f"Skipping {rec_id} — already done")
        continue

    print(f"Processing: {rec_id} ...")

    result = subprocess.run(
        [sys.executable, str(RPCA_SCRIPT), rec_id]
    )

    if result.returncode == 0:
        print(f"✅ Done: {rec_id}\n")
    else:
        print(f"❌ FAILED: {rec_id}\n")

print("\nBatch complete.")

# --- SUMMARY ---
print("\n" + "="*70)
print("RPCA SUMMARY")
print("="*70)
print(f"{'Recording':35s} | {'Rank':>4} | {'Gamma':>6} | {'nnz_S':>8} | {'Total':>8}")
print("-"*70)

for meta_file in sorted(OUTPUT_DIR.glob("rpca_meta_*.json")):
    rec_id = meta_file.stem.replace("rpca_meta_", "")
    with open(meta_file) as f:
        m = json.load(f)
    total = 971 * 28  # approximate, will vary per recording
    print(f"{rec_id:35s} | {m['rank']:>4d} | {m['gamma']:>6.4f} | {m.get('nnz_sparse',0):>8d} | {m.get('nnz_sparse',0)/27188*100:>7.1f}%")

print("="*70)