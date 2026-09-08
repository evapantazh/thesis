"""
run_comparison_batch.py
=========================

Batch-runs eigenvectors_comparison.py over all valid recordings for
a chosen METHOD, then writes a summary CSV identical in format to
your existing summary CSVs so you can use the same analysis tools.

Usage:
    # Edit METHOD at top of eigenvectors_comparison.py first.
    # Then:
    python run_comparison_batch.py

It calls the comparison script in-process (faster than subprocess) by
importing it. If you'd rather subprocess each recording for safety
(e.g., a crash in one doesn't bring down the whole batch), swap to the
subprocess version commented at the bottom.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import helpers from your existing utility module
# (adjust this import path to where your `list_recordings` lives)
sys.path.insert(0, r"C:\Projects\thesis\src")
try:
    from data_loaders import list_recordings
except ImportError:
    # If that path is wrong, just glob for FILTERED_*.csv yourself
    FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
    def list_recordings(verify_files=True):
        return sorted([p.stem.replace("FILTERED_", "")
                       for p in FILTERED_DIR.glob("FILTERED_*.csv")])

# ============================================================
# Choose method here (must match METHOD in eigenvectors_comparison.py)
# ============================================================
METHOD = "RPCA"   # "PCA" | "ICA" | "RPCA"

OUTPUT_BASE = Path(rf"C:\Projects\thesis\data\PULSE_files_{METHOD}")
SUMMARY_DIR = OUTPUT_BASE / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Run all recordings via subprocess (safer: a crash in one
# recording doesn't crash the whole batch)
# ============================================================

import subprocess

SCRIPT_PATH = Path(__file__).parent / "eigenvectors_comparison.py"

recordings = list_recordings(verify_files=True)
print(f"Found {len(recordings)} recordings to process with METHOD={METHOD}")
print(f"Output dir: {OUTPUT_BASE}")
print()

# Sanity check: the METHOD here must match what's set in eigenvectors_comparison.py
# (You can verify by grep)

for i, rec_id in enumerate(recordings, 1):
    print(f"[{i}/{len(recordings)}] {rec_id} ...", end=" ", flush=True)
    try:
        result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), rec_id],
         capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED (returncode={result.returncode})")
            print(result.stderr[-500:])
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================
# Aggregate all the per-recording JSONs into a summary CSV
# ============================================================

print(f"\nAggregating results from {OUTPUT_BASE}")

rows = []
for jf in sorted(OUTPUT_BASE.glob("selection_windowed_*.json")):
    with open(jf, 'r') as f:
        d = json.load(f)

    rec_id = d.get('rec_id', jf.stem.replace("selection_windowed_", ""))
    parts = rec_id.split('_')

    bpm_init = d.get('bpm_initial_median')
    bpm_ref = d.get('bpm_refined_median')
    bpm_sm = d.get('bpm_smoothed')
    gt = d.get('gt_bpm')

    rows.append({
        'rec_id': rec_id,
        'subject': parts[0] if len(parts) > 0 else "",
        'dist': int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
        'cloth': parts[2] if len(parts) > 2 else "",
        'method': d.get('method', METHOD),
        'status': 'ok',
        'bpm_initial': bpm_init,
        'bpm_refined': bpm_ref,
        'bpm_smoothed': bpm_sm,
        'bpm_min': d.get('bpm_min'),
        'bpm_max': d.get('bpm_max'),
        'n_windows_total': d.get('n_windows_total'),
        'n_windows_valid': d.get('n_windows_valid'),
        'n_windows_kept': d.get('n_windows_kept_smoothing'),
        'gt_bpm': gt,
        'err_initial': abs(bpm_init - gt) if (bpm_init is not None and gt is not None) else None,
        'err_refined': abs(bpm_ref - gt) if (bpm_ref is not None and gt is not None) else None,
        'err_smoothed': abs(bpm_sm - gt) if (bpm_sm is not None and gt is not None) else None,
    })

summary = pd.DataFrame(rows).sort_values('rec_id').reset_index(drop=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = SUMMARY_DIR / f"summary_{METHOD}_{stamp}.csv"
summary.to_csv(out_csv, index=False)
print(f"\nWrote summary: {out_csv}")
print(f"n recordings: {len(summary)}")

# Print headline numbers
ok = summary[summary['err_smoothed'].notna()]
print(f"\n=== HEADLINE for METHOD={METHOD} ===")
print(f"n with GT: {len(ok)}")
print(f"MAE: {ok['err_smoothed'].mean():.3f} BPM")
print(f"Median err: {ok['err_smoothed'].median():.3f} BPM")
print(f"<5 BPM: {(ok['err_smoothed'] < 5).sum()}/{len(ok)} ({(ok['err_smoothed'] < 5).mean()*100:.1f}%)")
print(f"<10 BPM: {(ok['err_smoothed'] < 10).sum()}/{len(ok)} ({(ok['err_smoothed'] < 10).mean()*100:.1f}%)")

# Breakdown by condition
print(f"\n=== BY DISTANCE x CLOTHING ({METHOD}) ===")
print(ok.groupby(['dist', 'cloth'])['err_smoothed'].agg(['mean', 'median', 'count']))