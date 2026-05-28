"""
Batch runner for windowed HR estimation across all recordings.

For each recording:
  - Runs 06_eigenvectors.py as a subprocess
  - Reads the per-recording JSON output
  - Aggregates results into an ablation summary table

Produces:
  - A printed summary table to stdout
  - A CSV of all results for further analysis
  - A text-file copy of the printed table

Usage:
  python 07_batch.py                      # run all recordings
  python 07_batch.py --skip-existing      # skip recordings already processed
  python 07_batch.py --pattern AVE_*      # only run recordings matching pattern
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
PULSE_DIR    = Path(r"C:\Projects\thesis\data\PULSE_files")
EIGENVECTOR_SCRIPT = Path(__file__).parent / "06_eigenvectors.py"

SUMMARY_DIR = PULSE_DIR / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER RECORDINGS
# ============================================================

def discover_recordings(pattern=None):
    """List all REC_IDs that have a FILTERED_*.csv file."""
    if not FILTERED_DIR.exists():
        print(f"❌ FILTERED_DIR not found: {FILTERED_DIR}")
        sys.exit(1)

    rec_ids = []
    for f in sorted(FILTERED_DIR.glob("FILTERED_*.csv")):
        rec_id = f.stem.replace("FILTERED_", "")
        if pattern is None or _matches_pattern(rec_id, pattern):
            rec_ids.append(rec_id)
    return rec_ids


def _matches_pattern(rec_id, pattern):
    """Simple glob-style match: AVE_* matches AVE_800_tshirt, etc."""
    from fnmatch import fnmatch
    return fnmatch(rec_id, pattern)


# ============================================================
# RUN ONE RECORDING
# ============================================================

def run_one(rec_id, skip_existing=False):
    """
    Run the eigenvector script on one recording. Returns the parsed
    result JSON, or None on failure.
    """
    out_json = PULSE_DIR / f"selection_windowed_{rec_id}.json"

    if skip_existing and out_json.exists():
        try:
            with open(out_json, 'r') as f:
                return json.load(f)
        except Exception:
            pass  # fall through and re-run

    cmd = [sys.executable, str(EIGENVECTOR_SCRIPT), rec_id]
    # Force UTF-8 so non-ASCII diagnostics (e.g. arrows) don't crash
    # on Windows consoles using cp1253 / cp1252.
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace", env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ TIMEOUT on {rec_id}")
        return None
    except Exception as ex:
        print(f"  ⚠️ EXCEPTION on {rec_id}: {ex}")
        return None

    if proc.returncode != 0:
        print(f"  ⚠️ FAILED on {rec_id} (exit {proc.returncode})")
        # Print last few lines of stderr for diagnosis
        if proc.stderr:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            print(f"     stderr tail:\n{tail}")
        return None

    if not out_json.exists():
        print(f"  ⚠️ No output JSON produced for {rec_id} (exit {proc.returncode})")
        # Show what the script actually did so we can diagnose
        if proc.stdout:
            tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
            print(f"     stdout tail:\n{tail}")
        if proc.stderr:
            tail = "\n".join(proc.stderr.strip().splitlines()[-10:])
            print(f"     stderr tail:\n{tail}")
        return None

    try:
        with open(out_json, 'r') as f:
            return json.load(f)
    except Exception as ex:
        print(f"  ⚠️ Could not read {out_json}: {ex}")
        return None


# ============================================================
# TABLE FORMATTING
# ============================================================

def format_row(rec_id, result):
    """One row of the summary table."""
    if result is None:
        return (f"{rec_id:<30} | {'FAILED':>7} | {'':>7} | {'':>13} | "
                f"{'':>7} | {'':>6} | {'':>6}")

    bpm_init  = result.get('bpm_initial_median')
    bpm_ref   = result.get('bpm_refined_median')
    bpm_smo   = result.get('bpm_smoothed')
    bpm_min   = result.get('bpm_min')
    bpm_max   = result.get('bpm_max')
    n_valid   = result.get('n_windows_valid', 0)
    n_total   = result.get('n_windows_total', 0)
    n_kept    = result.get('n_windows_kept_smoothing', 0)
    gt        = result.get('gt_bpm')

    range_str = (f"{bpm_min:.1f}-{bpm_max:.1f}" if bpm_min is not None
                 else "")
    valid_str = f"{n_kept}/{n_valid}/{n_total}"

    if gt is None:
        err_str = ""
        gt_str  = ""
    else:
        err = abs(bpm_smo - gt) if bpm_smo is not None else None
        err_str = f"{err:.1f}" if err is not None else ""
        gt_str  = f"{gt:.1f}"

    return (f"{rec_id:<30} | "
            f"{bpm_init:>7.1f} | "
            f"{bpm_ref:>7.1f} | "
            f"{bpm_smo:>7.1f} | "
            f"{range_str:>13} | "
            f"{valid_str:>9} | "
            f"{gt_str:>6} | "
            f"{err_str:>6}")


HEADER = ("Recording                      |"
          " Init    | Refined | Smoothed |"
          "         Range |    Kept/Val/Tot |     GT |  Error")
SEPARATOR = "=" * len(HEADER)
THIN_SEP  = "-" * len(HEADER)


def print_summary(results):
    """Print summary table + aggregate stats. Returns the text."""
    lines = []
    lines.append(SEPARATOR)
    lines.append("WINDOWED EIGENVECTOR SELECTION — ABLATION SUMMARY")
    lines.append("Columns: Initial = pre-refinement median, "
                 "Refined = post-comb-filter median, Smoothed = anchor-filtered final")
    lines.append(SEPARATOR)
    lines.append(HEADER)
    lines.append(THIN_SEP)

    for rec_id, result in sorted(results.items()):
        lines.append(format_row(rec_id, result))
    lines.append(SEPARATOR)

    # ─── Aggregate stats ──────────────────────────────────────
    valid = [(rid, r) for rid, r in results.items() if r is not None]
    with_gt = [(rid, r) for rid, r in valid if r.get('gt_bpm') is not None]

    lines.append(f"Total recordings   : {len(results)}")
    lines.append(f"Successful         : {len(valid)}")
    lines.append(f"With ground truth  : {len(with_gt)}")

    if not with_gt:
        return "\n".join(lines)

    # Errors at each stage of the ablation
    errs_init = []
    errs_ref  = []
    errs_smo  = []
    for _, r in with_gt:
        gt = r['gt_bpm']
        if r.get('bpm_initial_median') is not None:
            errs_init.append(abs(r['bpm_initial_median'] - gt))
        if r.get('bpm_refined_median') is not None:
            errs_ref.append(abs(r['bpm_refined_median'] - gt))
        if r.get('bpm_smoothed') is not None:
            errs_smo.append(abs(r['bpm_smoothed'] - gt))

    def _stats(errs, label):
        if not errs:
            return f"  {label:<22}: no data"
        arr = np.array(errs)
        return (f"  {label:<22}: "
                f"mean={arr.mean():5.2f}  median={np.median(arr):5.2f}  "
                f"<5BPM={(arr < 5).sum():2d}/{len(arr)}  "
                f"<10BPM={(arr < 10).sum():2d}/{len(arr)}")

    lines.append("")
    lines.append("ABLATION (absolute error vs Movesense GT):")
    lines.append(_stats(errs_init, "Stage 1: Initial"))
    lines.append(_stats(errs_ref,  "Stage 2: + Refinement"))
    lines.append(_stats(errs_smo,  "Stage 3: + Smoothing"))

    # Improvement deltas
    if errs_init and errs_smo:
        d_mean = np.mean(errs_init) - np.mean(errs_smo)
        d_med  = np.median(errs_init) - np.median(errs_smo)
        lines.append("")
        lines.append(f"Total improvement     : mean {d_mean:+.2f} BPM, "
                     f"median {d_med:+.2f} BPM")

    return "\n".join(lines)


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(results, csv_path):
    """Dump all per-recording metrics to a CSV for downstream analysis."""
    rows = []
    for rec_id, r in sorted(results.items()):
        # Try to split rec_id into subject/dist/cloth (e.g. "AVE_800_tshirt")
        parts = rec_id.split("_")
        subject = parts[0] if len(parts) >= 1 else ""
        dist    = parts[1] if len(parts) >= 2 else ""
        cloth   = parts[2] if len(parts) >= 3 else ""

        if r is None:
            rows.append({
                'rec_id': rec_id, 'subject': subject, 'dist': dist, 'cloth': cloth,
                'status': 'failed',
            })
            continue

        gt = r.get('gt_bpm')
        bpm_smo = r.get('bpm_smoothed')
        bpm_ref = r.get('bpm_refined_median')
        bpm_init = r.get('bpm_initial_median')

        rows.append({
            'rec_id': rec_id,
            'subject': subject,
            'dist': dist,
            'cloth': cloth,
            'status': 'ok',
            'bpm_initial': bpm_init,
            'bpm_refined': bpm_ref,
            'bpm_smoothed': bpm_smo,
            'bpm_min': r.get('bpm_min'),
            'bpm_max': r.get('bpm_max'),
            'n_windows_total': r.get('n_windows_total'),
            'n_windows_valid': r.get('n_windows_valid'),
            'n_windows_kept': r.get('n_windows_kept_smoothing'),
            'gt_bpm': gt,
            'err_initial':  abs(bpm_init - gt) if (gt and bpm_init is not None) else None,
            'err_refined':  abs(bpm_ref  - gt) if (gt and bpm_ref  is not None) else None,
            'err_smoothed': abs(bpm_smo  - gt) if (gt and bpm_smo  is not None) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch HR estimation runner")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern to filter recordings (e.g. 'AVE_*')")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip recordings that already have a result JSON")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Don't run the eigenvector script; just aggregate existing JSONs")
    args = parser.parse_args()

    if not EIGENVECTOR_SCRIPT.exists():
        print(f"❌ Cannot find eigenvector script: {EIGENVECTOR_SCRIPT}")
        sys.exit(1)

    rec_ids = discover_recordings(pattern=args.pattern)
    if not rec_ids:
        print("❌ No recordings found.")
        sys.exit(1)

    print(f"Found {len(rec_ids)} recordings")
    if args.pattern:
        print(f"Filter pattern : {args.pattern}")
    if args.skip_existing:
        print("Skip-existing  : ON (will reuse existing JSONs)")
    if args.no_rerun:
        print("No-rerun       : ON (aggregating existing JSONs only)")
    print()

    results = {}
    t_start = time.time()

    for i, rec_id in enumerate(rec_ids, 1):
        elapsed = time.time() - t_start
        avg_per = elapsed / max(i - 1, 1)
        eta = avg_per * (len(rec_ids) - i + 1)
        print(f"[{i:3d}/{len(rec_ids)}] {rec_id}   "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        if args.no_rerun:
            out_json = PULSE_DIR / f"selection_windowed_{rec_id}.json"
            if out_json.exists():
                try:
                    with open(out_json, 'r') as f:
                        results[rec_id] = json.load(f)
                    continue
                except Exception:
                    pass
            results[rec_id] = None
        else:
            results[rec_id] = run_one(rec_id, skip_existing=args.skip_existing)

    # ─── Print summary ───────────────────────────────────────
    print()
    summary_text = print_summary(results)
    print(summary_text)

    # ─── Save summary text + CSV ─────────────────────────────
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = SUMMARY_DIR / f"summary_{stamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSummary text saved: {summary_path}")

    csv_path = SUMMARY_DIR / f"summary_{stamp}.csv"
    export_csv(results, csv_path)


if __name__ == "__main__":
    main()