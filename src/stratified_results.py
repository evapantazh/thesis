"""
stratified_results.py
=====================
Compute stratified results based on oracle SNR analysis.

Combines:
  - Per-recording errors from PULSE_files_SQI/selection_windowed_{rec}.json
  - Per-recording SNR from oracle_cells/oracle_{rec}.json

Stratifies by:
  - All recordings
  - SNR > 3 (detectable signal)
  - SNR > 5 (strong signal)
  - SNR < 2 (undetectable - typically excluded with rationale)

Also stratifies within paper-comparable subset (1200mm + tshirt).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Projects\thesis\data")
PULSE_DIR = BASE / "PULSE_files_SQI"
ORACLE_DIR = BASE / "oracle_cells"
OUT_DIR = BASE / "stratified_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all():
    """Load all recording data into a unified DataFrame."""
    rows = []
    for pulse_file in sorted(PULSE_DIR.glob("selection_windowed_*.json")):
        rec_id = pulse_file.stem.replace("selection_windowed_", "")
        oracle_file = ORACLE_DIR / f"oracle_{rec_id}.json"

        if not oracle_file.exists():
            print(f"  ⚠ {rec_id}: no oracle file, skipping")
            continue

        pulse = json.load(open(pulse_file))
        oracle = json.load(open(oracle_file))

        # Get final BPM and GT
        bpm_final = pulse.get("bpm_smoothed") or pulse.get("bpm_refined") or pulse.get("bpm_median")
        gt_bpm = pulse.get("gt_bpm") or oracle.get("gt_hr_bpm")

        if bpm_final is None or gt_bpm is None:
            print(f"  ⚠ {rec_id}: missing bpm or gt")
            continue

        error = abs(bpm_final - gt_bpm)

        # Parse rec_id
        parts = rec_id.split("_")
        subject = parts[0]
        distance = parts[1] if len(parts) > 1 else ""
        clothing = parts[2] if len(parts) > 2 else ""

        # Oracle SNR
        snr = oracle["summary_stats"]["best_snr_value"]

        rows.append({
            "rec_id": rec_id,
            "subject": subject,
            "distance": distance,
            "clothing": clothing,
            "bpm_final": bpm_final,
            "gt_bpm": gt_bpm,
            "error": error,
            "best_snr": snr,
        })

    return pd.DataFrame(rows)


def summarize(df, label):
    """Print stats and return as dict."""
    if len(df) == 0:
        print(f"\n  {label}: n=0, no data")
        return None

    mae = df["error"].mean()
    median = df["error"].median()
    n_lt5 = (df["error"] < 5).sum()
    n_lt10 = (df["error"] < 10).sum()
    n = len(df)

    print(f"\n  {label} (n={n}):")
    print(f"    MAE:          {mae:.2f} BPM")
    print(f"    Median:       {median:.2f} BPM")
    print(f"    <5  BPM:      {n_lt5}/{n} ({100*n_lt5/n:.1f}%)")
    print(f"    <10 BPM:      {n_lt10}/{n} ({100*n_lt10/n:.1f}%)")

    return {
        "label": label,
        "n": int(n),
        "mae": float(mae),
        "median": float(median),
        "n_lt5": int(n_lt5),
        "n_lt10": int(n_lt10),
    }


def main():
    print("=" * 70)
    print("  STRATIFIED RESULTS BY ORACLE SNR")
    print("=" * 70)

    df = load_all()
    print(f"\nLoaded {len(df)} recordings")

    # Save full table
    df.sort_values("rec_id").to_csv(OUT_DIR / "all_recordings.csv", index=False)

    # Show distribution
    print(f"\nSNR distribution:")
    print(f"  SNR < 2:   {(df['best_snr'] < 2).sum()} recordings (undetectable)")
    print(f"  SNR 2-3:   {((df['best_snr'] >= 2) & (df['best_snr'] < 3)).sum()} recordings (marginal)")
    print(f"  SNR 3-5:   {((df['best_snr'] >= 3) & (df['best_snr'] < 5)).sum()} recordings (detectable)")
    print(f"  SNR >= 5:  {(df['best_snr'] >= 5).sum()} recordings (strong)")

    results = []

    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  LEVEL 1: PRIMARY RESULTS (all recordings)")
    print(f"{'─'*70}")
    results.append(summarize(df, "All recordings"))

    # Paper-comparable
    paper_sub = df[(df["distance"] == "1200") & (df["clothing"] == "tshirt")]
    results.append(summarize(paper_sub, "Paper-comparable (1200mm + tshirt)"))

    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  LEVEL 2: STRATIFIED BY SNR")
    print(f"{'─'*70}")

    df_snr3 = df[df["best_snr"] >= 3]
    results.append(summarize(df_snr3, "Detectable signal (SNR >= 3)"))

    df_snr5 = df[df["best_snr"] >= 5]
    results.append(summarize(df_snr5, "Strong signal (SNR >= 5)"))

    df_snr2 = df[df["best_snr"] < 2]
    results.append(summarize(df_snr2, "Undetectable signal (SNR < 2)"))

    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  LEVEL 3: PAPER-COMPARABLE × SNR STRATIFIED")
    print(f"{'─'*70}")

    paper_snr3 = paper_sub[paper_sub["best_snr"] >= 3]
    results.append(summarize(paper_snr3, "Paper-comparable + SNR >= 3"))

    paper_snr5 = paper_sub[paper_sub["best_snr"] >= 5]
    results.append(summarize(paper_snr5, "Paper-comparable + SNR >= 5"))

    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  LEVEL 4: BY CONDITION × SNR >= 3")
    print(f"{'─'*70}")

    for cloth in ["tshirt", "hoodie"]:
        for dist in ["800", "1200", "1800"]:
            sub = df[(df["distance"] == dist) & (df["clothing"] == cloth)
                     & (df["best_snr"] >= 3)]
            full_sub = df[(df["distance"] == dist) & (df["clothing"] == cloth)]
            if len(sub) == 0:
                continue
            mae = sub["error"].mean()
            n_kept = len(sub)
            n_total = len(full_sub)
            print(f"  {cloth:<8} @ {dist}mm | kept {n_kept}/{n_total} | MAE = {mae:.2f}")

    # ─────────────────────────────────────────────────────────
    # Save excluded recordings list
    excluded = df[df["best_snr"] < 3].sort_values("best_snr")
    if len(excluded) > 0:
        print(f"\n{'─'*70}")
        print(f"  EXCLUDED RECORDINGS (SNR < 3) — n={len(excluded)}")
        print(f"{'─'*70}")
        print(f"  {'rec_id':<30} {'SNR':>6} {'error':>7}")
        for _, row in excluded.iterrows():
            print(f"  {row['rec_id']:<30} {row['best_snr']:>6.2f}  {row['error']:>6.1f}")
        excluded.to_csv(OUT_DIR / "excluded_snr_lt3.csv", index=False)

    # Save results
    with open(OUT_DIR / "stratified_results.json", "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Saved to: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()