"""
coincidence_analysis.py
=======================
Identifies recordings where the ground-truth cardiac frequency coincides
with harmonics (k=2, k=3) of the respiration frequency. In such cases,
the harmonic-stack scoring cannot distinguish cardiac from breathing alias
and the eigenvector selection may lock onto the wrong component.

For each recording:
  1. Estimate f_resp from RAW grid data (sternum cells, settled portion)
     using Welch PSD in the respiration band [0.15, 0.45] Hz.
  2. Convert GT HR to Hz.
  3. Compute distance from k*f_resp for k=2, 3.
  4. Flag as 'coincidence' if min distance < tolerance.

Output:
  - coincidence_analysis.json (full per-recording data)
  - coincidence_summary.csv (table for thesis)
  - Console printout with the headline number

Usage:
  python coincidence_analysis.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, detrend

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"C:\Projects\thesis\data")
GRID_DIR = BASE / "GRID_files"
METADATA_DIR = BASE / "metadata"
TAP_DIR = BASE / "tap_info" / "json"
MOVESENSE_DIR = BASE / "movesense"
TIMESTAMPS_BASE = Path(r"D:\recordings")
TIMESTAMPS_BACKUP = BASE / "timestamps_backup"

OUTPUT_JSON = Path(r"C:\Projects\thesis\coincidence_analysis.json")
OUTPUT_CSV = Path(r"C:\Projects\thesis\coincidence_summary.csv")

# Sternum cells (confirmed anatomically correct via visualize_grid_cells.py)
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]

# Respiration search band (Hz) — typical resting breathing is 12-20 BPM
RESP_LO_HZ = 0.15  # 9 BPM
RESP_HI_HZ = 0.45  # 27 BPM

# Coincidence detection
COINCIDENCE_TOL_HZ = 0.08  # ~5 BPM tolerance
K_HARMONICS = [2, 3]       # k=2 and k=3 of respiration

# Timing
SETTLE_SEC = 5.0  # discard first 5s after tap to avoid transients
NPERSEG_SEC = 30.0  # Welch window length


# ============================================================
# HELPERS
# ============================================================

def load_fps(rec_id):
    """Load actual FPS from metadata, fallback to 15.0."""
    meta_file = METADATA_DIR / f"meta_{rec_id}.json"
    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        return float(meta.get("actual_fps", 15.0))
    except FileNotFoundError:
        return 15.0


def load_tap_info(rec_id):
    """Load camera tap time (seconds from recording start)."""
    tap_file = TAP_DIR / f"Tap_info_{rec_id}.json"
    if not tap_file.exists():
        return None
    try:
        with open(tap_file, 'r') as f:
            tap = json.load(f)
        return float(tap.get("manual_tap_sec", 0.0))
    except (json.JSONDecodeError, KeyError):
        return None


def load_timestamps(rec_id):
    """Load timestamps from D:\\recordings or backup."""
    ts_path = TIMESTAMPS_BASE / rec_id / "timestamps.csv"
    if not ts_path.exists():
        ts_path = TIMESTAMPS_BACKUP / rec_id / "timestamps.csv"
    if not ts_path.exists():
        return None
    return pd.read_csv(ts_path).set_index("frame")


def load_gt_hr(rec_id):
    """Load mean ground-truth HR from Movesense JSON."""
    hr_file = MOVESENSE_DIR / rec_id / "heartRate_stream.json"
    if not hr_file.exists():
        return None
    try:
        with open(hr_file, 'r') as f:
            data = json.load(f)
        if 'data' in data:
            hr_values = [entry['heartRate']['average'] for entry in data['data']]
        else:
            hr_values = [data['heartRate']['average']]
        return float(np.mean(hr_values))
    except (json.JSONDecodeError, KeyError):
        return None


def estimate_f_resp(rec_id):
    """
    Estimate respiration frequency (Hz) from raw grid data.
    Uses sternum cells, settled portion, Welch PSD in resp band.
    Returns None if data unavailable.
    """
    grid_path = GRID_DIR / f"GRID_{rec_id}.csv"
    if not grid_path.exists():
        return None

    fs = load_fps(rec_id)
    tap_sec = load_tap_info(rec_id)
    ts_df = load_timestamps(rec_id)

    grid_df = pd.read_csv(grid_path)

    # Trim to settled portion
    if tap_sec is not None and ts_df is not None:
        try:
            frames = grid_df["frame"].values
            ts_values = ts_df.loc[frames, "depth_timestamp"].values \
                if "depth_timestamp" in ts_df.columns \
                else ts_df.loc[frames, "timestamp"].values
            t_sec = (ts_values - ts_values[0]) / 1000.0
            keep = t_sec >= tap_sec + SETTLE_SEC
            if not keep.any():
                return None
            grid_df = grid_df.iloc[keep].reset_index(drop=True)
        except (KeyError, ValueError):
            pass  # use full recording if timing fails

    # Average sternum cells (handle zeros via interpolation)
    sternum_cols = [f"cell_{c}" for c in STERNUM_CELLS if f"cell_{c}" in grid_df.columns]
    if not sternum_cols:
        return None

    signals = []
    for col in sternum_cols:
        sig = grid_df[col].values.astype(float)
        mask = sig == 0
        if mask.any():
            idx = np.arange(len(sig))
            good = ~mask
            if good.sum() < 2:
                continue
            sig[mask] = np.interp(idx[mask], idx[good], sig[good])
        signals.append(sig)

    if not signals:
        return None

    sternum_signal = np.mean(signals, axis=0)
    sternum_signal = detrend(sternum_signal, type="linear")

    nperseg = min(int(NPERSEG_SEC * fs), len(sternum_signal))
    if nperseg < 64:
        return None

    freqs, psd = welch(sternum_signal, fs=fs,
                       nperseg=nperseg, noverlap=nperseg // 2)

    resp_band = (freqs >= RESP_LO_HZ) & (freqs <= RESP_HI_HZ)
    if not resp_band.any():
        return None

    f_resp = float(freqs[resp_band][np.argmax(psd[resp_band])])
    return f_resp


def check_coincidence(gt_hr_bpm, f_resp_hz):
    """
    Check if GT HR coincides with k*f_resp for k=2 or k=3.
    Returns dict with detection result and minimum distance.
    """
    gt_hz = gt_hr_bpm / 60.0
    min_dist_hz = float('inf')
    coincidence_k = None

    for k in K_HARMONICS:
        harmonic_hz = k * f_resp_hz
        dist = abs(gt_hz - harmonic_hz)
        if dist < min_dist_hz:
            min_dist_hz = dist
            if dist < COINCIDENCE_TOL_HZ:
                coincidence_k = k

    return {
        "coincidence": coincidence_k is not None,
        "coincidence_k": coincidence_k,
        "min_distance_hz": min_dist_hz,
        "min_distance_bpm": min_dist_hz * 60.0,
    }


def list_recordings():
    """List all recordings that have both GRID and movesense data."""
    rec_ids = []
    for grid_file in GRID_DIR.glob("GRID_*.csv"):
        rec_id = grid_file.stem.replace("GRID_", "")
        hr_file = MOVESENSE_DIR / rec_id / "heartRate_stream.json"
        if hr_file.exists():
            rec_ids.append(rec_id)
    return sorted(rec_ids)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COINCIDENCE ANALYSIS")
    print("=" * 70)
    print(f"Detecting recordings where GT HR coincides with k*f_resp")
    print(f"  k harmonics: {K_HARMONICS}")
    print(f"  Tolerance:   {COINCIDENCE_TOL_HZ} Hz ({COINCIDENCE_TOL_HZ*60:.1f} BPM)")
    print(f"  Resp band:   [{RESP_LO_HZ}, {RESP_HI_HZ}] Hz")
    print(f"  Sternum cells: {STERNUM_CELLS}")
    print()

    rec_ids = list_recordings()
    print(f"Found {len(rec_ids)} recordings\n")

    results = []
    for i, rec_id in enumerate(rec_ids, 1):
        gt_hr = load_gt_hr(rec_id)
        if gt_hr is None:
            print(f"[{i:3d}/{len(rec_ids)}] {rec_id:<25} SKIP (no GT)")
            continue

        f_resp = estimate_f_resp(rec_id)
        if f_resp is None:
            print(f"[{i:3d}/{len(rec_ids)}] {rec_id:<25} SKIP (no resp estimate)")
            continue

        check = check_coincidence(gt_hr, f_resp)
        f_resp_bpm = f_resp * 60.0

        # Parse rec_id: SUBJECT_DIST_CLOTH
        parts = rec_id.split("_")
        subject = parts[0] if len(parts) > 0 else ""
        distance = parts[1] if len(parts) > 1 else ""
        clothing = parts[2] if len(parts) > 2 else ""

        result = {
            "rec_id": rec_id,
            "subject": subject,
            "distance": distance,
            "clothing": clothing,
            "gt_hr_bpm": gt_hr,
            "f_resp_hz": f_resp,
            "f_resp_bpm": f_resp_bpm,
            "k2_harmonic_bpm": 2 * f_resp_bpm,
            "k3_harmonic_bpm": 3 * f_resp_bpm,
            "min_distance_bpm": check["min_distance_bpm"],
            "coincidence": check["coincidence"],
            "coincidence_k": check["coincidence_k"],
        }
        results.append(result)

        flag = " <-- COINCIDENCE" if check["coincidence"] else ""
        print(f"[{i:3d}/{len(rec_ids)}] {rec_id:<25} "
              f"GT={gt_hr:5.1f} f_resp={f_resp_bpm:5.1f} "
              f"min_dist={check['min_distance_bpm']:5.1f}{flag}")

    # ============================================================
    # SUMMARY
    # ============================================================
    if not results:
        print("\nNo recordings could be analyzed!")
        return

    coincidences = [r for r in results if r["coincidence"]]
    n_total = len(results)
    n_coin = len(coincidences)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total recordings analyzed: {n_total}")
    print(f"With coincidence problem:  {n_coin} ({100*n_coin/n_total:.1f}%)")
    print()

    if coincidences:
        print("Coincidence cases (sorted by min distance):")
        print(f"  {'rec_id':<25} {'GT':>6} {'f_resp':>8} {'k':>3} {'dist':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*3} {'-'*8}")
        for r in sorted(coincidences, key=lambda x: x["min_distance_bpm"]):
            print(f"  {r['rec_id']:<25} {r['gt_hr_bpm']:>5.1f}  "
                  f"{r['f_resp_bpm']:>6.1f}  "
                  f"{r['coincidence_k']:>3}  "
                  f"{r['min_distance_bpm']:>6.2f}")

    print()
    print("INTERPRETATION:")
    if n_coin <= 10:
        print(f"  -> Few coincidences ({n_coin}). Recommend: report as known")
        print(f"     limitation and exclude from main results table.")
    elif n_coin <= 20:
        print(f"  -> Moderate coincidences ({n_coin}). Consider implementing")
        print(f"     ICA fallback for these specific recordings.")
    else:
        print(f"  -> Many coincidences ({n_coin}). Systematic issue --")
        print(f"     ICA or DWT decomposition recommended.")

    # ============================================================
    # BREAKDOWN BY CATEGORY
    # ============================================================
    print()
    print("BREAKDOWN BY CONDITION:")
    print(f"  {'Condition':<20} {'n_total':>8} {'n_coin':>7} {'%':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*6}")

    df = pd.DataFrame(results)
    for cloth in sorted(df["clothing"].unique()):
        for dist in sorted(df["distance"].unique()):
            sub = df[(df["clothing"] == cloth) & (df["distance"] == dist)]
            if len(sub) == 0:
                continue
            n_sub_total = len(sub)
            n_sub_coin = sub["coincidence"].sum()
            print(f"  {cloth} @ {dist}mm".ljust(20),
                  f"{n_sub_total:>8d}",
                  f"{n_sub_coin:>7d}",
                  f"{100*n_sub_coin/n_sub_total:>5.1f}%")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            "config": {
                "k_harmonics": K_HARMONICS,
                "tolerance_hz": COINCIDENCE_TOL_HZ,
                "resp_band_hz": [RESP_LO_HZ, RESP_HI_HZ],
                "sternum_cells": STERNUM_CELLS,
            },
            "summary": {
                "n_total": n_total,
                "n_coincidence": n_coin,
                "coincidence_pct": 100 * n_coin / n_total,
            },
            "per_recording": results,
        }, f, indent=2)
    print(f"\nSaved JSON: {OUTPUT_JSON}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()