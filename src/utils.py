import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Projects\thesis\data")

GRID_DIR = BASE /"GRID_files"
FILTERED_DIR = BASE / "FILTERED_files"
METADATA_DIR = BASE /"metadata"
TAP_DIR = BASE / "tap_info"/"json"
MOVESENSE_DIR = BASE/ "movesense"
TIMESTAMPS_BASE = Path(r"D:\recordings")
TIMESTAMPS_BACKUP = BASE/ "timestamps_backup"

# --------------- Helper functions --------------- #
# _________________________________________________#


def load_movesense_hr(rec_id, tap_info = None, settle_sec=5.0, show_stats = False, movesense_path = MOVESENSE_DIR):

    hr_file = movesense_path / rec_id/ "heartRate_stream.json"
    
    if not hr_file.exists():
        print(f"⚠️ No HR file found at {hr_file}")
        return None
    
    try:
        with open(hr_file, 'r') as f:
            entries = json.load(f)
    
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Failed to read {hr_file.name}: {e}")
        return None
    
    hr_data = entries.get("data", [])
    if not hr_data:
        print(f"⚠️ Empty HR stream for {rec_id}")
        return None
    
    hr_raw = []
    rr_per_sample = []
    for data in hr_data:
        hr_block = data.get("heartRate", {})
        avg = hr_block.get("average")
        rr_list = hr_block.get("rrData", [])
        if avg is None or not isinstance(rr_list, list) or len(rr_list) == 0:
            continue
        hr_raw.append(float(avg))
        rr_per_sample.append(sum(float(x) for x in rr_list))


    if not hr_raw:
        print(f"⚠️ No usable HR samples in {rec_id}")
        return None
    hr_array = np.array(hr_raw, dtype=float)
    rr_array = np.array(rr_per_sample, dtype=float)

    # Reconstruct time axis (seconds from HR-stream start)
    # rr_array = [712, 700, ... , 730]
    # np.cumsum(rr_array) =[712, 1412, ... , 2345] = time_sec
    # time_sec = time_sec - time_sec[0] = 2345 - 712 /1000

    time_sec = np.cumsum(rr_array) / 1000.0
    time_sec = time_sec - time_sec[0] if len(time_sec) > 0 else time_sec

    # Tap-based trimming
    trimmed = False
    n_pre_tap = 0
    if tap_info is not None:
        ms_tap = tap_info.get("ms_tap_sec")
        if ms_tap is not None:
            trim_start = ms_tap + settle_sec
            # time_sec   = [0.000, 0.728, 1.520, ..., 5.875, 6.918, 7.711, ...]
            # >= 6.2s?   = [False, False, False, ..., False, True,  True,  ...]
            keep_mask = time_sec >= trim_start  
            n_pre_tap = int(np.sum(~keep_mask))
            if keep_mask.any():
                hr_array = hr_array[keep_mask]
                rr_array = rr_array[keep_mask]
                time_sec = time_sec[keep_mask]
                trimmed = True
            else:
                print(f"⚠️ {rec_id}: tap+settle ({trim_start:.1f}s) is after all "
                      f"HR samples (last at {time_sec[-1]:.1f}s). Using full stream.")

    # Filter physiologically implausible HR values
    n_before_filter = len(hr_array)
    valid_mask = (hr_array >= 30) & (hr_array <= 220)
    hr_array = hr_array[valid_mask]
    rr_array = rr_array[valid_mask]
    time_sec = time_sec[valid_mask]
    n_dropped = n_before_filter - len(hr_array)

    if len(hr_array) == 0:
        print(f"⚠️ No valid HR samples in {rec_id} after filtering")
        return None

    result = {
        "rec_id":    rec_id,
        "hr_mean":   float(np.mean(hr_array)),
        "hr_std":    float(np.std(hr_array, ddof=1)) if len(hr_array) > 1 else 0.0,
        "hr_min":    float(np.min(hr_array)),
        "hr_max":    float(np.max(hr_array)),
        "hr_values": hr_array,
        "rr_values": rr_array,
        "time_sec":  time_sec,
        "n_samples": len(hr_array),
        "n_dropped": n_dropped,
        "n_pre_tap": n_pre_tap,
        "trimmed":   trimmed,
    }
    if show_stats:
        trim_msg = (f"post-tap+{settle_sec:.0f}s ({n_pre_tap} pre-tap removed)"
                    if trimmed else "full stream (no trim)")
        print(f"✓ {rec_id} HR: mean={result['hr_mean']:.1f} ± {result['hr_std']:.1f} BPM "
              f"(range {result['hr_min']:.0f}–{result['hr_max']:.0f}, "
              f"n={result['n_samples']}, dropped={n_dropped}, {trim_msg})")


    return result
    

def load_tap_info(rec_id, tap_method = "manual", tap_dir = TAP_DIR, show_stats= False):
    
    tap_file = tap_dir/ f"Tap_info_{rec_id}.json"
    
    if not tap_file.exists():
        print(f"⚠️ No tap info found at {tap_file}")
        return None
    
    try:
        with open(tap_file, 'r') as f:
            info = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Failed to read {tap_file.name}: {e}")
        return None

    key_map = { 
        "manual":     ("manual_tap_sec",    "manual_tap_frame",    "offset_sec"),
        "full_algo":  ("full_algo_sec",     "full_algo_frame",     "offset_full_algo"),
        "stern_algo": ("stern_algo_sec",    "stern_algo_frame",    "offset_stern_algo"),
    }

    if tap_method not in key_map:
        raise ValueError(
            f"tap_method must be one of {list(key_map.keys())}, got '{tap_method}'"
        )
    
    tap_sec_key, tap_frame_key, offset_sec_key = key_map[tap_method]


    try:
        result = {
            "rec_id" : info.get("rec_id", rec_id),
            "cam_tap_sec": float(info[tap_sec_key]),
            "cam_tap_frame" : int(info[tap_frame_key]),
            "offset_sec": float(info[offset_sec_key]),
            "ms_tap_sec": float(info["ms_tap_sec"]),
            "sternum_cells": list(info.get("sternum_cells", [5, 6, 9, 10, 13, 14])),
            "method": tap_method,

        }
    except (KeyError, TypeError, ValueError) as e:
        print(f"Missing or invalid field in {tap_file.name}: {e}")
        return None
    
    if show_stats:
        print(f"✓ {rec_id} tap ({tap_method}): "
              f"cam={result['cam_tap_sec']:.3f}s, "
              f"ms={result['ms_tap_sec']:.3f}s, "
              f"offset={result['offset_sec']:+.3f}s")

    return result

def load_timestamps(rec_id, show_stats=False,
                     primary_dir=TIMESTAMPS_BASE,
                     backup_dir=TIMESTAMPS_BACKUP):
    ts_path = primary_dir / rec_id / "timestamps.csv"
    if not ts_path.exists():
        ts_path = backup_dir / rec_id / "timestamps.csv"
        if not ts_path.exists():
            print(f"⚠️ No timestamps file found for {rec_id}")
            print(f"   Tried: {primary_dir / rec_id / 'timestamps.csv'}")
            print(f"   Tried: {backup_dir / rec_id / 'timestamps.csv'}")
            return None
    try:
        df = pd.read_csv(ts_path).set_index("frame")
    except (OSError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"⚠️ Failed to read timestamps for {rec_id}: {e}")
        return None

    # Identify columns
    has_color = "color_timestamp" in df.columns
    has_depth = "depth_timestamp" in df.columns
    has_single = "timestamp" in df.columns

    if has_color:
        df["color_ts"] = df["color_timestamp"]
    elif has_single:
        df["color_ts"] = df["timestamp"]
    else:
        print(f"⚠️ No color timestamp column in {ts_path.name}. "
              f"Available: {df.columns.tolist()}")
        return None

    if has_depth:
        df["depth_ts"] = df["depth_timestamp"]
    elif has_single:
        df["depth_ts"] = df["timestamp"]
    else:
        print(f"⚠️ No depth timestamp column in {ts_path.name}. "
              f"Available: {df.columns.tolist()}")
        return None

    if show_stats:
        # Compute effective sample rate from depth timestamps
        ts = df["depth_ts"].values
        if len(ts) >= 2:
            intervals_ms = np.diff(ts)
            valid = intervals_ms[intervals_ms > 0]
            if len(valid) > 0:
                median_interval_ms = float(np.median(valid))
                fs = 1000.0 / median_interval_ms
                duration_sec = (ts[-1] - ts[0]) / 1000.0
                source = "primary" if "recordings" in str(ts_path) and "backup" not in str(ts_path) else "backup"
                print(f"✓ {rec_id} timestamps: {len(df)} frames, "
                      f"{fs:.2f} fps, {duration_sec:.1f}s duration ({source})")

   

    return df


def load_fps(rec_id, default = 15.0):
    
    df = load_timestamps(rec_id)
    if df is None or len(df) < 2:
        return default
    
    intervals_ms = np.diff(df["depth_ts"].values)
    valid = intervals_ms[intervals_ms > 0]
    if len(valid) == 0:
        return default
    return float(1000.0 / np.median(valid))



def list_recordings(verify_files=True):
    """
    List all available recording IDs in the dataset.

    Scans the GRID_files directory for files matching GRID_*.csv and
    extracts the recording ID. Optionally verifies that the required
    companion files exist for each recording.

    Parameters
    ----------
    verify_files : bool, default True
        If True, only return recordings that have ALL required files:
        - GRID_files/GRID_{rec_id}.csv
        - tap_info/json/Tap_info_{rec_id}.json
        - movesense/{rec_id}/heartRate_stream.json
        - timestamps in primary or backup location

    Returns
    -------
    list of str
        Recording IDs (e.g., ["AGE_800_tshirt", "AVE_1200_hoodie", ...]),
        sorted alphabetically.
    """
    if not GRID_DIR.exists():
        print(f"⚠️ Grid directory not found: {GRID_DIR}")
        return []

    grid_files = sorted(GRID_DIR.glob("GRID_*.csv"))
    candidates = [p.stem.replace("GRID_", "") for p in grid_files]

    if not verify_files:
        return candidates

    valid = []
    skipped = []
    for rec_id in candidates:
        # Check required companions
        tap_path = TAP_DIR / f"Tap_info_{rec_id}.json"
        ms_path = MOVESENSE_DIR / rec_id / "heartRate_stream.json"
        ts_primary = TIMESTAMPS_BASE / rec_id / "timestamps.csv"
        ts_backup = TIMESTAMPS_BACKUP / rec_id / "timestamps.csv"

        missing = []
        if not tap_path.exists():
            missing.append("tap")
        if not ms_path.exists():
            missing.append("hr")
        if not (ts_primary.exists() or ts_backup.exists()):
            missing.append("ts")

        if missing:
            skipped.append((rec_id, missing))
        else:
            valid.append(rec_id)

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} recording(s) with missing files:")
        for rec_id, missing in skipped:
            print(f"   {rec_id}: missing {', '.join(missing)}")

    return valid





