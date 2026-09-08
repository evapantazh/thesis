"""
diagnose_failures.py
====================
Read the per-window output of selection_windowed and classify WHY each
recording misses. Distinguishes:
  - respiration-harmonic lock (bpm near 2x/3x of f_resp)
  - sub-harmonic lock        (bpm near 0.5x GT)
  - double lock              (bpm near 2x GT)
  - scatter / no clear lock  (high variance, no dominant mode)

Needs:
  PULSE_files/selection_windowed_{rec}.json   (your methodology output)
  oracle_cells/oracle_{rec}.json              (has gt_hr + per-cell resp info)

f_resp is estimated from the oracle JSON's per-cell respiration band by
re-reading the RAW grid — but to stay light we approximate f_resp from the
dominant low-freq component reported per recording. If you want exact f_resp,
we can add a RAW-grid pass.
"""

import json
from pathlib import Path
import numpy as np
from scipy.signal import detrend, welch

from utils import load_tap_info, load_timestamps, load_fps, GRID_DIR

BASE = Path(r"C:\Projects\thesis\data")
PULSE_DIR = BASE / "PULSE_files"
ORACLE_DIR = BASE / "oracle_cells"

RECS = ["GBA_1200_tshirt", "EST_1200_tshirt", "KGI_1200_tshirt",
        "MST_1200_tshirt", "MPA_1200_tshirt"]

TOL_BPM = 6.0   # how close counts as "locked onto" a target

# Respiration search band (Hz) on the RAW grid
RESP_LO_HZ = 0.12   # ~7 BPM
RESP_HI_HZ = 0.45   # ~27 BPM
SETTLE_SEC = 5.0
NPERSEG_SEC = 30.0
N_CELLS = 28


def estimate_f_resp_bpm(rec):
    """Dominant respiration frequency (BPM) from the RAW grid, trimmed to tap+settle.
    Returns None if it can't be computed."""
    grid_path = GRID_DIR / f"GRID_{rec}.csv"
    if not grid_path.exists():
        return None
    import pandas as pd
    grid_df = pd.read_csv(grid_path)
    tap = load_tap_info(rec)
    ts_df = load_timestamps(rec)
    fs = load_fps(rec)
    if tap is not None and ts_df is not None:
        frames = grid_df["frame"].values
        t = (ts_df.loc[frames, "depth_ts"].values
             - ts_df.loc[frames, "depth_ts"].values[0]) / 1000.0
        keep = t >= tap["cam_tap_sec"] + SETTLE_SEC
        grid_df = grid_df.iloc[keep].reset_index(drop=True)

    nperseg = int(NPERSEG_SEC * fs)
    resp_peaks = []
    for c in range(N_CELLS):
        col = f"cell_{c}"
        if col not in grid_df.columns:
            continue
        sig = grid_df[col].values.astype(float)
        m = sig == 0
        if m.any():
            idx = np.arange(len(sig)); good = ~m
            if good.sum() < 2:
                continue
            sig[m] = np.interp(idx[m], idx[good], sig[good])
        if len(sig) < 16:
            continue
        sig = detrend(sig, type="linear")
        f, pxx = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)),
                       noverlap=min(nperseg, len(sig)) // 2)
        band = (f >= RESP_LO_HZ) & (f <= RESP_HI_HZ)
        if band.any():
            resp_peaks.append(f[band][np.argmax(pxx[band])])
    if not resp_peaks:
        return None
    return float(np.median(resp_peaks)) * 60.0


def classify(bpm, gt, f_resp_bpm):
    """Return a label for a single window's bpm estimate."""
    labels = []
    if abs(bpm - gt) <= TOL_BPM:
        labels.append("correct")
    if abs(bpm - 0.5 * gt) <= TOL_BPM:
        labels.append("sub_harmonic(0.5x GT)")
    if abs(bpm - 2.0 * gt) <= TOL_BPM:
        labels.append("double(2x GT)")
    if f_resp_bpm:
        for k in (2, 3, 4):
            if abs(bpm - k * f_resp_bpm) <= TOL_BPM:
                labels.append(f"resp_harmonic({k}x f_resp)")
    if not labels:
        labels.append("other")
    return labels


def analyze(rec):
    pulse_path = PULSE_DIR / f"selection_windowed_{rec}.json"
    oracle_path = ORACLE_DIR / f"oracle_{rec}.json"
    if not pulse_path.exists():
        print(f"\n{rec}: no pulse JSON ({pulse_path.name}) — skipping")
        return
    with open(pulse_path) as f:
        pulse = json.load(f)

    gt = pulse.get("gt_bpm")
    if gt is None and oracle_path.exists():
        with open(oracle_path) as f:
            gt = json.load(f).get("gt_hr_bpm")
    if gt is None:
        print(f"\n{rec}: no GT — skipping")
        return

    # Estimate f_resp (BPM) from the RAW grid
    f_resp_bpm = estimate_f_resp_bpm(rec)
    if f_resp_bpm:
        print(f"  f_resp ≈ {f_resp_bpm:.1f} BPM "
              f"({f_resp_bpm/60:.3f} Hz) | 2x={2*f_resp_bpm:.0f} "
              f"3x={3*f_resp_bpm:.0f} 4x={4*f_resp_bpm:.0f} BPM")

    windows = pulse.get("per_window", [])
    if not windows:
        print(f"\n{rec}: no per_window data")
        return

    bpm_init = np.array([w["bpm_initial"] for w in windows])
    bpm_ref  = np.array([w["bpm"] for w in windows])

    print(f"\n{'='*60}\n{rec}   GT={gt:.1f} BPM   "
          f"final={pulse.get('bpm_smoothed'):.1f}  "
          f"err={pulse.get('error_bpm'):.1f}\n{'='*60}")
    print(f"  windows: {len(windows)}")
    print(f"  bpm_initial: median={np.median(bpm_init):.1f}  "
          f"range={bpm_init.min():.0f}-{bpm_init.max():.0f}  "
          f"std={bpm_init.std():.1f}")
    print(f"  bpm_refined: median={np.median(bpm_ref):.1f}  "
          f"range={bpm_ref.min():.0f}-{bpm_ref.max():.0f}  "
          f"std={bpm_ref.std():.1f}")

    # Tally failure modes on the INITIAL estimate (before comb refine)
    tally = {}
    for b in bpm_init:
        for lab in classify(b, gt, f_resp_bpm):
            tally[lab] = tally.get(lab, 0) + 1
    print(f"  initial-estimate breakdown:")
    for lab, n in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"      {lab:<26} {n:>3}  ({100*n/len(windows):.0f}%)")

    # Which eigenvector tends to win?
    eidx = [w["eigvec_idx"] for w in windows]
    uniq, cnt = np.unique(eidx, return_counts=True)
    print(f"  eigvec winners: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # ── KEY CHECK: do correct windows have higher score than wrong ones? ──
    scores = np.array([w["score"] for w in windows])
    is_correct = np.array([abs(w["bpm_initial"] - gt) <= TOL_BPM for w in windows])
    sc_correct = scores[is_correct]
    sc_wrong   = scores[~is_correct]

    print(f"  score | correct: n={len(sc_correct):>2} "
          f"mean={sc_correct.mean():.2f} median={np.median(sc_correct):.2f}"
          if len(sc_correct) else "  score | correct: none")
    print(f"  score | wrong  : n={len(sc_wrong):>2} "
          f"mean={sc_wrong.mean():.2f} median={np.median(sc_wrong):.2f}"
          if len(sc_wrong) else "  score | wrong  : none")

    if len(sc_correct) and len(sc_wrong):
        ratio = sc_correct.mean() / max(sc_wrong.mean(), 1e-9)
        verdict = ("SCORE SEPARATES well" if ratio > 1.5 else
                   "score barely separates" if ratio > 1.05 else
                   "SCORE DOES NOT SEPARATE — score-based filtering won't help")
        print(f"  → correct/wrong score ratio = {ratio:.2f}  [{verdict}]")

        # If we kept only the top-50% by score, what would the median be?
        thr = np.percentile(scores, 50)
        top_half = scores >= thr
        if top_half.any():
            med_top = np.median([w["bpm_initial"]
                                 for w, k in zip(windows, top_half) if k])
            print(f"  → median of top-50%-score windows = {med_top:.1f} "
                  f"(GT={gt:.1f}, err={abs(med_top-gt):.1f})")


if __name__ == "__main__":
    for rec in RECS:
        analyze(rec)