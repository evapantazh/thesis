"""
compare_aggregation.py
======================
Same per-window bpm values you already have — different ways to collapse
them into one number. Tests whether a density-mode anchor beats the median
when correct windows are a minority but form the tightest cluster.

Methods compared (all read from selection_windowed_{rec}.json, per_window):
  1. median_all        - plain median (baseline)
  2. current_smoothed  - whatever your pipeline saved as bpm_smoothed
  3. hist_mode         - center of the most populated histogram bin
  4. kde_mode          - peak of a Gaussian-KDE over the bpm values
  5. kde_anchor_median - KDE peak as anchor, then median of windows within
                         SMOOTH_TOLERANCE_BPM of it  (robust hybrid)

No re-running of RPCA. Pure post-processing on cached results.
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde


BASE = Path(r"C:\Projects\thesis\data")
PULSE_DIR = BASE / "PULSE_files_SQI"
ORACLE_DIR = BASE / "oracle_cells"

import os
RECS = []
for f in os.listdir(PULSE_DIR):
    if f.startswith("selection_windowed_") and f.endswith(".json"):
        RECS.append(f.replace("selection_windowed_", "").replace(".json", ""))
RECS.sort()

HR_MIN_BPM = 55
HR_MAX_BPM = 150
HIST_BIN_BPM = 3.0
SMOOTH_TOLERANCE_BPM = 12.0


def hist_mode(bpms, bin_w=HIST_BIN_BPM):
    edges = np.arange(HR_MIN_BPM, HR_MAX_BPM + bin_w, bin_w)
    counts, _ = np.histogram(bpms, bins=edges)
    if counts.sum() == 0:
        return float(np.median(bpms))
    k = int(np.argmax(counts))
    # center of the winning bin
    return float((edges[k] + edges[k + 1]) / 2.0)


def kde_mode(bpms):
    if len(bpms) < 3 or np.std(bpms) < 1e-6:
        return float(np.median(bpms))
    kde = gaussian_kde(bpms)
    grid = np.linspace(HR_MIN_BPM, HR_MAX_BPM, 2000)
    dens = kde(grid)
    return float(grid[np.argmax(dens)])


def kde_anchor_median(bpms, tol=SMOOTH_TOLERANCE_BPM):
    anchor = kde_mode(bpms)
    keep = np.abs(bpms - anchor) <= tol
    if keep.sum() < 3:
        return anchor
    return float(np.median(bpms[keep]))


def get_gt(pulse, rec):
    gt = pulse.get("gt_bpm")
    if gt is None:
        op = ORACLE_DIR / f"oracle_{rec}.json"
        if op.exists():
            gt = json.load(open(op)).get("gt_hr_bpm")
    return gt


def main():
    rows = []
    for rec in RECS:
        p = PULSE_DIR / f"selection_windowed_{rec}.json"
        if not p.exists():
            print(f"{rec}: no pulse JSON")
            continue
        pulse = json.load(open(p))
        gt = get_gt(pulse, rec)
        windows = pulse.get("per_window", [])
        if gt is None or not windows:
            print(f"{rec}: missing gt or windows")
            continue

        bpms = np.array([w["bpm"] for w in windows])

        methods = {
            "median_all":        float(np.median(bpms)),
            "current_smoothed":  pulse.get("bpm_smoothed"),
            "hist_mode":         hist_mode(bpms),
            "kde_mode":          kde_mode(bpms),
            "kde_anchor_median": kde_anchor_median(bpms),
        }
        errs = {m: (abs(v - gt) if v is not None else None)
                for m, v in methods.items()}
        rows.append((rec, gt, methods, errs))

        print(f"\n{rec}   GT={gt:.1f}")
        for m in methods:
            v = methods[m]; e = errs[m]
            tag = "" if v is None else f"{v:6.1f}  err={e:4.1f}"
            print(f"   {m:<20} {tag}")

    # Summary: mean abs error per method across recordings
    print(f"\n{'='*50}\n  MEAN ABSOLUTE ERROR (all recordings)\n{'='*50}")
    method_names = ["median_all", "current_smoothed", "hist_mode",
                    "kde_mode", "kde_anchor_median"]
    for m in method_names:
        es = [errs[m] for _, _, _, errs in rows if errs[m] is not None]
        if es:
            print(f"   {m:<20} MAE={np.mean(es):5.2f}  "
                  f"(per-rec: {', '.join(f'{e:.1f}' for e in es)})")


if __name__ == "__main__":
    main()