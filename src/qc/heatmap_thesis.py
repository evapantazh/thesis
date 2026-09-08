"""
export_snr_heatmap_overlay.py
==============================
For a chosen recording + frame, produces two panels:

  (a) Standalone 7x4 SNR heatmap (cardiac SNR per cell, from raw signal)
  (b) The same heatmap overlaid semi-transparently on the actual chest
      ROI location in the color (or depth) image.

Illustrates how much each grid cell contributes to the final cardiac
signal estimate. Uses the same SNR metric as snr_heatmap_fullgrid.py.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.signal import detrend, welch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import (
    load_tap_info, load_movesense_hr, load_timestamps, load_fps,
    GRID_DIR, BASE,
)

# ─────────────────────────────────────────────────────────────
#  THESIS FIGURE STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size':        11,
    'axes.labelsize':   11,
    'axes.titlesize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'mathtext.fontset': 'cm',
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

# ─────────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────────
REC_ID       = "EST_1800_tshirt"
FRAME_NUMBER = 400              # frame to visualize (color + ROI)
IMAGE_SOURCE = "depth"          # "color" or "depth"

SETTLE_SEC       = 5.0
HR_TOLERANCE_BPM = 3.0
NPERSEG_SEC      = 30.0
NOVERLAP_FRAC    = 0.5
ROWS, COLS       = 7, 4
SNR_BG_LO_HZ     = 1.5
SNR_BG_HI_HZ     = 3.5
STERNUM_CELLS    = [5, 6, 9, 10, 13, 14]

SHRINK_X        = 0.08
SHRINK_Y_TOP    = 0.05
SHRINK_Y_BOTTOM = 0.30

REC_BASE  = Path(r"D:\recordings")
REC_DIR   = REC_BASE / REC_ID
COLOR_DIR = REC_DIR / "color"
DEPTH_DIR = REC_DIR / "depth"

OUTPUT_DIR = Path(r"C:\Projects\thesis\thesis_latex\images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
#  SNR COMPUTATION — same logic as snr_heatmap_fullgrid.py
# ─────────────────────────────────────────────────────────────
def compute_cell_snr(rec_id):
    tap = load_tap_info(rec_id, show_stats=True)
    hr = load_movesense_hr(rec_id, tap_info=tap, settle_sec=SETTLE_SEC, show_stats=True)
    ts_df = load_timestamps(rec_id, show_stats=True)
    fs = load_fps(rec_id)

    grid_df = pd.read_csv(GRID_DIR / f"GRID_{rec_id}.csv")
    grid_frames = grid_df["frame"].values
    ts_values = ts_df.loc[grid_frames, "depth_ts"].values
    t_cam = (ts_values - ts_values[0]) / 1000.0

    trim_start_sec = tap["cam_tap_sec"] + SETTLE_SEC
    keep = t_cam >= trim_start_sec
    grid_df = grid_df.iloc[keep].reset_index(drop=True)

    nperseg = int(NPERSEG_SEC * fs)
    if nperseg > len(grid_df):
        nperseg = len(grid_df) // 2
    noverlap = int(nperseg * NOVERLAP_FRAC)

    gt_hr_hz = hr["hr_mean"] / 60.0
    tol_hz = HR_TOLERANCE_BPM / 60.0

    cardiac_snr = np.zeros(ROWS * COLS)

    for c in range(ROWS * COLS):
        sig = grid_df[f"cell_{c}"].values.astype(float)
        zero_mask = sig == 0
        if zero_mask.any():
            idx = np.arange(len(sig))
            good = ~zero_mask
            if good.sum() < 2:
                continue
            sig[zero_mask] = np.interp(idx[zero_mask], idx[good], sig[good])

        try:
            bp = list(range(int(10 * fs), len(sig), int(10 * fs)))
            sig_dt = detrend(sig, type="linear", bp=bp) if bp else \
                     detrend(sig, type="linear")
        except ValueError:
            sig_dt = detrend(sig, type="linear")

        if len(sig_dt) < nperseg:
            continue

        f, pxx = welch(sig_dt, fs=fs, nperseg=nperseg, noverlap=noverlap)
        hr_mask = (f >= gt_hr_hz - tol_hz) & (f <= gt_hr_hz + tol_hz)
        peak = pxx[hr_mask].max() if hr_mask.any() else 0.0
        bg_mask = (f >= SNR_BG_LO_HZ) & (f <= SNR_BG_HI_HZ) & ~hr_mask
        bg = max(np.median(pxx[bg_mask]), 1e-20)
        cardiac_snr[c] = peak / bg

    return cardiac_snr.reshape(ROWS, COLS), gt_hr_hz * 60.0


# ─────────────────────────────────────────────────────────────
#  CHEST ROI — same logic as extract_grid.py / visualize_sternum_grid.py
# ─────────────────────────────────────────────────────────────
def get_chest_roi(img, pose):
    h, w = img.shape[:2]
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        raise RuntimeError("No pose landmarks detected in this frame.")

    lm = results.pose_landmarks.landmark
    norm_x1 = min(lm[11].x, lm[12].x)
    norm_x2 = max(lm[11].x, lm[12].x)
    norm_y1 = min(lm[11].y, lm[12].y)
    norm_y2 = max(lm[23].y, lm[24].y)

    torso_w = norm_x2 - norm_x1
    torso_h = norm_y2 - norm_y1

    chest_x1 = norm_x1 + torso_w * SHRINK_X
    chest_x2 = norm_x2 - torso_w * SHRINK_X
    chest_y1 = norm_y1 + torso_h * SHRINK_Y_TOP
    chest_y2 = norm_y2 - torso_h * SHRINK_Y_BOTTOM

    x1 = clamp(int(chest_x1 * w), 0, w - 1)
    x2 = clamp(int(chest_x2 * w), 0, w - 1)
    y1 = clamp(int(chest_y1 * h), 0, h - 1)
    y2 = clamp(int(chest_y2 * h), 0, h - 1)

    return x1, y1, x2, y2


def depth_to_display(depth_img, cmap_name="gray"):
    """Grayscale-ish background for the depth image so the SNR overlay
    (colored) stands out clearly on top."""
    valid = depth_img[depth_img > 0]
    if valid.size == 0:
        return np.zeros((*depth_img.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    clipped = np.clip(depth_img, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-6)
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[depth_img <= 0] = 0
    return rgb


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    snr_grid, gt_hr_bpm = compute_cell_snr(REC_ID)

    # Load background image (color or depth) for the overlay panel
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7,
                        model_complexity=1)

    color_path = COLOR_DIR / f"frame_{FRAME_NUMBER:05d}.jpg"
    color_img = cv2.imread(str(color_path))
    if color_img is None:
        raise FileNotFoundError(f"Could not load {color_path}")
    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = get_chest_roi(color_img, pose)

    if IMAGE_SOURCE == "depth":
        depth_path = DEPTH_DIR / f"frame_{FRAME_NUMBER:05d}.npy"
        depth_raw = np.load(depth_path).astype(np.float32)
        ch, cw = color_img.shape[:2]
        dh, dw = depth_raw.shape[:2]
        sx, sy = dw / cw, dh / ch
        bg_img = depth_to_display(depth_raw)
        rx1, rx2 = int(x1 * sx), int(x2 * sx)
        ry1, ry2 = int(y1 * sy), int(y2 * sy)
    else:
        bg_img = color_rgb
        rx1, rx2, ry1, ry2 = x1, x2, y1, y2

    pose.close()

    vmin, vmax = 0, np.nanmax(snr_grid)
    norm = Normalize(vmin=vmin, vmax=vmax)
    # Changed colormap to YlOrRd (Cream -> Yellow -> Orange -> Red)
    cmap = cm.get_cmap('YlOrRd')

    # ── Figure 1: standalone heatmap ──────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    im1 = ax1.imshow(snr_grid, cmap=cmap, norm=norm, aspect='equal')
    
    # Draw thick outer border for the whole grid
    ax1.add_patch(plt.Rectangle((-0.5, -0.5), COLS, ROWS,
                                fill=False, edgecolor='black',
                                linewidth=4.0))

    for r in range(ROWS):
        for c in range(COLS):
            cell_idx = r * COLS + c
            # Adjusted text color threshold for YlOrRd: Dark red gets white text, lighter colors get black
            txt_color = 'white' if snr_grid[r, c] > vmax * 0.7 else 'black'
            ax1.text(c, r, f"{snr_grid[r, c]:.1f}", ha='center', va='center',
                     fontsize=8.5, color=txt_color,
                     fontweight='bold' if cell_idx in STERNUM_CELLS else 'normal')
            
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Cardiac SNR per cell")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label('SNR (peak / background)')
    fig1.suptitle(f"{REC_ID}   |   GT HR = {gt_hr_bpm:.1f} BPM", fontsize=10)

    out1 = OUTPUT_DIR / f"snr_heatmap_{REC_ID}.png"
    fig1.savefig(out1)
    fig1.savefig(out1.with_suffix('.pdf'))
    print(f"Saved → {out1}")

    # ── Figure 2: heatmap overlaid on image ───────────────────
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax2.imshow(bg_img)
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1
    cell_w = roi_w / COLS
    cell_h = roi_h / ROWS

    for r in range(ROWS):
        for c in range(COLS):
            x0 = rx1 + c * cell_w
            y0 = ry1 + r * cell_h
            color = cmap(norm(snr_grid[r, c]))
            # Removed edge color so individual cells have no outline
            ax2.add_patch(plt.Rectangle((x0, y0), cell_w, cell_h,
                                        facecolor=color, linewidth=0, alpha=0.55))
            
    # Draw intense outline ONLY around the entire grid
    ax2.add_patch(plt.Rectangle((rx1, ry1), roi_w, roi_h,
                                fill=False, edgecolor='black',
                                linewidth=4.0))

    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"SNR overlay on {IMAGE_SOURCE} image")
    fig2.suptitle(f"{REC_ID}   |   GT HR = {gt_hr_bpm:.1f} BPM", fontsize=10)

    out2 = OUTPUT_DIR / f"snr_overlay_{REC_ID}.png"
    fig2.savefig(out2)
    fig2.savefig(out2.with_suffix('.pdf'))
    print(f"Saved → {out2}")

    plt.show()


if __name__ == "__main__":
    main()