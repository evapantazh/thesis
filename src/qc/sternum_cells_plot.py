"""
visualize_sternum_grid.py
==========================
Draws the 7x4 chest ROI grid on both the color frame and the
corresponding depth frame, highlighting the sternum cells in green.
Overlays real per-cell depth values loaded from
GRID_files/GRID_{REC_ID}.csv — matching Figure 3.3 style.
"""
import matplotlib.cm as cm
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  RECORDING — edit these
# ─────────────────────────────────────────────────────────────
SUBJECT  = "AVE"
DIST     = "1800"
CLOTH    = "tshirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"
FRAME_ID = 0          # which frame to visualize (must exist in the grid CSV)

BASE      = Path(r"D:\recordings")
COLOR_DIR = BASE / REC_ID / "color"
DEPTH_DIR = BASE / REC_ID / "depth"
PATH_COLOR_FRAME = COLOR_DIR / f"frame_{FRAME_ID:05d}.jpg"
PATH_DEPTH_FRAME = DEPTH_DIR / f"frame_{FRAME_ID:05d}.npy"

PATH_GRID_CSV = Path(r"C:\Projects\thesis\data\GRID_files") / f"GRID_{REC_ID}.csv"

OUTDIR = Path(r"C:\Projects\thesis\thesis_latex\images")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_COLOR_PATH = OUTDIR / "fig_sternum_grid_color.png"
OUT_DEPTH_PATH = OUTDIR / "fig_sternum_grid_depth.png"

# ─────────────────────────────────────────────────────────────
#  GRID / ROI CONFIG — same as extract_grid.py
# ─────────────────────────────────────────────────────────────
ROWS, COLS = 7, 4
SHRINK_X        = 0.08
SHRINK_Y_TOP    = 0.05
SHRINK_Y_BOTTOM = 0.30

STERNUM_CELLS = [5, 6, 9, 10, 13, 14]

GRID_COLOR      = (255, 120, 0)    # blue lines (BGR)
STERNUM_COLOR   = (0, 200, 0)      # green box (BGR)
TEXT_COLOR      = (255, 255, 255)  # white
STERNUM_TEXT_COLOR = (0, 255, 0)   # green

SHOW_VALUES = True   # overlay per-cell depth (mm) from the grid CSV


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def load_cell_values(csv_path, frame_id, n_cells):
    if not csv_path.exists():
        print(f"[WARN] Grid CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    row = df[df["frame"] == frame_id]
    if row.empty:
        print(f"[WARN] Frame {frame_id} not found in {csv_path.name}")
        return None
    row = row.iloc[0]
    return {i: row[f"cell_{i}"] for i in range(n_cells) if f"cell_{i}" in row}


def depth_to_color(depth_img, cmap_name="nipy_spectral"):
    """Convert a raw depth map (mm, float32) to a vivid colorized BGR image
    using a matplotlib colormap."""
    valid = depth_img[depth_img > 0]
    if valid.size == 0:
        return np.zeros((*depth_img.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [1, 99])
    clipped = np.clip(depth_img, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-6)   # 0..1 float

    cmap = cm.get_cmap(cmap_name)
    colored_rgba = cmap(norm)                   # H x W x 4, float 0..1
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

    colored_bgr[depth_img <= 0] = (0, 0, 0)      # invalid pixels black
    return colored_bgr


def draw_grid(img, x1, y1, x2, y2, rows, cols, sternum_cells, values=None):
    roi_w = x2 - x1
    roi_h = y2 - y1
    cell_w = roi_w / cols
    cell_h = roi_h / rows

    out = img.copy()

    # Scale font size and offsets relative to cell height so text never
    # overlaps between rows, regardless of subject distance/ROI size.
    font_scale_idx = clamp(cell_h / 90.0, 0.18, 0.40)
    font_scale_val = clamp(cell_h / 100.0, 0.16, 0.35)
    offset_idx = max(int(cell_h * 0.32), 9)
    offset_val = max(int(cell_h * 0.68), 20)

    for cell_idx in range(rows * cols):
        row = cell_idx // cols
        col = cell_idx % cols

        cx0 = int(x1 + col * cell_w)
        cy0 = int(y1 + row * cell_h)
        cx1 = int(x1 + (col + 1) * cell_w)
        cy1 = int(y1 + (row + 1) * cell_h)

        is_sternum = cell_idx in sternum_cells
        color = STERNUM_COLOR if is_sternum else GRID_COLOR
        thickness = 2 if is_sternum else 1
        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), color, thickness)

        text_color = STERNUM_TEXT_COLOR if is_sternum else TEXT_COLOR
        cv2.putText(out, str(cell_idx), (cx0 + 2, cy0 + offset_idx),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_idx, text_color, 1, cv2.LINE_AA)

        if values is not None and cell_idx in values:
            val = values[cell_idx]
            cv2.putText(out, f"{val:.0f}", (cx0 + 2, cy0 + offset_val),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_val, text_color, 1, cv2.LINE_AA)

    # Outer ROI box
    cv2.rectangle(out, (x1, y1), (x2, y2), GRID_COLOR, 1)
    return out

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.7,
        model_complexity=1
    )

    color_img = cv2.imread(str(PATH_COLOR_FRAME))
    if color_img is None:
        raise FileNotFoundError(f"Could not load color frame: {PATH_COLOR_FRAME}")

    if not PATH_DEPTH_FRAME.exists():
        raise FileNotFoundError(f"Could not load depth frame: {PATH_DEPTH_FRAME}")
    depth_raw = np.load(PATH_DEPTH_FRAME).astype(np.float32)

    # ROI is detected on the color frame (pose model needs RGB), then
    # reused for the depth frame — this matches extract_grid.py, where
    # ROI coords in normalized [0,1] space are applied to depth pixel size.
    x1_c, y1_c, x2_c, y2_c = get_chest_roi(color_img, pose)

    values = None
    if SHOW_VALUES:
        values = load_cell_values(PATH_GRID_CSV, FRAME_ID, ROWS * COLS)

    # ── Color figure ──────────────────────────────────────────
    out_color = draw_grid(color_img, x1_c, y1_c, x2_c, y2_c, ROWS, COLS,
                          STERNUM_CELLS, values)
    cv2.imwrite(str(OUT_COLOR_PATH), out_color)
    print(f"Saved → {OUT_COLOR_PATH}")

    # ── Depth figure ──────────────────────────────────────────
    # Rescale ROI from color-frame pixel space to depth-frame pixel space,
    # since color and depth images may have different resolutions.
    ch, cw = color_img.shape[:2]
    dh, dw = depth_raw.shape[:2]
    scale_x = dw / cw
    scale_y = dh / ch

    x1_d = clamp(int(x1_c * scale_x), 0, dw - 1)
    x2_d = clamp(int(x2_c * scale_x), 0, dw - 1)
    y1_d = clamp(int(y1_c * scale_y), 0, dh - 1)
    y2_d = clamp(int(y2_c * scale_y), 0, dh - 1)

    depth_vis = depth_to_color(depth_raw, cmap_name="plasma")   # standard scientific depth colormap           # classic blue→cyan→yellow→red
    out_depth = draw_grid(depth_vis, x1_d, y1_d, x2_d, y2_d, ROWS, COLS,
                          STERNUM_CELLS, values)
    cv2.imwrite(str(OUT_DEPTH_PATH), out_depth)
    print(f"Saved → {OUT_DEPTH_PATH}")

    cv2.imshow("Color grid", out_color)
    cv2.imshow("Depth grid", out_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()