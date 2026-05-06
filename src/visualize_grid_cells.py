"""
visualize_grid_cells.py
=======================
Loads one color + depth frame, runs MediaPipe pose,
draws the 7x4 grid with cell numbers labeled,
and highlights the proposed sternum cells.

Run this once to confirm which cells cover the sternum
before using them for tap detection.

Press any key to advance to the next frame, Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from collections import deque

# ─────────────────────────────────────────────────────────────
#  RECORDING  — edit these
# ─────────────────────────────────────────────────────────────
SUBJECT = "AGE"
DIST    = "800"
CLOTH   = "tshirt"

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
REC_BASE  = Path(r"D:\recordings")
REC_ID    = f"{SUBJECT}_{DIST}_{CLOTH}"
REC_DIR   = REC_BASE / REC_ID
COLOR_DIR = REC_DIR / "color"
DEPTH_DIR = REC_DIR / "depth"

# ─────────────────────────────────────────────────────────────
#  GRID CONFIG  (must match extract_grid.py)
# ─────────────────────────────────────────────────────────────
ROWS, COLS   = 7, 4
SHRINK_X     = 0.08
SHRINK_Y_TOP = 0.05
SHRINK_Y_BOTTOM = 0.30

MIN_ROI_W = 40
MIN_ROI_H = 40
MIN_VALID_PIXELS_PER_CELL = 10
SMOOTH_N  = 10

# ─────────────────────────────────────────────────────────────
#  STERNUM CELLS  (proposed — this is what we are verifying)
# ─────────────────────────────────────────────────────────────
#  Grid layout (row, col) -> cell index = row*4 + col
#
#        col0  col1  col2  col3
#  row0 [  0    1    2    3  ]  top chest
#  row1 [  4    5    6    7  ]
#  row2 [  8    9   10   11  ]
#  row3 [ 12   13   14   15  ]  sternum centre
#  row4 [ 16   17   18   19  ]
#  row5 [ 20   21   22   23  ]
#  row6 [ 24   25   26   27  ]  lower chest
#
STERNUM_CELLS = [5, 6, 9, 10, 13, 14]   # centre cols (1,2), rows 2-4

# How many frames to show (spread across recording)
N_FRAMES_TO_SHOW = 20

# ─────────────────────────────────────────────────────────────
#  HELPERS  (same as extract_grid.py)
# ─────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_mean_depth(cell):
    valid = cell[cell > 0]
    if valid.size < MIN_VALID_PIXELS_PER_CELL:
        return 0.0
    return float(np.mean(valid))


# ─────────────────────────────────────────────────────────────
#  LOAD DEPTH SCALE
# ─────────────────────────────────────────────────────────────
METADATA_PATH = REC_DIR / "metadata.json"
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    DEPTH_SCALE = meta.get("depth_scale", 1.0)
    print(f"Depth scale from metadata: {DEPTH_SCALE}")
else:
    DEPTH_SCALE = 1.0
    print("[i] No metadata.json — using depth_scale=1.0")


# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)


# ─────────────────────────────────────────────────────────────
#  GET FRAME LIST
# ─────────────────────────────────────────────────────────────
color_files = sorted(COLOR_DIR.glob("frame_*.jpg"),
                     key=lambda p: int(p.stem.split("_")[1]))
if not color_files:
    raise FileNotFoundError(f"No color frames in {COLOR_DIR}")

# Pick N frames spread across the recording
total = len(color_files)
step  = max(1, total // N_FRAMES_TO_SHOW)
selected = color_files[::step][:N_FRAMES_TO_SHOW]

print(f"Found {total} color frames — showing {len(selected)} spread evenly")
print(f"Proposed sternum cells: {STERNUM_CELLS}")
print()
print("Controls:  any key = next frame   Q = quit")
print()

# ─────────────────────────────────────────────────────────────
#  DRAW FUNCTION
# ─────────────────────────────────────────────────────────────

def draw_grid_overlay(img, depth_img, lm):
    """
    Computes the chest ROI and draws the 7x4 grid on img.
    Sternum cells highlighted in green, others in blue.
    Returns annotated image and list of cell mean depths.
    """
    h_img, w_img = img.shape[:2]
    depth_h, depth_w = depth_img.shape[:2]

    # Torso bounding box from landmarks
    norm_x1 = min(lm[11].x, lm[12].x)
    norm_x2 = max(lm[11].x, lm[12].x)
    norm_y1 = min(lm[11].y, lm[12].y)
    norm_y2 = max(lm[23].y, lm[24].y)

    norm_torso_w = norm_x2 - norm_x1
    norm_torso_h = norm_y2 - norm_y1

    norm_cx1 = norm_x1 + norm_torso_w * SHRINK_X
    norm_cx2 = norm_x2 - norm_torso_w * SHRINK_X
    norm_cy1 = norm_y1 + norm_torso_h * SHRINK_Y_TOP
    norm_cy2 = norm_y2 - norm_torso_h * SHRINK_Y_BOTTOM

    # Convert to image pixels for drawing overlay on color image
    ix1 = clamp(int(norm_cx1 * w_img), 0, w_img - 1)
    ix2 = clamp(int(norm_cx2 * w_img), 0, w_img - 1)
    iy1 = clamp(int(norm_cy1 * h_img), 0, h_img - 1)
    iy2 = clamp(int(norm_cy2 * h_img), 0, h_img - 1)

    # Convert to depth pixels for cell extraction
    dx1 = clamp(int(norm_cx1 * depth_w), 0, depth_w - 1)
    dx2 = clamp(int(norm_cx2 * depth_w), 0, depth_w - 1)
    dy1 = clamp(int(norm_cy1 * depth_h), 0, depth_h - 1)
    dy2 = clamp(int(norm_cy2 * depth_h), 0, depth_h - 1)

    if dx2 <= dx1 or dy2 <= dy1:
        cv2.putText(img, "ROI invalid", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img, []

    roi_w = ix2 - ix1
    roi_h = iy2 - iy1
    cell_h_px = roi_h // ROWS
    cell_w_px = roi_w // COLS

    depth_roi = depth_img[dy1:dy2, dx1:dx2]
    d_cell_h  = (dy2 - dy1) // ROWS
    d_cell_w  = (dx2 - dx1) // COLS

    cell_depths = []
    out = img.copy()

    for r in range(ROWS):
        for c in range(COLS):
            cell_idx = r * COLS + c

            # Pixel coords for drawing on color image
            x0 = ix1 + c * cell_w_px
            y0 = iy1 + r * cell_h_px
            x1 = ix1 + (c + 1) * cell_w_px
            y1 = iy1 + (r + 1) * cell_h_px

            # Depth value for this cell
            dy0 = r * d_cell_h
            dy1_ = (r + 1) * d_cell_h
            dx0 = c * d_cell_w
            dx1_ = (c + 1) * d_cell_w
            cell_depth = safe_mean_depth(depth_roi[dy0:dy1_, dx0:dx1_])
            cell_depths.append(cell_depth)

            # Color: green for sternum, blue for others
            is_sternum = cell_idx in STERNUM_CELLS
            color      = (0, 220, 0) if is_sternum else (220, 100, 0)
            thickness  = 2 if is_sternum else 1

            # Semi-transparent fill for sternum cells
            if is_sternum:
                overlay = out.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 0), -1)
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

            # Cell border
            cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)

            # Cell number label
            cx = (x0 + x1) // 2 - 10
            cy = (y0 + y1) // 2 + 6
            cv2.putText(out, str(cell_idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Depth value (mm) below cell number
            if cell_depth > 0:
                cv2.putText(out, f"{cell_depth:.0f}", (cx - 8, cy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            (255, 255, 255), 1)

    return out, cell_depths


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────

cv2.namedWindow("Grid Cell Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Grid Cell Viewer", 960, 600)

roi_buffer = deque(maxlen=SMOOTH_N)

for color_path in selected:
    frame_id   = int(color_path.stem.split("_")[1])
    depth_path = DEPTH_DIR / f"frame_{frame_id:05d}.npy"

    img = cv2.imread(str(color_path))
    if img is None:
        print(f"  Cannot load {color_path.name}")
        continue

    if not depth_path.exists():
        print(f"  No depth for frame {frame_id}")
        continue

    depth_img = np.load(str(depth_path)).astype(np.float32) * DEPTH_SCALE

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print(f"  Frame {frame_id}: pose not detected")
        continue

    lm = results.pose_landmarks.landmark

    annotated, cell_depths = draw_grid_overlay(img, depth_img, lm)

    # Header bar
    header = annotated.copy()
    cv2.rectangle(header, (0, 0), (annotated.shape[1], 36), (20, 20, 20), -1)
    cv2.addWeighted(header, 0.7, annotated, 0.3, 0, annotated)
    cv2.putText(annotated,
                f"Frame {frame_id}   |   GREEN = sternum cells {STERNUM_CELLS}   "
                f"|   any key = next   Q = quit",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Print sternum cell depths to terminal
    sternum_vals = [cell_depths[i] for i in STERNUM_CELLS if i < len(cell_depths)]
    print(f"Frame {frame_id:5d} | sternum cell depths: "
          + "  ".join(f"cell_{i}={cell_depths[i]:.1f}" for i in STERNUM_CELLS
                      if i < len(cell_depths)))

    cv2.imshow("Grid Cell Viewer", annotated)
    cv2.waitKey(1)
    key = cv2.waitKey(0) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cv2.destroyAllWindows()
pose.close()
print("\nDone. If sternum cells look correct, use STERNUM_CELLS = [9,10,13,14,17,18]")
print("in the tap detector. Adjust the list if needed based on what you saw.")