import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from collections import deque
import json  # use for metadata file
import sys

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    REC_DIR    = Path(sys.argv[1])
    REC_ID     = REC_DIR.name
    BASE       = REC_DIR.parent
    DEPTH_DIR  = REC_DIR / "depth"
    COLOR_DIR  = REC_DIR / "color"
    OUTPUT_CSV = Path(r"C:\Projects\thesis\data\GRID_files") / f"GRID_{REC_ID}.csv"

else:

    BASE     = Path(r"C:\Projects\thesis\data")
    SUBJECT  = "andreas"
    DIST     = "1800"                             # 800, 1200, 1800
    CLOTH    = "hoodie"                           # Tshirt, Hoodie
    REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"
    REC_DIR   = BASE / REC_ID

    DEPTH_DIR = BASE / f"{REC_ID}" / "depth"
    COLOR_DIR = BASE / f"{REC_ID}" / "color"
    OUTPUT_CSV = BASE / "GRID_files" / f"GRID_{REC_ID}.csv"

ROWS, COLS = 7, 4

# Chest ROI shrink factors 
SHRINK_X = 0.08
SHRINK_Y_TOP = 0.05
SHRINK_Y_BOTTOM = 0.30

# Quality thresholds (tune if needed)
MIN_ROI_W = 40   # pixels
MIN_ROI_H = 40   # pixels
MIN_VALID_PIXELS_PER_CELL = 10  # if less, treat as 0

DEBUG_VIZ = False # True to show the visual overlay


# Load depth scale from metadata if available 
# FOR OLD RECORDINGS CANNOT BE USED 
METADATA_PATH = REC_DIR / "metadata.json"
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    DEPTH_SCALE = meta.get("depth_scale", 1.0)
    print(f"Depth scale loaded from metadata: {DEPTH_SCALE}")
else:
    # Set explicitely for old recs
    DEPTH_SCALE = 1.0
    print("[i] No metadata.json found- probably old recording, using depth_scale=1.0")


# Instead of hardcoding START/END frames, it's safer to look at what's actually in the folder
color_files = sorted(glob.glob(os.path.join(COLOR_DIR, "frame_*.jpg")))
# Extract frame numbers from filenames to handle skips
frame_indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in color_files]

# =========================
# HELPERS
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_mean_depth(cell):
    valid = cell[cell > 0]
    if valid.size < MIN_VALID_PIXELS_PER_CELL:
        return 0.0
    return float(np.mean(valid))

# =========================
# INIT MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# =========================
# CHECK PATHS
# =========================
if not os.path.isdir(COLOR_DIR):
    raise FileNotFoundError(f"COLOR_DIR not found: {COLOR_DIR}")
if not os.path.isdir(DEPTH_DIR):
    raise FileNotFoundError(f"DEPTH_DIR not found: {DEPTH_DIR}")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

data_list = []

# Counters for debugging
n_total = 0
n_img_missing = 0
n_pose_fail = 0
n_roi_invalid = 0
n_roi_too_small = 0
n_cell_zero = 0
n_ok = 0

print("🚀 Starting Grid Extraction (7x4)...")

# Smoothing buffer for ROI coordinates
SMOOTH_N = 10  # number of frames to average over
roi_buffer = deque(maxlen=SMOOTH_N)


for i in frame_indices:
    n_total += 1

    color_path = os.path.join(COLOR_DIR, f"frame_{i:05d}.jpg")
    depth_path = os.path.join(DEPTH_DIR, f"frame_{i:05d}.npy")

    # 1. Load Files
    img = cv2.imread(color_path)
    if not os.path.exists(depth_path) or img is None:
        n_img_missing += 1
        continue

    try:
        # Load raw depth and convert to float32 for sub-millimeter precision
        depth_img = np.load(depth_path).astype(np.float32) * DEPTH_SCALE
    except Exception as e:
        print(f"Error loading {depth_path}: {e}")
        continue

    # Depth image shape
    depth_h, depth_w = depth_img.shape[:2]

    # Pose detection on RGB
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        n_pose_fail += 1
        continue

    lm = results.pose_landmarks.landmark

    # Landmarks used:
    # 11 left shoulder, 12 right shoulder, 23 left hip, 24 right hip
    norm_x1 = min(lm[11].x, lm[12].x)
    norm_x2 = max(lm[11].x, lm[12].x)
    norm_y1 = min(lm[11].y, lm[12].y)
    norm_y2 = max(lm[23].y, lm[24].y)

    norm_torso_w = norm_x2 - norm_x1
    norm_torso_h = norm_y2 - norm_y1

    # Chest ROI in normalized coords
    norm_chest_x1 = norm_x1 + (norm_torso_w * SHRINK_X)
    norm_chest_x2 = norm_x2 - (norm_torso_w * SHRINK_X)
    norm_chest_y1 = norm_y1 + (norm_torso_h * SHRINK_Y_TOP)
    norm_chest_y2 = norm_y2 - (norm_torso_h * SHRINK_Y_BOTTOM)

    # Add current ROI to buffer
    roi_buffer.append((norm_chest_x1, norm_chest_x2, norm_chest_y1, norm_chest_y2))

    # Average over buffer
    avg_x1 = sum(r[0] for r in roi_buffer) / len(roi_buffer)
    avg_x2 = sum(r[1] for r in roi_buffer) / len(roi_buffer)
    avg_y1 = sum(r[2] for r in roi_buffer) / len(roi_buffer)
    avg_y2 = sum(r[3] for r in roi_buffer) / len(roi_buffer)

    # Convert smoothed coords to depth pixels
    d_x1 = clamp(int(avg_x1 * depth_w), 0, depth_w - 1)
    d_x2 = clamp(int(avg_x2 * depth_w), 0, depth_w - 1)
    d_y1 = clamp(int(avg_y1 * depth_h), 0, depth_h - 1)
    d_y2 = clamp(int(avg_y2 * depth_h), 0, depth_h - 1)

    # Ensure proper ordering (x1 < x2, y1 < y2)
    if d_x2 <= d_x1 or d_y2 <= d_y1:
        n_roi_invalid += 1
        continue

    roi_w = d_x2 - d_x1
    roi_h = d_y2 - d_y1
    if roi_w < MIN_ROI_W or roi_h < MIN_ROI_H:
        n_roi_too_small += 1
        continue

    depth_chest_roi = depth_img[d_y1:d_y2, d_x1:d_x2]
    if depth_chest_roi.size == 0:
        n_roi_invalid += 1
        continue

    # Grid cell sizes
    cell_h = roi_h // ROWS
    cell_w = roi_w // COLS
    if cell_h <= 0 or cell_w <= 0:
        n_cell_zero += 1
        continue

    # Extract 28 cell averages
    frame_data = {"frame": i}
    cell_idx = 0

    for r in range(ROWS):
        for c in range(COLS):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            cell = depth_chest_roi[y0:y1, x0:x1]
            frame_data[f"cell_{cell_idx}"] = safe_mean_depth(cell)
            cell_idx += 1

    if DEBUG_VIZ:

        # --- DEBUG VISUALIZATION ---
        # Φτιάχνουμε ένα αντίγραφο της εικόνας για να μην χαλάσουμε την πρωτότυπη
        debug_img = img.copy()

        # 1. Σχεδίασε το κεντρικό ROI (Στήθος)
        cv2.rectangle(debug_img, (d_x1, d_y1), (d_x2, d_y2), (0, 255, 0), 2)

        # 2. Σχεδίασε το πλέγμα 7x4
        for r in range(ROWS + 1):
            y = d_y1 + r * cell_h
            cv2.line(debug_img, (d_x1, y), (d_x2, y), (255, 0, 0), 1)
        for c in range(COLS + 1):
            x = d_x1 + c * cell_w
            cv2.line(debug_img, (x, d_y1), (x, d_y2), (255, 0, 0), 1)

        # 3. Εμφάνισε την εικόνα
        cv2.imshow("Verifying Chest Grid", debug_img)
        
        # Πάτα 'q' για να κλείσει το παράθυρο ή οποιοδήποτε πλήκτρο για το επόμενο frame
        if cv2.waitKey(66) & 0xFF == ord('q'):
            DEBUG_VIZ = False          # ← stop showing window
            cv2.destroyAllWindows()    # ← close it


    data_list.append(frame_data)
    n_ok += 1

    if i % 100 == 0:
        print(f"Frame {i}/{len(frame_indices)} processed | kept={n_ok}")

# Save CSV (consistent column order)
df = pd.DataFrame(data_list)
# Ensure all 28 columns exist even if something weird happens
for k in range(ROWS * COLS):
    col = f"cell_{k}"
    if col not in df.columns:
        df[col] = 0.0

cols = ["frame"] + [f"cell_{k}" for k in range(ROWS * COLS)]
df = df[cols]
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Grid Extraction Complete.")
print(f"Saved: {OUTPUT_CSV}")
print("---------- STATS ----------")
print(f"Total frames attempted: {n_total}")
print(f"Kept frames:           {n_ok}")
print(f"Missing images:        {n_img_missing}")
print(f"Pose failed:           {n_pose_fail}")
print(f"ROI invalid:           {n_roi_invalid}")
print(f"ROI too small:         {n_roi_too_small}")
print(f"Cell size zero:        {n_cell_zero}")
print("---------------------------")

cv2.destroyAllWindows()
pose.close()