import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
# =========================
# CONFIG
# =========================

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Projects\thesis\data")
SUBJECT  = "Sub01"
DIST     = "500"
CLOTH    = "Tshirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"


DEPTH_DIR = BASE / f"Camera_{REC_ID}" / "depth"
COLOR_DIR = BASE / f"Camera_{REC_ID}" / "color"
OUTPUT_CSV = BASE / f"GRID_{REC_ID}.csv"

ROWS, COLS = 7, 4

# Chest ROI shrink factors (your chosen values)
SHRINK_X = 0.08
SHRINK_Y_TOP = 0.05
SHRINK_Y_BOTTOM = 0.30

# Quality thresholds (tune if needed)
MIN_ROI_W = 40   # pixels
MIN_ROI_H = 40   # pixels
MIN_VALID_PIXELS_PER_CELL = 10  # if less, treat as 0

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
        depth_img = np.load(depth_path).astype(np.float32) 
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

    # Convert to depth pixel coords + clamp to image bounds
    d_x1 = clamp(int(norm_chest_x1 * depth_w), 0, depth_w - 1)
    d_x2 = clamp(int(norm_chest_x2 * depth_w), 0, depth_w - 1)
    d_y1 = clamp(int(norm_chest_y1 * depth_h), 0, depth_h - 1)
    d_y2 = clamp(int(norm_chest_y2 * depth_h), 0, depth_h - 1)

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



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
