import cv2
import mediapipe as mp
import numpy as np
import os

# 1. Setup paths for ONE frame (Frame #500 is usually stable)
FRAME_NUM = 500
COLOR_PATH = r"C:\Projects\thesis\data\Sub01_Tshirt_800mm_color\color_{:05d}.jpg".format(FRAME_NUM)
DEPTH_PATH = r"C:\Projects\thesis\data\Sub01_Tshirt_800mm_depth\depth_{:05d}.png".format(FRAME_NUM)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Load images
img = cv2.imread(COLOR_PATH)
depth_img = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

if img is None or depth_img is None:
    print("❌ Error: Could not find images. Check your paths!")
else:
    h, w, _ = img.shape
    dh, dw = depth_img.shape # Should be 576, 640

    # Step 1: Find Normalised Coordinates
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Get Torso Extremes (Normalized 0.0 - 1.0)
        n_x1 = min(lm[11].x, lm[12].x)
        n_x2 = max(lm[11].x, lm[12].x)
        n_y1 = min(lm[11].y, lm[12].y)
        n_y2 = max(lm[23].y, lm[24].y)
        
        n_w = n_x2 - n_x1
        n_h = n_y2 - n_y1

        # Step 2: Shrink the Normalised Coordinates (25% sides, 20% top, 40% bottom)
        n_chest_x1 = n_x1 + (n_w * 0.15)
        n_chest_x2 = n_x2 - (n_w * 0.15)
        n_chest_y1 = n_y1 + (n_h * 0.10)
        n_chest_y2 = n_y2 - (n_h * 0.30)

        # Step 3: Map to Depth Pixels (640x576)
        d_x1, d_x2 = int(n_chest_x1 * dw), int(n_chest_x2 * dw)
        d_y1, d_y2 = int(n_chest_y1 * dh), int(n_chest_y2 * dh)

        # Step 4: Map to Color Pixels for Visual Verification (1920x1080)
        c_x1, c_x2 = int(n_chest_x1 * w), int(n_chest_x2 * w)
        c_y1, c_y2 = int(n_chest_y1 * h), int(n_chest_y2 * h)

        # --- EXTRACT DATA ---
        roi_depth = depth_img[d_y1:d_y2, d_x1:d_x2]
        avg_depth = np.mean(roi_depth[roi_depth > 0]) if roi_depth.size > 0 else 0

        # --- VISUAL FEEDBACK ---
        # Draw the Chest ROI on the color image
        cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 3)
        cv2.putText(img, f"Depth: {avg_depth:.2f}mm", (c_x1, c_y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(f"✅ Success for Frame {FRAME_NUM}")
        print(f"📊 Calculated Chest Depth: {avg_depth:.2f} mm")
        print(f"📍 Depth ROI Pixels: X({d_x1}-{d_x2}) Y({d_y1}-{d_y2})")

        # Show the result
        cv2.imshow("Verification - Green box should be on your chest", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ MediaPipe could not detect a person in this frame.")