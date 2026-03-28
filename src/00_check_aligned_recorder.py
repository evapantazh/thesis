import cv2
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path(r"C:\Projects\thesis\data\chrys_1500_hoodie")
COLOR_DIR = DATA_DIR / "color"
DEPTH_DIR = DATA_DIR / "depth"

def verify_alignment():
    # 1. Check if directories exist
    if not COLOR_DIR.exists() or not DEPTH_DIR.exists():
        print(f"Error: Directories not found!\nChecked: {COLOR_DIR}")
        return 

    # 2. Get files and sort them
    color_files = sorted(list(COLOR_DIR.glob("*.jpg")))
    depth_files = sorted(list(DEPTH_DIR.glob("*.npy")))

    print(f"Checking: {COLOR_DIR}")
    print(f"Found {len(color_files)} JPGs and {len(depth_files)} NPYs")

    if len(color_files) == 0:
        print("No files found. Did you record into a different folder?")
        return

    # 3. Create the window explicitly
    window_name = "Verification Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 4. Loop through the 248 frames
    for c_path, d_path in zip(color_files, depth_files):
        color_img = cv2.imread(str(c_path))
        depth_map = np.load(str(d_path))
        
        # Colorize depth (JET Map)
        # We use a fixed range (500-1500mm) so the colors don't flicker
        depth_viz = np.clip(depth_map, 500, 1500) 
        depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # Create Ghost Overlay
        overlay = cv2.addWeighted(color_img, 0.5, depth_viz, 0.5, 0)
        
        # Draw the 7x4 Grid you use for your thesis
        h, w = overlay.shape[:2]
        for i in range(1, 4): # Vertical lines
            cv2.line(overlay, (i * w // 4, 0), (i * w // 4, h), (0, 255, 0), 1)
        for j in range(1, 7): # Horizontal lines
            cv2.line(overlay, (0, j * h // 7), (w, j * h // 7), (0, 255, 0), 1)

        cv2.putText(overlay, f"Frame: {c_path.name}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, overlay)
        
        # WAIT 100ms per frame so you can actually see it
        # Press 'q' to stop the video
        if cv2.waitKey(1) in  [ord('q'), 27]  :  #& 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Playback finished.")

if __name__ == "__main__":
    verify_alignment()