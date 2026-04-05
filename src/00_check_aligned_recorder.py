import pandas as pd
import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Pictures")
COLOR_DIR = DATA_DIR / "color"
DEPTH_DIR = DATA_DIR / "depth"
TIMESTAMPS_DIR = DATA_DIR / "timestamps.csv"


def verify_alignment():
    # 1. Check if directories exist
    if not COLOR_DIR.exists() or not DEPTH_DIR.exists():
        print(f"Error: Directories not found!\nChecked: {COLOR_DIR}")
        return 

    # 2. Get files and sort them
    color_files = sorted(list(COLOR_DIR.glob("*.jpg")))
    depth_files = sorted(list(DEPTH_DIR.glob("*.npy")))

    print(f"Checking: {COLOR_DIR}")

    if len(color_files) == 0:
        print("No files found. Did you record into a different folder?")
        return
    
    print(f"Found {len(color_files)} JPGs and {len(depth_files)} NPYs")

    if len(color_files) != len(depth_files):
        print(f"[!] WARNING: {len(color_files)} color files but {len(depth_files)} depth files — mismatch!")
        return

    if TIMESTAMPS_DIR.exists():
        df = pd.read_csv(TIMESTAMPS_DIR)

        # Handle both old (timestamp) and new (depth_timestamp) column names
        ts_col = "depth_timestamp" if "depth_timestamp" in df.columns else "timestamp"
        df['gap'] = df[ts_col].diff()

        print(f"Total frames: {len(df)}")
        print(f"Average FPS: {1000 / df['gap'].mean():.1f}")
        print(f"Frame drops (gap > 100ms): {(df['gap'] > 100).sum()}")

        # Sync gap only available in new recordings
        if "color_timestamp" in df.columns:
            df['sync_gap'] = abs(df['depth_timestamp'] - df['color_timestamp'])
            print(f"Mean sync gap color-depth: {df['sync_gap'].mean():.2f} ms")
        else:
            print("[i] Single timestamp format — sync gap not available")
    else:
        print("[!] No timestamps.csv found")


    # 3. Create the window explicitly
    window_name = "Verification Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 4. Loop through the 248 frames
    for c_path, d_path in zip(color_files, depth_files):
        color_img = cv2.imread(str(c_path))
        depth_map = np.load(str(d_path))
        if c_path == color_files[0]:
            print(f"Color shape: {color_img.shape}")
            print(f"Depth shape: {depth_map.shape}")
            print(f"Depth min/max: {depth_map.min()} / {depth_map.max()}")
            print(f"Depth dtype: {depth_map.dtype}")

        # Colorize depth (JET Map)
        # We use a fixed range (500-2000mm) so the colors don't flicker
        depth_viz = np.clip(depth_map, 20, 2000) 
        depth_viz = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # Create Ghost Overlay
        overlay = cv2.addWeighted(color_img, 0.5, depth_viz, 0.5, 0)
        cv2.putText(overlay, f"Frame: {c_path.stem} / {len(color_files)}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, overlay)
        # WAIT 66ms (15 fps) per frame so you can actually see it
        # Press 'q' to stop the video
        if cv2.waitKey(66) in  [ord('q'), 27]  :  #& 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Playback finished.")

if __name__ == "__main__":
    verify_alignment()