# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2
import numpy as np
import csv
import queue 
import threading
import shutil # Add this
import os     # Add this
from pathlib import Path
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import json


# --- Configuration Constants ---
ESC_KEY = 27
MIN_DEPTH = 20    # Minimum valid depth distance in mm
MAX_DEPTH = 5000 # Maximum valid depth distance in mm 10meters

OUTPUT_DIR = Path(r"C:\Pictures")
DEPTH_DIR = OUTPUT_DIR / "depth"
COLOR_DIR = OUTPUT_DIR / "color"
METADATA_DIR = OUTPUT_DIR / "metadata.json"

# Create folders immediately
DEPTH_DIR.mkdir(parents=True, exist_ok=True)
COLOR_DIR.mkdir(parents=True, exist_ok=True)

MIN_GB_REQUIRED = 8

# Queue implementation
class SaveWorker(threading.Thread):
    def __init__(self, save_queue):
        super().__init__()
        self.save_queue = save_queue
        self.running = True

    def run(self):
        while self.running or not self.save_queue.empty():
            try:
                # Get the "order slip" from the queue
                # Task format: (depth_path, depth_array, color_path, color_array)
                d_path, d_data, c_path, c_img = self.save_queue.get(timeout=1)
                # ΠΡΟΣΘΕΣΕ ΑΥΤΕΣ ΤΙΣ 2 ΓΡΑΜΜΕΣ ΕΔΩ:
                os.makedirs(os.path.dirname(d_path), exist_ok=True)
                os.makedirs(os.path.dirname(c_path), exist_ok=True)
                
                # Do the slow work here
                np.save(d_path, d_data)
                cv2.imwrite(str(c_path), c_img)
                
                self.save_queue.task_done()
            except queue.Empty:
                continue

def verify_storage_safety(path):
    """Checks if there is enough space for a high-quality 2-minute run."""
    abs_path = os.path.abspath(path)
    total, used, free = shutil.disk_usage(abs_path)
    free_gb = free // (2**30)
    
    if free_gb < MIN_GB_REQUIRED:
        print(f"\n[!] WARNING: Low Disk Space ({free_gb} GB available).")
        print(f"    You might not have enough room for a full 2-minute recording.")
        print(f"    Proceed with caution!")
        # We return True anyway because you don't want it to be a hard limit
        return True 
    
    print(f"[*] Storage Safety Check: {free_gb} GB available. (Sufficient for >2 mins)")
    return True

def main():
    window_name = "SyncAlignViewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Initialize the pipeline and configuration objects
    pipeline = Pipeline()
    config = Config()
    
    try:
        # 1. Setup Color Stream Profile
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # We search the list for exactly what we need
            color_profile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 15)
            for i in range(profile_list.get_count()):
                p = profile_list.get_video_stream_profile(i)
                print(f"Color profile {i}: {p.get_width()}x{p.get_height()} @ {p.get_fps()}fps")
        except Exception:
            # Fallback if the specific resolution isn't found
            color_profile = profile_list.get_default_video_stream_profile()
            print("Warning: Requested 1280x720@15fps not found, using default.")
        
        config.enable_stream(color_profile)
        
        # 2. Setup Depth Stream Profile
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            # Depth resolution is usually different from Color (Femto Bolt is 640x576)
            depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 15)
        except Exception:
            depth_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(depth_profile)
        
        
        #Ensure pipeline waits for a full frameset (Color + Depth) before outputting
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception as e:
        print(f"Stream configuration error: {e}")
        return


    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return

    # Enable hardware-level frame synchronization if requested
    pipeline.enable_frame_sync()

    # Initialize the alignment filter. D2C is the most common use case (overlaying depth on RGB).
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    
    # Grab one frame just to read actual delivered resolutions
    test_frames = pipeline.wait_for_frames(1000)
    test_frames = align_filter.process(test_frames).as_frame_set()
    test_color = test_frames.get_color_frame()
    test_depth = test_frames.get_depth_frame()

    metadata = {
        "depth_scale": test_depth.get_depth_scale(),        
        "fps": 15,
        "color_width": test_color.get_width(),
        "color_height": test_color.get_height(),
        "depth_width": test_depth.get_width(),
        "depth_height": test_depth.get_height()
    }

    with open(METADATA_DIR, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved: {metadata}")


    recording = False
    frame_idx = 0
    ts_rows = []

    print("System Ready. Press SPACE to start/stop recording.")

    # Initialize queue
    save_q = queue.Queue()
    worker = SaveWorker(save_q)
    worker.start()

    while True:
        try:
            # Retrieve a frameset with a 100ms timeout
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            # --- Spatial Alignment ---
            # Transforms one stream to the coordinate system/FOV of the other
            frames = align_filter.process(frames)
            if not frames:
                continue

            # Extract frames after alignment
            frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            

            # Get timestamps
            depth_timestamp = depth_frame.get_timestamp()
            # save color timestamp as well to verify the time alignment is done correctly
            color_timestamp = color_frame.get_timestamp()


            # Convert raw color frame to BGR for OpenCV rendering
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert color frame")
                continue
                
            # --- Depth Image Processing ---
            try:
                # Convert raw buffer to 2D numpy array
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
            
            except ValueError:
                print("Failed to reshape depth data")
                continue

            # Visualization 
            # Apply depth scale to get actual distance in mm and filter by range
            depth_data_mm = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
            depth_data_visual = np.where((depth_data_mm > MIN_DEPTH) & (depth_data_mm < MAX_DEPTH), depth_data_mm, 0)
           
            # Normalize and colormap for visualization
            depth_data_visual = cv2.normalize(depth_data_visual, None, 0, 255, cv2.NORM_MINMAX)
            depth_data_visual = cv2.applyColorMap(depth_data_visual.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blended Visualization 
            # Alpha-blending of Color and Depth images to check alignment accuracy
            overlay_image = cv2.addWeighted(color_image, 0.5, depth_data_visual, 0.5, 0)

            # Recording 
            if recording:
                d_file = DEPTH_DIR / f"frame_{frame_idx:05d}.npy"
                c_file = COLOR_DIR / f"frame_{frame_idx:05d}.jpg"

                # Put the "order" in the queue and keep going!
                save_q.put((d_file, depth_data, c_file, color_image))

                ts_rows.append([frame_idx, depth_timestamp, color_timestamp, d_file.name, c_file.name])

                # Saving logic goes here 
                cv2.circle(overlay_image, (30, 30), 10, (0, 0, 255), -1)

                if len(ts_rows) > 1:
                    gap = depth_timestamp - ts_rows[-2][1]
                    if gap > 100:
                        print(f"[!] Possible frame drop at frame {frame_idx} | gap: {gap:.1f} ms")

                frame_idx += 1
    

            cv2.imshow(window_name, overlay_image)

            # Handle Keyboard Input
            key = cv2.waitKey(1)
            if key in [ord('q'), ESC_KEY]:
                # Final safety save: if user quits while recording is active
                if recording and ts_rows:
                    with open(OUTPUT_DIR / "timestamps.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["frame", "depth_timestamp", "color_timestamp", "depth_file", "color_file"])
                        writer.writerows(ts_rows)
                break
            elif key == ord(' '):
                recording = not recording
                if recording:
                    # Check if files already exist from a previous recording
                    existing = list(DEPTH_DIR.glob("*.npy"))
                    if existing:
                        print(f"[!] WARNING: {len(existing)} depth files already exist in {DEPTH_DIR}")
                        print(f"    They will be overwritten if you continue.")
                    print("Started recording...")
                    frame_idx = 0
                    ts_rows = []
                else:
                    print(f"Stopped. Saving {len(ts_rows)} frames to CSV...")
                    if ts_rows:
                        with open(OUTPUT_DIR / "timestamps.csv", "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["frame", "depth_timestamp", "color_timestamp", "depth_file", "color_file"])
                            writer.writerows(ts_rows)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime error: {e}")
            continue
        
    # Clean up resources
    worker.running = False
    worker.join()
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check the actual output directory you defined
    verify_storage_safety(OUTPUT_DIR)
    main()