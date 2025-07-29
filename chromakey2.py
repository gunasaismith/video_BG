import cv2
import numpy as np
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# === CONFIG ===
INPUT_VIDEO = "C0023.MP4"
REPLACEMENT_BG_VIDEO = "bg.MP4"
OUTPUT_DIR = "output_frames"
BG_FRAMES_DIR = "bg_frames"
OUTPUT_VIDEO = "output_video.mp4"

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BG_FRAMES_DIR, exist_ok=True)

# === Step 1: Extract frames with ffmpeg ===
print("[INFO] Extracting input and background video frames...")
subprocess.run(["ffmpeg", "-y", "-i", INPUT_VIDEO, f"{OUTPUT_DIR}/frame_%04d.png"])
subprocess.run(["ffmpeg", "-y", "-i", REPLACEMENT_BG_VIDEO, f"{BG_FRAMES_DIR}/bg_%04d.png"])

# === Frame lists ===
input_frames = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
bg_frames = sorted([f for f in os.listdir(BG_FRAMES_DIR) if f.endswith(".png")])
frame_count = min(len(input_frames), len(bg_frames))

# === Load reference background ===
ref_bg = cv2.imread(os.path.join(OUTPUT_DIR, input_frames[-1]))

# === Load frame size ===
sample_frame = cv2.imread(os.path.join(OUTPUT_DIR, input_frames[0]))
frame_h, frame_w = sample_frame.shape[:2]

# === Worker function ===
def process_frame(i, input_path, bg_path, ref_bg, output_path):
    frame = cv2.imread(input_path)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (frame_w, frame_h))

    # Background subtraction
    diff = cv2.absdiff(frame, ref_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, person_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)

    # Green spill suppression
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    s_channel = hsv[..., 1].astype(np.float32)
    s_channel[green_mask > 0] *= 0.2
    hsv[..., 1] = np.clip(s_channel, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Composite
    alpha = person_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    composite = (alpha * frame + (1 - alpha) * bg).astype(np.uint8)

    cv2.imwrite(output_path, composite)
    return i

# === Parallel processing ===
print(f"[INFO] Processing {frame_count} frames in parallel...")

with ProcessPoolExecutor() as executor:
    futures = []
    for i in range(frame_count):
        input_path = os.path.join(OUTPUT_DIR, input_frames[i])
        bg_path = os.path.join(BG_FRAMES_DIR, bg_frames[i])
        output_path = os.path.join(OUTPUT_DIR, input_frames[i])
        futures.append(executor.submit(process_frame, i, input_path, bg_path, ref_bg, output_path))

    for f in futures:
        f.result()  # Wait for all to complete

print("[INFO] Frame processing complete.")

# === Step 3: Encode final video ===
print("[INFO] Encoding final video with Apple hardware acceleration...")
subprocess.run([
    "ffmpeg", "-y", "-framerate", "30",
    "-i", f"{OUTPUT_DIR}/frame_%04d.png",
    "-c:v", "h264_videotoolbox",
    "-pix_fmt", "yuv420p",
    OUTPUT_VIDEO
])

print(f"[✅ DONE] Final video saved as: {OUTPUT_VIDEO}")
