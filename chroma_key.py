import cv2
import numpy as np
import os
import subprocess

# === CONFIG ===
INPUT_VIDEO = "C0005.MP4"
REPLACEMENT_BG = "background.jpg"
OUTPUT_DIR = "output_frames"
OUTPUT_VIDEO = "output_video.mp4"

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Open video ===
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === Use last frame as reference background ===
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
ret, ref_bg = cap.read()
if not ret:
    raise Exception("❌ Failed to read last frame as reference background.")
ref_bg = cv2.resize(ref_bg, (frame_w, frame_h))

# Reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Load replacement background
bg_img = cv2.imread(REPLACEMENT_BG)
bg_img = cv2.resize(bg_img, (frame_w, frame_h))

print(f"[INFO] Processing {int(frame_count)} frames...")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Step 1: Background subtraction ===
    diff = cv2.absdiff(frame, ref_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to get person mask
    _, person_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)

    # Create alpha matte
    mask_f = person_mask.astype(np.float32) / 255.0
    alpha = mask_f[..., np.newaxis]  # shape (H, W, 1)

    # === Optional: green spill suppression ===
    # === Optional: green spill suppression ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_pixels = (green_mask > 0)

    s_channel = hsv[..., 1].astype(np.float32)
    s_channel[green_pixels] *= 0.2
    s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)
    hsv[..., 1] = s_channel

    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # === Step 2: Blend person over new background ===
    composite = (alpha * frame + (1 - alpha) * bg_img).astype(np.uint8)

    # Save frame
    out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, composite)
    frame_idx += 1

cap.release()
print("[INFO] Frame processing complete.")

# === Step 3: Encode final video ===
print("[INFO] Encoding final video with ffmpeg...")
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(int(fps)),
    "-i", f"{OUTPUT_DIR}/frame_%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", OUTPUT_VIDEO
])
print(f"[✅ DONE] Final video saved as: {OUTPUT_VIDEO}")
