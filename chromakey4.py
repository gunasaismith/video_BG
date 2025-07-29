import cv2
import numpy as np
import os

# === CONFIG ===
INPUT_VIDEO = "C0005.MP4"
REPLACEMENT_BG_VIDEO = "bg.MP4"
OUTPUT_VIDEO = "output_final.mp4"
MAX_FRAMES = 500

# === Step 1: Prepare video readers ===
input_cap = cv2.VideoCapture(INPUT_VIDEO)
bg_cap = cv2.VideoCapture(REPLACEMENT_BG_VIDEO)

width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_cap.get(cv2.CAP_PROP_FPS)

# === Read reference background (last frame of input video) ===
print("[INFO] Reading reference background (last frame)...")
ref_bg = None
while True:
    ret, frame = input_cap.read()
    if not ret:
        break
    ref_bg = frame.copy()

# Restart video for processing
input_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# === Setup output writer ===
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print("[INFO] Processing frames...")

frame_idx = 0
while frame_idx < MAX_FRAMES:
    ret1, input_frame = input_cap.read()
    ret2, bg_frame = bg_cap.read()

    if not ret1 or not ret2:
        break

    # Resize bg
    bg_frame = cv2.resize(bg_frame, (width, height))

    # === CHROMA-based subtraction (light-robust) ===
    input_ycc = cv2.cvtColor(input_frame, cv2.COLOR_BGR2YCrCb)
    ref_ycc = cv2.cvtColor(ref_bg, cv2.COLOR_BGR2YCrCb)
    cr_diff = cv2.absdiff(input_ycc[..., 1], ref_ycc[..., 1])
    cb_diff = cv2.absdiff(input_ycc[..., 2], ref_ycc[..., 2])
    chroma_diff = cv2.addWeighted(cr_diff, 0.5, cb_diff, 0.5, 0)

    # Create binary person mask
    _, person_mask = cv2.threshold(chroma_diff, 25, 255, cv2.THRESH_BINARY)
    person_mask = cv2.medianBlur(person_mask, 5)

    # === Green spill suppression ===
    hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    s_channel = hsv[..., 1].astype(np.float32)
    s_channel[green_mask > 0] *= 0.2
    hsv[..., 1] = np.clip(s_channel, 0, 255).astype(np.uint8)
    cleaned_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # === Feather mask edges ===
    person_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)

    # === Alpha blending ===
    alpha = person_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, 2)
    composite = (alpha * cleaned_frame + (1 - alpha) * bg_frame).astype(np.uint8)

    out.write(composite)
    frame_idx += 1

print(f"[✅ DONE] Saved final video to: {OUTPUT_VIDEO}")

input_cap.release()
bg_cap.release()
out.release()
