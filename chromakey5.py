import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# === CONFIG ===
INPUT_VIDEO = "C0027.MP4"
REPLACEMENT_BG_VIDEO = "bg.MP4"
OUTPUT_VIDEO = "output_final_parallel.mp4"
MAX_FRAMES = 500

# === GLOBAL DATA (shared for workers) ===
ref_bg = None
bg_frames = []
input_frames = []
width = height = fps = None

def init_video():
    global ref_bg, bg_frames, input_frames, width, height, fps

    # Read input video
    input_cap = cv2.VideoCapture(INPUT_VIDEO)
    bg_cap = cv2.VideoCapture(REPLACEMENT_BG_VIDEO)

    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_cap.get(cv2.CAP_PROP_FPS)

    # Get last frame as reference
    last_frame_idx = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    input_cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
    ret, ref_bg = input_cap.read()
    if not ret:
        raise ValueError("❌ Failed to read last frame")

    # Load frames into memory
    input_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    input_frames = []
    bg_frames = []

    for _ in range(MAX_FRAMES):
        ret1, f1 = input_cap.read()
        ret2, f2 = bg_cap.read()
        if not ret1 or not ret2:
            break
        input_frames.append(f1)
        bg_frames.append(cv2.resize(f2, (width, height)))

    input_cap.release()
    bg_cap.release()

    return input_frames, bg_frames, ref_bg, width, height, fps


def process_frame(args):
    idx, input_frame, bg_frame, ref_bg = args

    # === Chroma diff ===
    input_ycc = cv2.cvtColor(input_frame, cv2.COLOR_BGR2YCrCb)
    ref_ycc = cv2.cvtColor(ref_bg, cv2.COLOR_BGR2YCrCb)
    cr_diff = cv2.absdiff(input_ycc[..., 1], ref_ycc[..., 1])
    cb_diff = cv2.absdiff(input_ycc[..., 2], ref_ycc[..., 2])
    chroma_diff = cv2.addWeighted(cr_diff, 0.5, cb_diff, 0.5, 0)

    _, person_mask = cv2.threshold(chroma_diff, 25, 255, cv2.THRESH_BINARY)
    person_mask = cv2.medianBlur(person_mask, 5)

    # === Green spill removal ===
    hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    s_channel = hsv[..., 1].astype(np.float32)
    s_channel[green_mask > 0] *= 0.2
    hsv[..., 1] = np.clip(s_channel, 0, 255).astype(np.uint8)
    cleaned_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # === Feather mask ===
    person_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)

    # === Alpha blend ===
    alpha = person_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, 2)
    composite = (alpha * cleaned_frame + (1 - alpha) * bg_frame).astype(np.uint8)

    return idx, composite


if __name__ == "__main__":
    print("[INFO] Initializing and preloading video frames...")
    input_frames, bg_frames, ref_bg, width, height, fps = init_video()

    print(f"[INFO] Processing {len(input_frames)} frames using {cpu_count()} cores...")

    with Pool(processes=cpu_count()) as pool:
        args_list = [(i, input_frames[i], bg_frames[i], ref_bg) for i in range(len(input_frames))]
        results = pool.map(process_frame, args_list)

    print("[INFO] Writing to output video...")

    # Sort results to maintain frame order
    results.sort(key=lambda x: x[0])
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    for _, frame in results:
        out.write(frame)
    out.release()

    print(f"[✅ DONE] Saved final video to: {OUTPUT_VIDEO}")

