import cv2
import numpy as np

# === CONFIG ===
REPLACEMENT_BG_VIDEO = "bg.MP4"
CAM_INDEX = 0  # Webcam index

# === Load background video ===
bg_cap = cv2.VideoCapture(REPLACEMENT_BG_VIDEO)
cap = cv2.VideoCapture(CAM_INDEX)

# === Get webcam properties ===
ret, ref_bg = cap.read()
if not ret:
    print("[ERROR] Could not access webcam.")
    cap.release()
    exit()

height, width = ref_bg.shape[:2]

# === Resize BG frame to match webcam ===
def get_bg_frame():
    global bg_cap
    ret, bg = bg_cap.read()
    if not ret:
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop background video
        ret, bg = bg_cap.read()
    return cv2.resize(bg, (width, height))

print("[INFO] Capturing reference background. Stay out of frame.")
cv2.imshow("Reference BG", ref_bg)
cv2.waitKey(1500)  # Wait for 1.5s to show captured frame
cv2.destroyWindow("Reference BG")

print("[INFO] Starting real-time background replacement. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bg_frame = get_bg_frame()

    # === Chroma subtraction ===
    input_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ref_ycc = cv2.cvtColor(ref_bg, cv2.COLOR_BGR2YCrCb)
    cr_diff = cv2.absdiff(input_ycc[..., 1], ref_ycc[..., 1])
    cb_diff = cv2.absdiff(input_ycc[..., 2], ref_ycc[..., 2])
    chroma_diff = cv2.addWeighted(cr_diff, 0.5, cb_diff, 0.5, 0)

    _, person_mask = cv2.threshold(chroma_diff, 25, 255, cv2.THRESH_BINARY)
    person_mask = cv2.medianBlur(person_mask, 5)
    person_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)

    # === Green spill suppression ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    s_channel = hsv[..., 1].astype(np.float32)
    s_channel[green_mask > 0] *= 0.2
    hsv[..., 1] = np.clip(s_channel, 0, 255).astype(np.uint8)
    cleaned_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # === Alpha blending ===
    alpha = person_mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, 2)
    composite = (alpha * cleaned_frame + (1 - alpha) * bg_frame).astype(np.uint8)

    cv2.imshow("Real-Time BG Removal", composite)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
bg_cap.release()
cv2.destroyAllWindows()
