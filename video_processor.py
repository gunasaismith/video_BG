import cv2
import numpy as np
import os


def process_video(
    input_path,
    bg_path,
    output_path="static/output_final.mp4",
    person_scale=0.81,          # 1.0 = fit tallest side; <1.0 shrinks further
    h_align="center",          # "left", "center", "right"
    v_align="bottom"           # "top", "center", "bottom"
):
    """
    Composites a portrait‑oriented green‑screen video (input_path) over a
    background video (bg_path).  The background keeps its native resolution;
    the person is scaled down to fit and positioned by alignment parameters.
    """

    # Load Haar cascade for face detection
    # Try to find the cascade in common locations
    possible_haar_paths = [
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        './haarcascade_frontalface_default.xml',
    ]
    haar_path = None
    for p in possible_haar_paths:
        if os.path.exists(p):
            haar_path = p
            break
    if haar_path is None:

        
        raise RuntimeError("Haar cascade XML not found. Please install OpenCV data files.")
    face_cascade = cv2.CascadeClassifier(haar_path)

    # ─────────────────────────── Open videos ──────────────────────────────
    fg_cap = cv2.VideoCapture(input_path)      # foreground / person
    bg_cap = cv2.VideoCapture(bg_path)         # background (keeps full res)

    if not fg_cap.isOpened() or not bg_cap.isOpened():
        raise RuntimeError("Unable to open input or background video.")

    # Foreground properties
    fg_w  = int(fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fg_h  = int(fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fg_fps = fg_cap.get(cv2.CAP_PROP_FPS) or 30

    # Background properties
    bg_w  = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg_h  = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg_fps = bg_cap.get(cv2.CAP_PROP_FPS) or fg_fps

    # ─────────────────────── Get last frame as reference ──────────────────
    ref_bg = None
    while True:
        ok, frame = fg_cap.read()
        if not ok:
            break
        ref_bg = frame.copy()                  # ends with the *last* good frame
    if ref_bg is None:
        raise RuntimeError("Foreground video is empty.")

    ref_ycc = cv2.cvtColor(ref_bg, cv2.COLOR_BGR2YCrCb)
    fg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # rewind foreground

    # ───────────────────────── VideoWriter setup ──────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fg_fps, (bg_w, bg_h))

    # Match background pace to foreground fps
    bg_frame_interval = max(1, int(round(bg_fps / fg_fps)))

    # ────────────────────────── Processing loop ───────────────────────────
    buffer_size = int(fg_fps * 0.25)
    frame_buffer = []

    while True:
        fg_ok, fg = fg_cap.read()
        if not fg_ok:
            # Write any remaining buffered frames
            for buffered_bg in frame_buffer:
                out.write(buffered_bg)
            break

        # -------- read/loop background frames to keep fps in sync ----------
        for _ in range(bg_frame_interval):
            bg_ok, bg = bg_cap.read()
            if not bg_ok:
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                bg_ok, bg = bg_cap.read()
        # -------------------------------------------------------------------

        # ------------- quick green‑screen / chroma difference --------------
        fg_ycc = cv2.cvtColor(fg, cv2.COLOR_BGR2YCrCb)
        cr_diff = cv2.absdiff(fg_ycc[..., 1], ref_ycc[..., 1])
        cb_diff = cv2.absdiff(fg_ycc[..., 2], ref_ycc[..., 2])
        chroma_diff = cv2.addWeighted(cr_diff, 0.5, cb_diff, 0.5, 0)
        _, person_mask = cv2.threshold(chroma_diff, 25, 255, cv2.THRESH_BINARY)
        person_mask = cv2.medianBlur(person_mask, 5)

        # --------------------- green‑spill suppression ---------------------
        hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([35, 40, 40], dtype=np.uint8),
            np.array([85, 255, 255], dtype=np.uint8)
        )
        s = hsv[..., 1].astype(np.float32)
        s[green_mask > 0] *= 0.2
        hsv[..., 1] = np.clip(s, 0, 255).astype(np.uint8)
        fg_clean = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        person_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)
        # -------------------------------------------------------------------

        # ---------------------- resize person to fit -----------------------
        scale_fit = min(bg_h / fg_h, bg_w / fg_w)
        scale = scale_fit * person_scale
        new_w, new_h = int(fg_w * scale), int(fg_h * scale)

        fg_small   = cv2.resize(fg_clean,  (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(person_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ---------------------- add black tint to foreground ----------------------
        # Blend fg_small with black to reduce glow (15% black)
        black = np.zeros_like(fg_small)
        tint_strength = 0.15  # 15% black
        fg_small = cv2.addWeighted(fg_small, 1 - tint_strength, black, tint_strength, 0)

        # ----------------------- alignment offsets -------------------------
        h_map = {"left": 0, "center": (bg_w - new_w) // 2, "right": bg_w - new_w}
        v_map = {"top": 0, "center": (bg_h - new_h) // 2, "bottom": bg_h - new_h}
        x0 = h_map.get(h_align, h_map["center"]) + 100  # Shift 50 pixels to the right
        y0 = v_map.get(v_align, v_map["bottom"])       # Keep vertical alignment as is

        # Optional safety check to stay within bounds
        x0 = min(x0, bg_w - new_w)


        # ---------------------- alpha‑blend composite ----------------------
        alpha = (mask_small.astype(np.float32) / 255.0)[..., None]
        roi_bg = bg[y0:y0+new_h, x0:x0+new_w]
        comp_roi = (alpha * fg_small + (1 - alpha) * roi_bg).astype(np.uint8)
        bg[y0:y0+new_h, x0:x0+new_w] = comp_roi

        # Buffer the frame instead of writing immediately
        frame_buffer.append(bg.copy())
        if len(frame_buffer) > buffer_size:
            # Write the oldest frame in the buffer
            out.write(frame_buffer.pop(0))

        # ------------------- FACE DETECTION & EXIT LOGIC -------------------
        gray_small = cv2.cvtColor(fg_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            if fy < 5:
                print("[i] Head is leaving the frame. Cutting video 0.25s early.")
                # Discard the buffer and break (do not write last 0.25s)
                break

    # ───────────────────────────── Cleanup ────────────────────────────────
    fg_cap.release()
    bg_cap.release()
    out.release()
    print(f"[✓] Saved output video → {os.path.abspath(output_path)}")
