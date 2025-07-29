import cv2, numpy as np, argparse, os, sys

# ──────────────────────────────────────────────────────────────────────────
def ecc_align(ref_bgr, mov_bgr, warp_mode=cv2.MOTION_EUCLIDEAN, iterations=200):
    """Rigidly align mov_bgr to ref_bgr using ECC; returns the warped image."""
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)                 # 2×3 rigid transform
    try:
        _, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, warp_mode,
                                       criteria=(cv2.TERM_CRITERIA_EPS |
                                                 cv2.TERM_CRITERIA_COUNT,
                                                 iterations, 1e-6))
        aligned = cv2.warpAffine(mov_bgr, warp,
                                 (ref_bgr.shape[1], ref_bgr.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except cv2.error:
        print("[WARN] ECC failed → returning original foreground image")
        return mov_bgr

# ──────────────────────────────────────────────────────────────────────────
def build_alpha(bg_bgr, fg_bgr, blur_k=5, use_otsu=True):
    """Return binary/soft mask that separates fg from bg."""
    # 1 | difference in YCrCb space (ignores brightness)
    bg_ycc = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    fg_ycc = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    cr = cv2.absdiff(fg_ycc[..., 1], bg_ycc[..., 1])
    cb = cv2.absdiff(fg_ycc[..., 2], bg_ycc[..., 2])
    chroma_diff = cv2.addWeighted(cr, .5, cb, .5, 0).astype(np.uint8)

    # 2 | threshold
    if use_otsu:
        _, mask = cv2.threshold(chroma_diff, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(chroma_diff, 25, 255, cv2.THRESH_BINARY)

    # 3 | clean + feather
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), blur_k)        # feather edge
    return mask

# ──────────────────────────────────────────────────────────────────────────
def despill_edge(fg_bgr, alpha, despill_strength=.2):
    """Reduce green spill only on fuzzy edge pixels."""
    hsv = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    edge = (alpha > 0) & (alpha < 255)                   # 0 < α < 255
    h, s, v = cv2.split(hsv)
    greenish = (h >= 35) & (h <= 85)                     # rough green hue range
    target = edge & greenish
    s[target] *= despill_strength
    hsv[..., 1] = np.clip(s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ──────────────────────────────────────────────────────────────────────────
def extract_foreground(bg_path, fg_path, out_path,
                       align=False, use_otsu=True, blur_k=5):
    bg = cv2.imread(bg_path)
    fg = cv2.imread(fg_path)
    if bg is None or fg is None:
        sys.exit("❌ Couldn’t read one of the images.")

    # resize foreground to background size if needed
    if fg.shape[:2] != bg.shape[:2]:
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]))

    # optional ECC alignment
    if align:
        fg = ecc_align(bg, fg)

    alpha = build_alpha(bg, fg, blur_k=blur_k, use_otsu=use_otsu)

    # green‑spill removal
    clean = despill_edge(fg, alpha)

    # compose BGRA
    out = cv2.cvtColor(clean, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    cv2.imwrite(out_path, out)
    print("✅ Saved →", out_path)

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Subtract background from person shot and output PNG with transparency.")
    ap.add_argument("background", help="Background (empty) image path")
    ap.add_argument("with_person", help="Image path containing the person")
    ap.add_argument("-o", "--output", default="foreground.png",
                    help="Output 32‑bit PNG path")
    ap.add_argument("--align", action="store_true",
                    help="Run ECC alignment before subtracting")
    ap.add_argument("--no-otsu", action="store_true",
                    help="Use fixed threshold 25 instead of Otsu")
    ap.add_argument("--blur", type=int, default=5,
                    help="Edge feather radius (σ) in pixels")
    args = ap.parse_args()

    extract_foreground(args.background, args.with_person, args.output,
                       align=args.align,
                       use_otsu=not args.no_otsu,
                       blur_k=args.blur)
