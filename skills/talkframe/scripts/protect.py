#!/usr/bin/env python3
"""Keep a flat detail (a logo card, a sign, a lower third printed in the photo) pixel-exact in a talkframe video.

Video models redraw everything, so small text such as a microphone flag's logo wobbles from frame to frame.
This tracks that flat region through the video with a homography (OpenCV ECC), then pastes the original pixels from the
photo back over it, matched to each frame's lighting and feathered at the edges. Frames where tracking is unsure,
for example a hand passing in front, are left as generated.

    python3 protect.py --video talk.mp4 --image photo.jpg --region X,Y,W,H --out talk_fixed.mp4 [--debug dbg.mp4]

--region is in the photo's own pixel coordinates (the same photo given to talkframe).
Needs: pip install numpy opencv-python-headless   (only for this optional step)
"""
import argparse
import subprocess
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("protect.py needs numpy and OpenCV: pip install numpy opencv-python-headless")


def crop_like_render(img, w, h):
    """Reproduce talkframe's framing (scale to cover, center crop) so photo and video pixels line up."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (round(iw * s), round(ih * s)), interpolation=cv2.INTER_AREA)
    ox, oy = (r.shape[1] - w) // 2, (r.shape[0] - h) // 2
    return r[oy:oy + h, ox:ox + w], s, ox, oy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--region", required=True, help="X,Y,W,H in the photo's pixels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.55, help="skip frames whose aligned match is below this")
    ap.add_argument("--debug", help="also write a video with the tracked outline drawn")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ref, s, ox, oy = crop_like_render(cv2.imread(a.image), W, H)
    x, y, w, h = (float(v) for v in a.region.split(","))
    rx, ry, rw, rh = round(x * s - ox), round(y * s - oy), round(w * s), round(h * s)
    if rx < 0 or ry < 0 or rx + rw > W or ry + rh > H or rw < 16 or rh < 8:
        sys.exit(f"region maps to {rx},{ry},{rw},{rh} in the {W}x{H} video: outside the frame or too small")
    patch = ref[ry:ry + rh, rx:rx + rw].copy()
    tmpl = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)

    # feathered alpha so the pasted pixels blend into the redrawn surroundings
    m = np.zeros((rh, rw), np.float32)
    b = max(2, min(rw, rh) // 12)
    m[b:-b, b:-b] = 1
    m = cv2.GaussianBlur(m, (0, 0), b / 1.5)

    enc = lambda path: subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-i", a.video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    out = enc(a.out)
    dbg = enc(a.debug) if a.debug else None

    Hm = np.array([[1, 0, rx], [0, 1, ry], [0, 0, 1]], np.float32)  # template coords -> frame coords
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    n = pasted = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        cand, score = Hm.copy(), -1.0
        for attempt in range(2):
            try:
                _, cand = cv2.findTransformECC(tmpl, g, cand, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
                al = cv2.warpPerspective(g, cand, (rw, rh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                score = float(cv2.matchTemplate(al, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            except cv2.error:
                score = -1.0
            if score >= a.min_score or attempt:
                break
            # lost it: re-acquire by translation search around the last position, then refine again
            pad = max(rw, rh)
            x0, y0 = int(max(Hm[0, 2] - pad, 0)), int(max(Hm[1, 2] - pad, 0))
            win = g[y0:int(min(Hm[1, 2] + rh + pad, H)), x0:int(min(Hm[0, 2] + rw + pad, W))]
            if win.shape[0] < rh or win.shape[1] < rw:
                break
            _, _, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED))
            cand = np.array([[1, 0, x0 + mx], [0, 1, y0 + my], [0, 0, 1]], np.float32)
        res = f
        if score >= a.min_score:
            Hm = cand
            # match the photo patch to this frame's light (per-channel gain over the region)
            al = cv2.warpPerspective(f, Hm, (rw, rh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype(np.float32)
            gain = (al.reshape(-1, 3).mean(0) + 1) / (patch.reshape(-1, 3).mean(0) + 1)
            p = np.clip(patch.astype(np.float32) * np.clip(gain, 0.7, 1.4), 0, 255)
            wp = cv2.warpPerspective(p, Hm, (W, H), flags=cv2.INTER_LINEAR)
            wa = cv2.warpPerspective(m, Hm, (W, H), flags=cv2.INTER_LINEAR)[..., None]
            res = (f.astype(np.float32) * (1 - wa) + wp * wa).astype(np.uint8)
            pasted += 1
        out.stdin.write(res.tobytes())
        if dbg:
            d = res.copy()
            corners = cv2.perspectiveTransform(np.array([[[0, 0], [rw, 0], [rw, rh], [0, rh]]], np.float32), Hm)
            cv2.polylines(d, [corners.astype(np.int32)], True, (0, 255, 0) if score >= a.min_score else (0, 0, 255), 1)
            cv2.putText(d, f"{score:.2f}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            dbg.stdin.write(d.tobytes())
    codes = []
    for p in (out, dbg):
        if p:
            p.stdin.close()
            codes.append(p.wait())
    if any(codes):
        sys.exit(f"[protect] ffmpeg failed while writing the output (exit codes {codes})")
    print(f"[protect] region {rx},{ry},{rw}x{rh} in video; pasted on {pasted}/{n} frames ({100 * pasted / max(n, 1):.0f}%) -> {a.out}")


if __name__ == "__main__":
    main()
