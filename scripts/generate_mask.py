#!/usr/bin/env python3
"""
Generate a binary mask of a static watermark/logo from a single video frame.

Picks the brightest connected blob in the specified corner of the frame at
the given timestamp, dilates it for safety margin, and writes a PGM file.
The mask is what `remove_watermark.py` uses to know which pixels to inpaint.

For best results, pick a `--timestamp` where the watermark sits over a
uniform-ish background (concrete, sky, wall) so it's easy to threshold.

Usage:
  generate_mask.py --video sample.mp4 --timestamp 0.5 --corner br \
                   --output mask.pgm
"""
import argparse
import sys

import cv2
import numpy as np


CORNERS = {
    # name: (x_start_ratio, y_start_ratio, x_end_ratio, y_end_ratio)
    "br": (0.55, 0.75, 1.00, 1.00),
    "bl": (0.00, 0.75, 0.45, 1.00),
    "tr": (0.55, 0.00, 1.00, 0.25),
    "tl": (0.00, 0.00, 0.45, 0.25),
    "center": (0.30, 0.30, 0.70, 0.70),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Source MP4")
    ap.add_argument("--timestamp", type=float, default=0.5,
                    help="Seconds into the video to grab the reference frame")
    ap.add_argument("--corner", choices=CORNERS.keys(), default="br",
                    help="Which region of the frame to scan")
    ap.add_argument("--threshold", type=float, default=0.88,
                    help="Brightness percentile of the corner ROI used to "
                         "binarize the watermark (0-1). Lower if the mask "
                         "misses faint edges; raise if it catches background.")
    ap.add_argument("--dilate", type=int, default=5,
                    help="Pixels of dilation around the detected shape "
                         "(safety margin so inpainting has clean boundary).")
    ap.add_argument("--output", default="mask.pgm",
                    help="Output PGM path (white = inpaint, black = keep).")
    return ap.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.timestamp * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("could not read reference frame")

    print(f"video: {W}x{H} @ {fps:.2f} fps")
    rx0, ry0, rx1, ry1 = CORNERS[args.corner]
    x0, y0 = int(W * rx0), int(H * ry0)
    x1, y1 = int(W * rx1), int(H * ry1)
    print(f"scanning {args.corner} region x={x0}-{x1} y={y0}-{y1}")

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th = int(gray.max() * args.threshold)
    _, bin_roi = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    # Find largest connected component (the watermark)
    n, _, stats, cent = cv2.connectedComponentsWithStats(bin_roi)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 30 < area < 50000 and 5 < w < (x1 - x0) // 2 and 5 < h < (y1 - y0) // 2:
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
    if best is None:
        sys.exit("no plausible watermark blob found — try --threshold lower "
                 "or pick a different --timestamp")

    bx, by, bw, bh, area = best
    abs_x, abs_y = x0 + bx, y0 + by
    print(f"detected watermark: abs x={abs_x}-{abs_x+bw} y={abs_y}-{abs_y+bh} "
          f"({bw}x{bh}, area={area})")

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = bin_roi

    if args.dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (args.dilate * 2 + 1, args.dilate * 2 + 1))
        mask = cv2.dilate(mask, k)

    ys, xs = np.where(mask > 0)
    print(f"final mask: {(mask > 0).sum()} pixels, "
          f"bbox x={xs.min()}-{xs.max()} y={ys.min()}-{ys.max()}")

    cv2.imwrite(args.output, mask)
    print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
