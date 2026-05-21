#!/usr/bin/env python3
"""
Remove a MOVING object from a video by tracking it and inpainting with LaMa.

Use when the target shifts position across frames (drifting anti-piracy
watermark, swinging boom mic, a person walking through the shot). For
truly static overlays, use remove_watermark.py instead — it's simpler.

Workflow:
  1. Pick a frame where the target is clearly visible.
  2. Find its bounding box (x, y, w, h) in any image viewer.
  3. Run this script; CSRT tracker follows the box through the video and
     LaMa inpaints the contents each frame.

Usage:
  track_and_remove.py --src in.mp4 --out clean.mp4 --bbox X,Y,W,H
                     [--init-frame 0] [--padding 8] [--crop 256]
                     [--preview tracking.mp4]
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source MP4")
    ap.add_argument("--out", required=True, help="Output MP4")
    ap.add_argument("--bbox", required=True,
                    help="Initial bounding box on --init-frame as X,Y,W,H")
    ap.add_argument("--init-frame", type=int, default=0,
                    help="Frame index where --bbox is defined (default 0)")
    ap.add_argument("--padding", type=int, default=8,
                    help="Padding px around tracked bbox when building the mask")
    ap.add_argument("--crop", type=int, default=256,
                    help="LaMa crop size around the tracked region (default 256)")
    ap.add_argument("--preview", default=None,
                    help="Optional MP4 path to write a tracking-only visualization")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    return ap.parse_args()


def main():
    args = parse_args()
    bx, by, bw, bh = (int(v) for v in args.bbox.split(","))
    init_bbox = (bx, by, bw, bh)

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        sys.exit(f"could not open {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {w}x{h} @ {fps:.2f} fps, {n} frames")
    print(f"init bbox at frame {args.init_frame}: {init_bbox}")

    # Seek to init frame, initialize tracker
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.init_frame)
    ok, frame = cap.read()
    if not ok:
        sys.exit("could not read --init-frame")
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, init_bbox)

    # Reset to frame 0 for full-video processing.
    # CSRT requires sequential frames in order, so if init_frame > 0 we
    # re-initialize from frame 0 by tracking backward isn't supported —
    # in that case we just track forward from --init-frame and skip
    # earlier frames (they pass through unchanged).
    if args.init_frame == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # else: cap is already positioned just after init_frame

    print("loading LaMa...")
    lama = SimpleLama()

    # Set up ffmpeg
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "pipe:0",
        "-i", args.src,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
        "-c:a", "copy",
        "-movflags", "+faststart",
        args.out,
    ], stdin=subprocess.PIPE)

    preview = None
    if args.preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview = cv2.VideoWriter(args.preview, fourcc, fps, (w, h))

    # Process from frame 0 (or init_frame) forward
    if args.init_frame == 0:
        cur = 0
    else:
        # Frames before init_frame: pass through unchanged
        cap2 = cv2.VideoCapture(args.src)
        for i in range(args.init_frame):
            ok, f = cap2.read()
            if not ok: break
            proc.stdin.write(f.tobytes())
            if preview: preview.write(f)
        cap2.release()
        cur = args.init_frame

    last_bbox = init_bbox
    misses = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur += 1

        ok_track, bbox = tracker.update(frame)
        if ok_track:
            last_bbox = tuple(int(v) for v in bbox)
        else:
            misses += 1
            bbox = last_bbox  # fall back to previous box

        x, y, bw_, bh_ = (int(v) for v in last_bbox)
        # Pad and clip to frame bounds
        p = args.padding
        mx0 = max(0, x - p)
        my0 = max(0, y - p)
        mx1 = min(w, x + bw_ + p)
        my1 = min(h, y + bh_ + p)

        # Build mask (filled rectangle at bbox + padding)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[my0:my1, mx0:mx1] = 255

        # Crop around bbox center
        cx_, cy_ = (mx0 + mx1) // 2, (my0 + my1) // 2
        cx0 = max(0, min(w - args.crop, cx_ - args.crop // 2))
        cy0 = max(0, min(h - args.crop, cy_ - args.crop // 2))
        crop_img = frame[cy0:cy0 + args.crop, cx0:cx0 + args.crop]
        crop_mask = mask[cy0:cy0 + args.crop, cx0:cx0 + args.crop]

        if crop_mask.sum() == 0:
            # bbox falls entirely outside the crop window — nothing to inpaint
            proc.stdin.write(frame.tobytes())
            if preview: preview.write(frame)
            continue

        rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        out = lama(Image.fromarray(rgb), Image.fromarray(crop_mask))
        frame[cy0:cy0 + args.crop, cx0:cx0 + args.crop] = cv2.cvtColor(
            np.array(out), cv2.COLOR_RGB2BGR)
        proc.stdin.write(frame.tobytes())

        if preview:
            vis = frame.copy()
            color = (0, 255, 0) if ok_track else (0, 0, 255)
            cv2.rectangle(vis, (mx0, my0), (mx1, my1), color, 2)
            preview.write(vis)

        processed += 1
        if processed % 24 == 0:
            print(f"  frame {cur}/{n}  bbox={last_bbox}  misses={misses}", flush=True)

    cap.release()
    if preview:
        preview.release()
    if proc.stdin:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg exited {rc}")

    print(f"\ndone. inpainted {processed} frames, tracker misses: {misses}")
    if args.preview:
        print(f"tracking visualization: {args.preview}")


if __name__ == "__main__":
    main()
