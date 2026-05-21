#!/usr/bin/env python3
"""
Batch-remove a STATIC watermark from MP4 files using LaMa neural inpainting.

For each .mp4 in --src:
  - Crops a small region around the watermark (faster than full-frame inpaint)
  - Runs LaMa to inpaint the masked pixels
  - Pipes the inpainted frames into ffmpeg (libx264 crf 18 preset slow)
  - Copies the original audio stream
  - Skips outputs that already exist (safe to re-run)

Usage:
  remove_watermark.py --src ./inputs --out ./outputs --mask mask.pgm
"""
import argparse
import glob
import os
import subprocess
import sys
import traceback

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder of source MP4s")
    ap.add_argument("--out", required=True, help="Folder for cleaned outputs")
    ap.add_argument("--mask", required=True,
                    help="PGM mask (white = inpaint), full-frame size")
    ap.add_argument("--crop", type=int, default=256,
                    help="LaMa crop size around the mask (default 256). "
                         "Increase if mask is larger than ~200px.")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    return ap.parse_args()


def main():
    args = parse_args()
    full_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if full_mask is None:
        sys.exit(f"could not read mask: {args.mask}")
    H, W = full_mask.shape

    # Crop window centered on the mask
    ys, xs = np.where(full_mask > 0)
    if len(xs) == 0:
        sys.exit("mask is empty")
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    crop_x = max(0, min(W - args.crop, cx - args.crop // 2))
    crop_y = max(0, min(H - args.crop, cy - args.crop // 2))
    crop_mask = full_mask[crop_y:crop_y + args.crop, crop_x:crop_x + args.crop]
    pil_mask = Image.fromarray(crop_mask)
    print(f"mask: {(full_mask > 0).sum()} px, crop window "
          f"x={crop_x} y={crop_y} size={args.crop}")

    print("loading LaMa...")
    lama = SimpleLama()
    print("ready")

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "_errors.log")

    def inpaint_frame(frame):
        crop = frame[crop_y:crop_y + args.crop, crop_x:crop_x + args.crop]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out = lama(Image.fromarray(rgb), pil_mask)
        frame[crop_y:crop_y + args.crop, crop_x:crop_x + args.crop] = \
            cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        return frame

    def process(src, dst):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("could not open source")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (h, w) != full_mask.shape:
            cap.release()
            raise RuntimeError(
                f"frame {w}x{h} doesn't match mask {W}x{H}; "
                "regenerate mask against this video's resolution"
            )

        proc = subprocess.Popen([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "pipe:0",
            "-i", src,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
            "-c:a", "copy",
            "-movflags", "+faststart",
            dst,
        ], stdin=subprocess.PIPE)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                proc.stdin.write(inpaint_frame(frame).tobytes())
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited {rc}")

    files = sorted(glob.glob(os.path.join(args.src, "*.mp4")))
    total = len(files)
    processed = skipped = failed = 0

    with open(log_path, "w") as logf:
        if total == 0:
            print(f"no .mp4 files in {args.src}")
        for i, src in enumerate(files, 1):
            name = os.path.basename(src)
            dst = os.path.join(args.out, name)
            print(f"[{i}/{total}] {name} (remaining: {total - i})", flush=True)
            if os.path.exists(dst):
                print("  -> skip (already exists)")
                skipped += 1
                continue
            try:
                process(src, dst)
                print("  -> done")
                processed += 1
            except Exception as e:
                print(f"  -> FAILED ({e})")
                traceback.print_exc(file=logf)
                logf.write(f"--- {name}: {e}\n")
                failed += 1
                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except OSError:
                        pass

    print()
    print("Summary:")
    print(f"  processed: {processed}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    print(f"  total:     {total}")
    if failed:
        print(f"  errors logged to: {log_path}")


if __name__ == "__main__":
    main()
