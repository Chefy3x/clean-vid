---
name: clean-vid
description: Remove watermarks, logos, or unwanted objects from MP4 videos using LaMa neural inpainting — both static overlays (AI watermarks like Veo/Sora/Runway, TV station bugs, date stamps) AND moving targets (drifting anti-piracy marks, swinging boom mics, a person walking through a shot). Use whenever the user wants to clean a video — including phrases like "remove the watermark", "strip the sparkle from these clips", "delogo this MP4", "the watermark is still showing", "remove that boom mic", "erase the person in the background", "clean up this footage". Pulls in LaMa so output is seamless even where subjects pass through the cleaned region — far cleaner than ffmpeg's built-in delogo, which leaves a visible smudge.
---

# Video Watermark / Object Removal

Two modes, decide which by whether the target stays still:

| Mode | When | Script |
|------|------|--------|
| **Static** | Target stays at the same pixel position every frame | `remove_watermark.py` |
| **Tracking** | Target shifts position across frames | `track_and_remove.py` |

## Setup (one-time)

```bash
brew install ffmpeg
python3.12 -m venv ~/.venvs/clean-vid
~/.venvs/clean-vid/bin/pip install -r requirements.txt
```

Python 3.12 specifically — 3.14 breaks pillow's old build setup. First run auto-downloads `big-lama.pt` (~196 MB).

## Static workflow

1. **Generate the mask** from a sample MP4 at a timestamp where the watermark sits over a uniform background:

   ```bash
   ~/.venvs/clean-vid/bin/python scripts/generate_mask.py \
     --video sample.mp4 --timestamp 0.5 --corner br --output mask.pgm
   ```

   Confirm by opening `mask.pgm` — white pixels should outline the watermark cleanly with a small safety margin.

2. **Batch-process the folder:**

   ```bash
   ~/.venvs/clean-vid/bin/python scripts/remove_watermark.py \
     --src ./inputs --out ./outputs --mask mask.pgm
   ```

## Tracking workflow

1. **Extract the initial frame and read the target's bbox:**

   ```bash
   ffmpeg -ss 0 -i input.mp4 -frames:v 1 -update 1 frame0.png
   ```

   Open in any image viewer that shows pixel coordinates. Note X, Y, W, H.

2. **Track and inpaint:**

   ```bash
   ~/.venvs/clean-vid/bin/python scripts/track_and_remove.py \
     --src input.mp4 --out clean.mp4 --bbox 596,1155,57,58 \
     --preview tracking.mp4
   ```

   The `--preview` MP4 visualizes the tracked box (green = ok, red = miss). Scrub through to confirm no drift.

## Verification

Spot-check the output by extracting frames at varied timestamps — especially:
- A frame where the target sits over a **smooth background** (delogo's worst case)
- A frame where a **subject's body overlaps** the target area
- A frame where there's a **high-contrast edge** adjacent to the target

```bash
ffmpeg -ss 9.5 -i out.mp4 -frames:v 1 -update 1 -vf "crop=200:200:540:1080" check.png
```

## Common gotchas

- **"It's still there"** after first pass → eyeball-estimated bbox is usually 10-20 px off. `generate_mask.py` prints the detected bounds; confirm against `mask.pgm`.
- **Python 3.14** fails on simple-lama-inpainting install. Use 3.12.
- **Mask polarity** — LaMa uses white = inpaint (same as cv2.inpaint). cv2.xphoto.inpaint (FSR) is inverted.
- **Tracker drift** — if CSRT loses the target, the target probably looks too similar to its background. Tighten the initial bbox or pick a frame with more contrast.

## When NOT to use

- Watermarks covering >25% of frame (LaMa hallucinates on large holes)
- Audio watermarks (video-only pipeline)
- Single one-off videos where one ffmpeg call would do (overkill, just use delogo)
- Non-rigid moving targets (CSRT will lose them — use SAM2 instead, see README)

## Algorithms tried before settling on LaMa

For reference, in case LaMa is unavailable:

| Approach | Outcome |
|----------|---------|
| `ffmpeg -vf delogo` | Visible horizontal smudge artifact. Bad over smooth surfaces and when a subject crosses the watermark area. |
| `cv2.inpaint` (TELEA/NS) | Better than delogo. Still streaks at high-contrast edges. |
| `cv2.xphoto.inpaint` (FSR_FAST/BEST) | Cleaner than TELEA; faint residual on hard edges. |
| Alpha-matte recovery (estimate α from cross-frame variance) | Theoretically exact, but α estimates under-shoot in practice. |
| **LaMa (simple-lama-inpainting)** | **Seamless. Final answer.** |
