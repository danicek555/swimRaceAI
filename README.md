# swimRaceAI

Swim-start analysis from a side-view video, fully local except an optional
vision-LM check. Measures, per one clicked swimmer:

- **start beep** — found in the audio track (band-pass + RMS onset),
- **reaction time (RT)** — beep → feet leave the block, World Aquatics style,
- **hand entry** — first hand-water contact, and **flight time** (leave → hands).

## How it works

1. `find_beep` locates the start signal in the first seconds of audio.
2. A picker window opens ~1 s before the beep (swimmers are set on their
   blocks); you draw a tight box around ONE swimmer. Tracking starts **on the
   frame you draw on** — that frame becomes frame 0 of the tracked video, and
   every time/frame reported afterwards uses that timeline.
3. SAM 2 follows the swimmer. Static red pixels (block, lane ropes) are cut
   from the mask each frame so it cannot latch onto the empty block.
4. **RT**: a crop window is locked around the swimmer's OWN starting block
   (found under their feet at beep time, re-locked from the empty platform),
   other lanes are dimmed, and a vision LM answers ON/LEFT per sample. A
   colour-blind **occlusion veto** (frame diff against the empty platform,
   shadow-filtered) overrides LM mistakes while anything still covers the
   platform. Two LEFT answers in a row lock the leave and stop sampling.
5. **Hand entry**, three stages: (a) free per-frame **local scan** — surface
   disturbance inside the swimmer's own lane band (between detected lane
   ropes), cell grid + per-cell ambient baseline + persistence; (b) the LM
   confirms 2–3 crops around the candidate; (c) full LM sweep only as a
   fallback (occlusions). Works without an API key via stage (a) alone.
6. The output video gets burned-in times and flashes: red **BEEP**, green
   leave, cyan hand entry.

## Project layout

```
main.py            CLI entry point (thin facade over swim/)
swim/config.py     all tunable constants + .env loading
swim/audio.py      beep detection
swim/utils.py      clips, crops, small helpers
swim/vlm.py        OpenAI-compatible vision-LM calls + few-shot refs
swim/blocks.py     starting-block geometry, occlusion veto, dimming
swim/tracking.py   click UI, SAM mask cleanup, trajectory helpers
swim/reaction.py   leave-block detection (LM + veto + motion refine)
swim/hands.py      hand-entry detection (local scan + LM)
swim/overlay.py    burned-in overlay + H.264 encode
swim/cuts.py       hard camera-cut detection + audit previews
swim/detect.py     optional YOLO people-detection mode (no --track)
swim/pipeline.py   orchestration (track_one_swimmer, run)
vlm_refs/          few-shot example photos for the LM
videos/            sample clips (test1.mp4, test2.mp4)
```

Unused historical code is kept commented out in place (marked
`NEPOUZIVANE`) rather than deleted.

## Setup

Requires Python 3.10+, **ffmpeg** on PATH, and ~250 MB for model weights
(downloaded on first run, not committed).

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env   # put OPENAI_API_KEY=sk-... inside (optional)
```

Without an API key the pipeline still runs: RT falls back to motion-only and
hand entry uses the local scan. `.env` is gitignored — never commit API keys.

## Run

```bash
python3 main.py -f videos -v test1.mp4 --track
```

Keys in the picker: `A`/`D` frame step, `Enter` draw the box, `Esc` cancel.
Draw the box tight on the body — without the block platform under the feet.

Optional: `--no-vlm` (motion-only RT), `--known-rt 0.62` (one-time threshold
calibration against an official RT), `SWIM_VLM_MODEL` in `.env`.

Detect hard camera edits without loading SAM or YOLO:

```bash
python3 main.py -f videos -v test1.mp4 --detect-cuts
```

For every consecutive frame pair this compares the HSV colour histogram,
mean grayscale pixel difference, and Canny edge disagreement. A cut must pass
both whole-frame signal floors, be a local score peak, and be at least 0.5 s
from the previous accepted cut. This rejects most local swimmer/splash motion.

Evaluate open-vocabulary / specialized swimmer detection after the first cut:

```bash
python3 main.py -f videos -v test1.mp4 --yolo-after-cut --preview-seconds 30
python3 main.py -f videos -v test1.mp4 --yolo-after-cut --lane 7 --preview-seconds 20
```

This draws every raw box from the DBDoco YOLOv5 `person_swimmer` model
([yolo-swimmer-detection](https://github.com/DBDoco/yolo-swimmer-detection),
MIT) from the first detected cut to the requested end time. With `--lane`,
the requested vertical lane band is tinted and its best box is highlighted.

Setup once (weights + classic YOLOv5 loader; both are gitignored):

```bash
mkdir -p models/yolo_swimmer
curl -L -o models/yolo_swimmer/best.pt \
  https://raw.githubusercontent.com/DBDoco/yolo-swimmer-detection/main/models/exp5/weights/best.pt
git clone --depth 1 https://github.com/ultralytics/yolov5.git third_party/yolov5
```

Re-acquire after the first cut with SAM 3 (text=`swimmer`), then hand the
stable lane box to SAM 2 for the whole shot (cut → next cut):

```bash
# Place Meta's gated sam3.pt in the project root first
python3 main.py -f videos -v test1.mp4 \
  --retrack-after-cut --lane 1 --closest-lane 8

# Same for the second hard cut (cut #2 at ~24s on test1):
python3 main.py -f videos -v test1.mp4 \
  --retrack-after-cut --cut 2 --lane 2 --closest-lane 8
```

Each SAM 3 frame first detects the red/yellow/blue lane ropes and turns them
into perspective lane polygons. `--closest-lane 8` means the bottom/nearest
polygon is physical lane 8; from the opposite pool side use `--closest-lane 1`.
SAM 3 samples about every 0.5s until a lane-matched box appears at least twice
with confidence ≥ 0.50. SAM 2 then tracks that box backward to the cut and
forward to the next cut. During the forward pass, 0.5s of continuously missing
mask triggers a new lane-filtered SAM 3 search and a fresh SAM 2 predictor
(maximum three re-seeds per shot). Lane ropes are still detected throughout
the shot. Each fit receives a quality score based on real rope support, water
coverage, and perspective residual. Temporal smoothing follows gradual camera
motion, while explicitly rejecting fits whose new lane numbering matches the
old geometry one lane off. A re-seed is constrained by SAM 2's last position (not its velocity — wake
speed would throw away the real swimmer). Allowance grows with time while the
mask is missing. If the specialized swimmer YOLO sees a person in the lane,
SAM 3 boxes must overlap it (foam veto); if YOLO sees nothing, SAM 3 is used
alone. SAM 2 masks whose centre leaves the requested physical lane are
rejected instead of being allowed to switch identity.
SAM 3 also performs a sparse identity check every 2 seconds against the
strongest lane-matched swimmer box (not any overlapping foam box). An empty
SAM 2 mask at a checkpoint where SAM 3 sees a swimmer counts as a miss; two
misses trigger a re-seed even when SAM 2 is still returning a non-empty mask
on wake or foam. After a re-seed, the empty seconds between loss and the new
seed are filled by tracking SAM 2 backward from the new seed. SAM 3
detections are associated by centre motion as well as IoU, so a moving
swimmer does not fragment into unrelated one-hit candidates. When rope
detection drops out, the last good lane ladder is kept so SAM 3 can still run.

SAM 3 preview only (no SAM 2 tracking), filtered by lane:

```bash
python3 main.py -f videos -v test1.mp4 --sam3-after-cut --lane 7 --preview-seconds 20
```
## Outputs

Everything goes to `output/<video_stem>/`:

- `<video>_track6s.mp4` — tracked video (starts at your box frame),
- `vlm_crops/` — RT audit: exactly what the LM saw, with verdict + occlusion %,
- `vlm_hand_crops/` — hand-entry audit incl. `local_t*.jpg` scan strips,
- `cuts/frame_change_scores.csv` + `cuts/cut_*.jpg` — every frame-change score
  and side-by-side evidence for each detected hard cut,
- `<video>_yolo_swimmers_after_cut_30s.mp4` — raw swimmer detections after the
  first cut, with original audio retained.

## Known limits

- Designed for a **static side-view camera**; head-on views are unsupported.
- Cut detection finds hard edits; slow cross-fades/dissolves are not yet
  classified. Tracking still needs re-acquisition after a detected cut.
- 30 fps video quantizes every time to ±0.03 s — treat RT as `value ±0.03 s`.
- A camera microphone far from the start speaker hears the beep late
  (~0.1 s per 35 m) and biases RT low; broadcast audio is fine.
- Red-ish blocks and lane ropes are detected directly; other colours fall
  back to geometry estimated from the tracked body (less precise).
- Far lanes have fewer pixels: expect noisier results than for near lanes.
