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

## Outputs

Everything goes to `output/<video_stem>/`:

- `<video>_track6s.mp4` — tracked video (starts at your box frame),
- `vlm_crops/` — RT audit: exactly what the LM saw, with verdict + occlusion %,
- `vlm_hand_crops/` — hand-entry audit incl. `local_t*.jpg` scan strips.

## Known limits

- Designed for a **static side-view camera**; head-on views are unsupported.
- 30 fps video quantizes every time to ±0.03 s — treat RT as `value ±0.03 s`.
- A camera microphone far from the start speaker hears the beep late
  (~0.1 s per 35 m) and biases RT low; broadcast audio is fine.
- Red-ish blocks and lane ropes are detected directly; other colours fall
  back to geometry estimated from the tracked body (less precise).
- Far lanes have fewer pixels: expect noisier results than for near lanes.
