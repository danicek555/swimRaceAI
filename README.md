# swimRaceAI

Local swim-start reaction-time estimate from video: detect the start beep, track one swimmer with SAM 2, then estimate when feet leave the block (World Aquatics-style RT ≈ beep → leave).

## Setup

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
# put OPENAI_API_KEY=sk-... in .env (needed for vision-LM leave detection)
```

Put race videos in `videos/`. Model weights (`.pt`) download on first run and are not committed.

## Run

```bash
python3 main.py -f videos -v test1.mp4 --track
```

Optional: set `SWIM_VLM_MODEL` in `.env` (default `gpt-4o`). Use `--no-vlm` for motion-only RT.

## Notes

- `.env` is gitignored — never commit API keys.
- Outputs go under `output/<video_stem>/`.
