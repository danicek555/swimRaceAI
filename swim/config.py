"""swimRaceAI — konstanty a .env."""

import os
from pathlib import Path


def load_dotenv_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (if missing)."""
    env_path = path or (Path(__file__).resolve().parent.parent / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()

# Only look at the start of the video. The start beep should be here.
SEARCH_SECONDS = 15

# YOLO only looks at the first 10 seconds of picture. Faster on a laptop.
DETECT_SECONDS = 6

# Click-to-track also only uses the start of the race.
TRACK_SECONDS = 6

# Official World Aquatics "reaction time" = beep -> feet LEAVE the block
# (pressure switch), usually ~0.55-0.80s for freestyle. We approximate that
# with a BIG body move, not a 12px SAM wobble.
#
# Threshold is a FRACTION of the swimmer's on-screen size (not raw pixels),
# so zoom / distance / resolution do not need re-calibration every time.
# Optional: once tune with --known-rt 0.62, then set LEAVE_BLOCK_BODY_FRAC.
LEAVE_BLOCK_BODY_FRAC = 0.75
# Require that big move to last this long (seconds) so noise does not count.
LEAVE_HOLD_SECONDS = 0.05
# After the beep, resample motion every 0.01s (like the timing console).
RT_STEP_SECONDS = 0.01
# If the mask center jumps farther than this in one frame, treat it as an
# identity switch (neighbor steal) and keep the previous mask instead.
# At 30fps a real dive move is usually under this; a lane-jump is bigger.
MAX_CENTER_JUMP_PX = 90

# Hybrid RT: vision LM finds when feet leave the block. Needs OPENAI_API_KEY.
# Pick a vision Chat Completions model via SWIM_VLM_MODEL in .env, e.g.:
#   gpt-4o-mini | gpt-4.1-mini | gpt-4o | gpt-4.1 | gpt-5.2 | gpt-5.4 | gpt-5.6-terra
# See .env.example for the full list. Default: gpt-4o
VLM_MODEL = os.environ.get("SWIM_VLM_MODEL", "gpt-4o")
VLM_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Ask the LM in this full window after the beep.
VLM_RT_MIN_SECONDS = 0.50
VLM_RT_SEARCH_SECONDS = 1.20
# Dense band (typical RT): sample every VLM_SAMPLE_DENSE_STEP.
VLM_DENSE_MIN_SECONDS = 0.58
VLM_DENSE_MAX_SECONDS = 0.80
VLM_SAMPLE_DENSE_STEP_SECONDS = 0.03
# Outside the dense band (still inside 0.50–1.20): coarser samples.
VLM_SAMPLE_COARSE_STEP_SECONDS = 0.05
# Need this many LEFT_BLOCK answers in a row (no ON in between) to trust leave.
# Stops "LEFT then 0.2s later still ON" false alarms when the crop missed a foot.
# 2 is enough WITH the occlusion veto active (the veto catches false LEFTs
# during contact); without the veto the code requires one more. Confirming
# fast matters: once locked, sampling stops, so a NEIGHBOUR's foot flying
# through the crop later can no longer flip a parasitic ON.
VLM_LEFT_CONFIRM = 2
# Tight crop around feet + block (not the whole pool / other lanes).
# Final crop side length is clamped to this range (pixels).
VLM_CROP_MIN_SIDE = 360
VLM_CROP_MAX_SIDE = 520
# Max side length of JPEG sent to the LM.
VLM_JPEG_MAX_SIDE = 512

# Tight crop mode: build the window from the swimmer's OWN block (found once
# at beep time, when the feet are provably standing on it) instead of from the
# whole body box. A small region is UPSCALED to VLM_JPEG_MAX_SIDE rather than
# widened, so neighbouring lanes never enter the picture just to hit a minimum
# size. Set False to fall back to the old body-box crop.
VLM_TIGHT_CROP = True
# Padding around the block rect, as multiples of the block width / height.
# "fwd" is the dive direction (towards the water), inferred per video.
# PAD_UP is deliberately small: on a side camera the flying swimmer's feet
# visually cross the FARTHER lanes' blocks (parallax, higher in the image).
# Keeping those blocks out of frame stops the LM judging them.
VLM_BLOCK_PAD_FWD = 0.85
VLM_BLOCK_PAD_BACK = 0.40
VLM_BLOCK_PAD_UP = 0.90
VLM_BLOCK_PAD_DOWN = 0.55
# Search window (fraction of frame size) used to find the swimmer's block
# around their feet at beep time.
VLM_BLOCK_SEARCH_FRAC = 0.30
# Darken anything outside the swimmer's own block region so leftover
# neighbouring lanes cannot be mistaken for "the" block. 1.0 disables it.
VLM_DIM_OUTSIDE = 0.30
# Motion refine (reported on its own line; primary RT stays LM).
VLM_REFINE_SIGNAL_FRAC = 0.55
# Few-shot reference photos shown to the LM before each crop.
VLM_REFS_DIR = Path(__file__).resolve().parent.parent / "vlm_refs"
VLM_REF_ON = VLM_REFS_DIR / "example_ON_BLOCK.jpg"
VLM_REF_ON_HARD = VLM_REFS_DIR / "example_ON_BLOCK_hard.jpg"
VLM_REF_LEFT = VLM_REFS_DIR / "example_LEFT_BLOCK.jpg"
VLM_REF_LEFT_HARD = VLM_REFS_DIR / "example_LEFT_BLOCK_hard.jpg"
VLM_REF_ON_WEDGE = VLM_REFS_DIR / "example_ON_BLOCK_wedge.jpg"
# Local foot∩block overlap: force ON_BLOCK if this many pixels touch.
# (Legacy skin-hue veto — only used when the tight block crop is unavailable.
# Skin hue misses shadowed feet entirely, so prefer the occlusion veto below.)
VLM_OVERLAP_FORCE_ON_PX = 80
# Occlusion veto: the platform shape is locked from a reference frame where it
# is EMPTY (after the swimmer left). Per sample, changed pixels inside that
# shape (vs the empty reference, shadows filtered out) = something still
# covering the platform. If a LEFT answer arrives while at least this fraction
# is covered, force ON_BLOCK.
# 0.04 measured on test1: empty platform noise is 0-2 percent, the last real
# toe contact at the front edge reads 6-7 percent. The LM misses that toe
# (dark shin against the pillar) — the veto is what carries those frames.
VLM_OCCLUSION_FORCE_ON_FRAC = 0.04

# Hand entry: first time hands touch the water after leave-block.
VLM_HAND_SEARCH_SECONDS = 0.90
VLM_HAND_SAMPLE_STEP_SECONDS = 0.10
VLM_HAND_CONFIRM = 3
# Local hand-entry scan (no API, every frame): surface disturbance inside the
# swimmer's OWN lane band. An outdoor pool with people in it shimmers and
# ripples EVERYWHERE, so a plain frame diff fires constantly — the entry blob
# differs from shimmer by being COMPACT (dense in one grid cell), ABOVE the
# per-cell ambient baseline, and PERSISTENT at one spot across frames.
HAND_LOCAL_MIN_FLIGHT_SECONDS = 0.15
# After this long past leave the hands are in the water no matter what the
# LM sees — a late UNKNOWN (splash hides the hands) counts as WATER.
HAND_UNKNOWN_IS_WATER_AFTER_SECONDS = 0.60
HAND_LOCAL_CELL_PX = 24
HAND_LOCAL_CELL_SIGNAL = 0.30
HAND_LOCAL_CONFIRM_FRAMES = 3
# Local splash second vote (bright foam fraction / frame-to-frame change).
VLM_HAND_SPLASH_WHITE_FRAC = 0.07
VLM_HAND_SPLASH_DELTA = 0.040
# Soft override: very strong splash can flip AIR → WATER.
VLM_HAND_SPLASH_STRONG_FRAC = 0.14
VLM_REF_HANDS_AIR = VLM_REFS_DIR / "example_HANDS_AIR.jpg"
VLM_REF_HANDS_WATER = VLM_REFS_DIR / "example_HANDS_WATER.jpg"
VLM_REF_HANDS_WATER_SPLASH = VLM_REFS_DIR / "example_HANDS_WATER_splash.jpg"

# A swim start beep is usually a mid/high tone, not a splash or a voice.
# Hz = Hertz = "how many vibrations per second". Higher Hz = higher pitch.
BEEP_MIN_HZ = 800
BEEP_MAX_HZ = 3000

# Measure loudness in tiny slices of audio (20 milliseconds each).
# Smaller windows = more precise time, but noisier measurements.
WINDOW_MS = 20

# File types we treat as videos inside the folder.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Detect model (COCO "person"), not pose. Pose missed the crouched starts.
YOLO_DETECT_MODEL = "yolo26m.pt"

# Open-vocabulary model: we tell it the words "swimmer" / "starting block".
YOLO_WORLD_MODEL = "yolov8m-worldv2.pt"
WORLD_CLASSES = ["swimmer", "person on starting block", "athlete"]

# In the COCO dataset, class 0 = person.
PERSON_CLASS_ID = 0

# Low-ish, but 0.12 was showing junk like 0.13 fragments.
PERSON_CONFIDENCE = 0.25
WORLD_CONFIDENCE = 0.20

YOLO_IMAGE_SIZE = 960
WORLD_IMAGE_SIZE = 1280

# Lower iou = merge overlapping boxes more aggressively (tiles + two models).
YOLO_IOU = 0.32

# Wider left crop so leaning bodies are not cut off.
BLOCK_CROP_WIDTH = 0.42

# Smaller tiles = more zoom on 1-2 lanes.
TILE_HEIGHT_RATIO = 0.30
TILE_OVERLAP_RATIO = 0.16

# Skip frames to keep a laptop usable (2 = every other frame, reuse last boxes).
DETECT_EVERY_N_FRAMES = 2

# Ignore people too far left (crowd/officials) or too far right (in the water).
ON_BLOCK_X_MIN = 0.12
ON_BLOCK_X_MAX = 0.60

LANE_COUNT = 8

# COCO pose is unused for the main detector now. Standing filter is optional.
MAX_UPRIGHT = 0.28
KEYPOINT_MIN_CONF = 0.25

# Soft red hint only (not a hard requirement).
RED_OVERLAP = 0.008

# Tiny SAM 2 model. First run downloads it. Follows the person you click.
SAM_TRACK_MODEL = "sam2_t.pt"

# Where we save the video with boxes drawn on it.
OUTPUT_FOLDER = Path("output")


