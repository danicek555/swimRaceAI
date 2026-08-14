# =============================================================================
# HOW THIS PROGRAM WORKS (big picture)
#
# You already have videos on your computer. You point this script at a folder.
# It does NOT download anything except model weights on the first run.
#
# Example:
#   python main.py -f videos -v test1.mp4 --track      <- the main mode
#   python main.py -f videos                           <- YOLO detection mode
#
# main.py is only the command line entry point. The real code lives in the
# swim/ package, split by topic (see README.md for the full map):
#   swim/audio.py     find the start beep in the sound track
#   swim/tracking.py  pick + follow ONE swimmer with SAM 2
#   swim/reaction.py  reaction time = beep -> feet leave the block
#   swim/hands.py     hand entry into the water + flight time
#   swim/pipeline.py  glues the steps together (run, track_one_swimmer)
#
# What --track does, step by step:
#   1. find_beep() listens to the audio: band-pass 800-3000 Hz, RMS loudness
#      in 20 ms slices, walk back from the peak to the beep START.
#   2. A picker window opens ~1 second BEFORE the beep (everyone is already
#      set on the blocks). You draw a tight box around ONE swimmer.
#   3. The frame you drew on becomes FRAME 0 of the tracked video. All times
#      and frame numbers printed from here on use that timeline, so they
#      match the position shown by your video player.
#   4. SAM 2 follows the swimmer frame by frame. Static red pixels (block,
#      lane ropes) are cut out of the mask so it cannot latch onto the block.
#   5. Reaction time: a crop is locked around the swimmer's OWN block (found
#      under their feet at beep time), other lanes are dimmed, and a vision
#      LM answers ON/LEFT per sample. An occlusion veto (frame difference
#      against the empty platform, shadows filtered out) overrides the LM
#      while anything still covers the platform. Two LEFT answers in a row
#      lock the leave time and sampling stops.
#   6. Hand entry: a free per-frame local scan looks for a persistent
#      surface disturbance inside the swimmer's own lane band; the LM only
#      confirms 2-3 crops around the candidate (full LM sweep is a fallback).
#   7. The output video in output/<video>/ gets burned-in times plus flashes:
#      red BEEP, green leave-block, cyan hand entry. Audit crops showing
#      exactly what the LM saw are saved next to it.
#
# Without --track (YOLO mode): detect_people() draws a box around every
# person on the starting blocks in the first seconds — detection only,
# no identity, no timing beyond the beep.
#
# Why look at sound for the start, not the picture?
#   A start beep is a short, loud, high tone. That is much easier to find in
#   the audio than by watching pixels.
# =============================================================================

# argparse reads flags from the terminal, like:
#   python main.py -f videos -v test1.mp4 --track
import argparse

from swim.pipeline import *  # noqa: F401,F403 — facade: all names available as before


def main() -> None:
    parser = argparse.ArgumentParser(description="Swim Race AI")

    # -f is the folder that already contains your videos.
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        help="Path to the folder that contains your videos",
    )
    # -v is optional. Leave it out to check every video in the folder.
    parser.add_argument(
        "-v",
        "--video",
        help="Optional: one video filename inside that folder",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Click one swimmer: SAM 2 tracks and measures reaction time after the beep",
    )
    parser.add_argument(
        "--known-rt",
        type=float,
        default=None,
        help="Optional: official RT (e.g. 0.62) to tune LEAVE_BLOCK_BODY_FRAC once",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip vision LM; use motion-only reaction time",
    )
    args = parser.parse_args()
    run(args.folder, args.video, args.track, args.known_rt, use_vlm=not args.no_vlm)


# This special check is True only when you run this file directly:
#   python main.py -f videos
# It is False if another file does: import main
# That way importing this file will not accidentally start the program.
if __name__ == "__main__":
    main()
