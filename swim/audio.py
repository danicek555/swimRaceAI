"""swimRaceAI — audio."""

import base64
import json
import os
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from .config import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403


def extract_audio(input_path: Path, wav_path: Path) -> None:
    """Turn the video into a short, simple WAV file with ffmpeg."""
    print(f"Extracting audio from {input_path.name}...")
    # ffmpeg is a separate program on your computer. We call it like a
    # terminal command. It reads the video and writes only the sound.
    command = [
        "ffmpeg",
        "-y",  # overwrite the output file if it already exists
        "-i", str(input_path),
        "-t", str(SEARCH_SECONDS),  # only the first SEARCH_SECONDS
        "-ac", "1",  # 1 channel = mono (easier to analyze)
        "-ar", "44100",  # 44100 samples per second
        "-sample_fmt", "s16",  # a common WAV format scipy can read
        str(wav_path),
    ]
    subprocess.run(command, check=True, capture_output=True)



def find_beep(wav_path: Path) -> float | None:
    """
    Find when the start beep happens.

    Returns the time in seconds, or None if it does not find a clear beep.
    In the first minute, broadcast intros can be louder than the starter, so
    the loudest band-pass spike is not enough: we prefer a SHORT burst with a
    sharp onset over sustained music / crowd noise.
    """
    # sample_rate = how many audio samples are in one second (we asked for 44100)
    # audio = a big list of numbers, one per sample
    sample_rate, audio = wavfile.read(wav_path)

    # If the file is stereo (2 columns), average the two speakers into one.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert from integers (like -32768 to 32767) into small floats (-1 to 1).
    # That way "loud" always means close to 1.0, no matter the file format.
    audio = audio.astype(np.float64)
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value

    # Keep only the first SEARCH_SECONDS (ffmpeg already trimmed, this is extra safety).
    max_samples = int(SEARCH_SECONDS * sample_rate)
    audio = audio[:max_samples]

    # Keep frequencies that sound like a beep, and remove rumble / splash.
    # A "band-pass filter" keeps a band of pitches and throws the rest away.
    # scipy wants the cutoffs as a fraction of the Nyquist frequency
    # (Nyquist = half the sample rate, here 22050 Hz).
    nyquist = sample_rate / 2
    low = BEEP_MIN_HZ / nyquist
    high = BEEP_MAX_HZ / nyquist
    filter_b, filter_a = butter(4, [low, high], btype="band")
    beep_band = filtfilt(filter_b, filter_a, audio)

    # Walk through the audio in small windows and measure loudness in each one.
    # Example: 44100 samples/sec * 0.020 sec = 882 samples per window.
    window_samples = int(sample_rate * WINDOW_MS / 1000)
    energies = []
    for start in range(0, len(beep_band) - window_samples, window_samples):
        window = beep_band[start : start + window_samples]
        # RMS = a common way to measure "how loud is this chunk?"
        # It is basically: square each sample, average them, then square-root.
        loudness = np.sqrt(np.mean(window ** 2))
        energies.append(loudness)

    energies = np.array(energies)
    if len(energies) == 0:
        return None

    # The beep should be much louder than typical background sound.
    typical = float(np.median(energies))
    if typical <= 0:
        return None
    threshold = typical * 4.0
    window_seconds = WINDOW_MS / 1000.0
    # About 0.5s of quiet history for the onset ratio.
    onset_lookback = max(1, int(round(0.5 / window_seconds)))

    best_index = None
    best_score = 0.0
    for index in range(1, len(energies) - 1):
        peak = float(energies[index])
        if peak < threshold:
            continue
        if peak < energies[index - 1] or peak < energies[index + 1]:
            continue

        # How long the tone stays hot around this peak. Music and commentary
        # linger; a starter beep collapses within a few tenths of a second.
        left = index
        while left > 0 and energies[left] > peak * 0.4:
            left -= 1
        right = index
        while right < len(energies) - 1 and energies[right] > peak * 0.4:
            right += 1
        duration = (right - left) * window_seconds
        if duration < 0.04 or duration > 0.55:
            continue

        previous = energies[max(0, index - onset_lookback) : index]
        baseline = float(np.median(previous)) if len(previous) else typical
        onset = peak / max(baseline, 1e-6)
        # Sharp, short, loud bursts win. Absolute loudness alone would pick
        # the intro music in many broadcast files.
        score = (peak / typical) * min(onset / 8.0, 3.0)
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        return None

    # Walk backward from the chosen peak to where the beep started.
    peak = float(energies[best_index])
    start_threshold = peak * 0.4
    start_index = best_index
    while start_index > 0 and energies[start_index] > start_threshold:
        start_index -= 1

    return start_index * window_seconds



def find_beep_for_video(video_path: Path) -> float | None:
    """Extract audio and return the start-beep time in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        audio_start = time.perf_counter()
        extract_audio(video_path, wav_path)
        log_time("Extract audio", audio_start)
        beep_start = time.perf_counter()
        beep_time = find_beep(wav_path)
        log_time("Find beep", beep_start)
    return beep_time



