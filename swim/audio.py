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
        "-t", str(SEARCH_SECONDS),  # only the first 15 seconds
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
    # argmax = index of the biggest number. median = a typical/middle value,
    # which is less fooled by one huge spike than an average would be.
    peak_index = int(np.argmax(energies))
    peak = energies[peak_index]
    typical = np.median(energies)
    if peak < typical * 4:
        return None

    # Walk backward from the loudest moment to where the beep started.
    # The peak is often the MIDDLE of the beep. We want the START.
    # Keep stepping left while the slice is still at least 40% as loud as the peak.
    start_threshold = peak * 0.4
    start_index = peak_index
    while start_index > 0 and energies[start_index] > start_threshold:
        start_index -= 1

    # Convert "which window?" into seconds.
    # Window 0 = 0.000s, window 1 = 0.020s, window 92 = 1.840s, and so on.
    window_seconds = WINDOW_MS / 1000
    beep_time = start_index * window_seconds
    return beep_time



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



