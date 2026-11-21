"""Load audio into the mono 16 kHz float32 waveform the model expects."""

from __future__ import annotations

import io
from math import gcd
from pathlib import Path
from typing import Union

import numpy as np

from .model import SAMPLE_RATE

AudioInput = Union[str, Path, bytes, bytearray, np.ndarray]


def load_audio(audio: AudioInput, sample_rate: int | None = None) -> np.ndarray:
    """Return `audio` as a mono float32 waveform at 16 kHz.

    `audio` is a file path, encoded file bytes (e.g. the MP3 that `speak`
    returns), or a numpy waveform. A waveform needs `sample_rate` unless it
    is already at 16 kHz; for paths and bytes the rate is read from the file.
    """
    if isinstance(audio, np.ndarray):
        waveform = audio
        rate = SAMPLE_RATE if sample_rate is None else sample_rate
    else:
        try:
            import soundfile
        except ImportError as exc:
            raise RuntimeError(
                "Decoding audio files needs soundfile: "
                "pip install 'ancient-language-tts[check]'"
            ) from exc
        source = io.BytesIO(bytes(audio)) if isinstance(audio, (bytes, bytearray)) else str(audio)
        waveform, rate = soundfile.read(source, dtype="float32", always_2d=False)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        # soundfile gives (frames, channels); a raw array might be (channels, frames).
        axis = 1 if waveform.shape[1] <= waveform.shape[0] else 0
        waveform = waveform.mean(axis=axis)
    return resample(waveform, rate, SAMPLE_RATE)


def resample(waveform: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if rate == target:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError:
        # Linear interpolation: fine for recognition, if a little muffled.
        n_out = int(round(len(waveform) * target / rate))
        x_out = np.arange(n_out) * (rate / target)
        return np.interp(x_out, np.arange(len(waveform)), waveform).astype(np.float32)
    g = gcd(int(rate), int(target))
    return resample_poly(waveform, target // g, int(rate) // g).astype(np.float32)
