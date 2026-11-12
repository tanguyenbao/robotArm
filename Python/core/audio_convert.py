"""Raw PCM conversion between Gemini's formats (16-bit mono) and whatever the sound card wants (pure numpy).

Gemini Live takes 16 kHz 16-bit mono in and gives 24 kHz 16-bit mono out. When the sound card refuses those, the
adapter asks it for its preferred format and converts here.
"""
from __future__ import annotations

import numpy as np

_SAMPLE_FORMATS = {"Int16": (np.int16, 32768.0, 0.0), "Int32": (np.int32, 2147483648.0, 0.0),
                   "UInt8": (np.uint8, 128.0, 128.0), "Float": (np.float32, 1.0, 0.0)}


def audio_convert_resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a mono float signal: block average for whole-number downsampling, linear interpolation otherwise."""
    if from_rate == to_rate or samples.size == 0:
        return samples
    if from_rate % to_rate == 0:
        ratio = from_rate // to_rate
        usable = samples.size // ratio * ratio
        return samples[:usable].reshape(-1, ratio).mean(axis=1) if usable else samples[:0]
    count = max(1, int(round(samples.size * to_rate / from_rate)))
    return np.interp(np.linspace(0.0, samples.size - 1, count), np.arange(samples.size), samples)


def audio_convert_from_device(raw: bytes, rate: int, channels: int, sample_format: str, target_rate: int) -> bytes:
    """Microphone data in the device format -> 16-bit mono PCM at `target_rate`."""
    dtype, scale, offset = _SAMPLE_FORMATS[sample_format]
    samples = np.frombuffer(raw[:len(raw) // np.dtype(dtype).itemsize // channels * np.dtype(dtype).itemsize * channels],
                            dtype=dtype).astype(np.float32)
    if samples.size == 0:
        return b""
    mono = ((samples - offset) / scale).reshape(-1, channels).mean(axis=1)
    resampled = audio_convert_resample(mono, rate, target_rate)
    return (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def audio_convert_to_device(pcm16_mono: bytes, source_rate: int, rate: int, channels: int, sample_format: str) -> bytes:
    """16-bit mono PCM at `source_rate` -> device format (rate, channel count, sample format)."""
    dtype, scale, offset = _SAMPLE_FORMATS[sample_format]
    mono = np.frombuffer(pcm16_mono[:len(pcm16_mono) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    resampled = np.repeat(audio_convert_resample(mono, source_rate, rate)[:, None], channels, axis=1).reshape(-1)
    if np.issubdtype(dtype, np.floating):
        return resampled.astype(dtype).tobytes()
    limit = scale - 1.0
    return np.clip(np.round(resampled * scale + offset), -scale + offset, limit + offset).astype(dtype).tobytes()
