"""Speaker playback and microphone capture through Qt Multimedia. Needs the Qt application; use on the UI thread only.

Gemini wants 16 kHz 16-bit mono in and gives 24 kHz 16-bit mono out. When the sound card does not take those formats,
its preferred format is used and the audio is converted (core/audio_convert).
"""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QAudioSource, QMediaDevices

from core.audio_convert import audio_convert_from_device, audio_convert_to_device
from adapters.gemini_live_session import GEMINI_LIVE_INPUT_RATE, GEMINI_LIVE_OUTPUT_RATE

_FLUSH_MS = 25


class AudioError(Exception):
    """No usable sound device."""


def _format(rate: int) -> QAudioFormat:
    wanted = QAudioFormat()
    wanted.setSampleRate(rate)
    wanted.setChannelCount(1)
    wanted.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return wanted


class AudioPlayer:
    """Plays the model's voice. write() takes 16-bit mono 24 kHz PCM and queues it."""

    def __init__(self) -> None:
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            raise AudioError("no audio output device")
        self._format = _format(GEMINI_LIVE_OUTPUT_RATE)
        self._convert = not device.isFormatSupported(self._format)
        if self._convert:
            self._format = device.preferredFormat()
        self._sink = QAudioSink(device, self._format)
        self._sink.setBufferSize(self._format.bytesForDuration(1_000_000))
        self._device = self._sink.start()
        self._pending = bytearray()
        self._playing_until = 0.0
        self._timer = QTimer()
        self._timer.setInterval(_FLUSH_MS)
        self._timer.timeout.connect(self._flush)

    def write(self, pcm16_24k: bytes) -> None:
        data = pcm16_24k
        if self._convert:
            data = audio_convert_to_device(pcm16_24k, GEMINI_LIVE_OUTPUT_RATE, self._format.sampleRate(),
                                           self._format.channelCount(), self._format.sampleFormat().name)
        self._playing_until = max(time.monotonic(), self._playing_until) + len(pcm16_24k) / (2.0 * GEMINI_LIVE_OUTPUT_RATE)
        self._pending += data
        self._flush()
        if self._pending:
            self._timer.start()

    def _flush(self) -> None:
        if self._pending:
            written = self._device.write(bytes(self._pending[:max(0, self._sink.bytesFree())]))
            del self._pending[:max(0, written)]
        if not self._pending:
            self._timer.stop()

    def is_playing(self) -> bool:
        """True while queued speech has not finished sounding."""
        return time.monotonic() < self._playing_until

    def interrupt(self) -> None:
        """Drop everything queued (the person talked over the model)."""
        self._pending.clear()
        self._timer.stop()
        self._sink.stop()
        self._device = self._sink.start()
        self._playing_until = 0.0


class MicCapture:
    """Reads the microphone and hands 16-bit mono 16 kHz PCM chunks to `on_chunk`."""

    def __init__(self, on_chunk: Callable[[bytes], None]) -> None:
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            raise AudioError("no microphone found")
        self._on_chunk = on_chunk
        self._format = _format(GEMINI_LIVE_INPUT_RATE)
        self._convert = not device.isFormatSupported(self._format)
        if self._convert:
            self._format = device.preferredFormat()
        self._source = QAudioSource(device, self._format)
        self._device = None

    def start(self) -> None:
        self._device = self._source.start()
        if self._device is None:
            raise AudioError("the microphone could not be opened")
        self._device.readyRead.connect(self._read)

    def _read(self) -> None:
        raw = bytes(self._device.readAll().data())
        if not raw:
            return
        if self._convert:
            raw = audio_convert_from_device(raw, self._format.sampleRate(), self._format.channelCount(),
                                            self._format.sampleFormat().name, GEMINI_LIVE_INPUT_RATE)
        if raw:
            self._on_chunk(raw)

    def stop(self) -> None:
        self._source.stop()
        self._device = None
