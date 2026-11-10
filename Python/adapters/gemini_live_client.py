"""Gemini Live API client (WebSocket, text in / text out). Needs the `websockets` package.

Live models answer in audio. The text is taken from the audio transcription (outputAudioTranscription), so the
audio bytes are received but never decoded or played. The API key goes in the WebSocket URL, as the docs require;
the URL is never logged and the key is scrubbed from every error message.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from adapters.gemini_client import GeminiError

_LIVE_URL = ("wss://generativelanguage.googleapis.com/ws/"
             "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}")


def _scrub(text: str, api_key: str) -> str:
    return text.replace(api_key, "<key>") if api_key else text


def gemini_live_generate_text(api_key: str, model_name: str, prompt: str, timeout_s: float = 60.0,
                              response_modality: str = "AUDIO", thinking_level: str = "HIGH",
                              trace: Callable[[str], None] | None = None, system_instruction: str = "",
                              accept: Callable[[str], bool] | None = None, follow_up: str = "",
                              max_follow_ups: int = 2) -> str:
    """One text prompt in, the model's text out (from the transcription when the modality is AUDIO).

    thinking_level (LOW / MEDIUM / HIGH) is sent only to '...extended-thinking' models.
    Live models like to say a filler line ("Let me evaluate ...") and end the turn. When `accept` says the text so far
    is not good enough, `follow_up` is sent on the same session (up to `max_follow_ups` times) and the new words are
    added to the text."""
    if not api_key:
        raise GeminiError("no API key (type it in the field or set GEMINI_API_KEY)")
    if not model_name:
        raise GeminiError("no model selected")
    try:
        from websockets.sync.client import connect
        from websockets.exceptions import ConnectionClosed, WebSocketException
    except ImportError:
        raise GeminiError("the 'websockets' package is missing: run  .venv\\Scripts\\python.exe -m pip install websockets") from None

    setup: dict = {"model": f"models/{model_name}", "generationConfig": {"responseModalities": [response_modality]}}
    if response_modality == "AUDIO":
        setup["outputAudioTranscription"] = {}
    if "extended-thinking" in model_name:  # these models refuse a session without a level; plain 3.8-live refuses one with it
        setup["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level}
    if system_instruction:
        setup["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    deadline = time.monotonic() + timeout_s
    pieces: list[str] = []
    turn_done = lambda message: message.get("serverContent", {}).get("turnComplete", False)  # noqa: E731
    try:
        with connect(_LIVE_URL.format(key=api_key), open_timeout=20, max_size=None) as socket:
            socket.send(json.dumps({"setup": setup}))
            _read_until(socket, deadline, lambda message: "setupComplete" in message, [])
            next_words = prompt
            for turn in range(max_follow_ups + 1):
                socket.send(json.dumps({"realtimeInput": {"text": next_words}}))
                turn_pieces: list[str] = []
                _read_until(socket, deadline, turn_done, turn_pieces, trace)
                pieces.extend(turn_pieces + [" "])
                if accept is None or not follow_up or accept("".join(pieces)):
                    break
                if trace is not None:
                    trace(f"no usable answer yet, sending follow-up {turn + 1}/{max_follow_ups}")
                next_words = follow_up
    except ConnectionClosed as closed:
        code = closed.rcvd.code if closed.rcvd else 0
        reason = closed.rcvd.reason if closed.rcvd else ""
        raise GeminiError(_scrub(f"Live connection closed (code {code}): {reason or 'no reason given'}", api_key), code) from None
    except GeminiError:
        raise
    except (WebSocketException, OSError, TimeoutError) as error:
        raise GeminiError(_scrub(f"Live connection failed: {type(error).__name__}: {error}", api_key)) from None
    text = "".join(pieces).strip()
    if not text:
        raise GeminiError("Live answered but sent no text (no transcription and no text part)")
    return text


def _describe(message: dict) -> str:
    """One short line about a server message for the trace: its keys, flags and text, never audio bytes."""
    content = message.get("serverContent", {})
    notes = [key for key in message if key != "serverContent"]
    for flag in ("turnComplete", "generationComplete", "interrupted"):
        if content.get(flag):
            notes.append(flag)
    for key in ("inputTranscription", "outputTranscription"):
        if content.get(key, {}).get("text"):
            notes.append(f"{key}={content[key]['text']!r}")
    for part in content.get("modelTurn", {}).get("parts", []):
        kind = "thought-text" if part.get("thought") else "text"
        if part.get("text"):
            notes.append(f"{kind}={part['text']!r}")
        elif "inlineData" in part:
            notes.append(f"audio({len(part['inlineData'].get('data', ''))} b64 chars)")
        else:
            notes.append(f"part:{','.join(part)}")
    known = {"turnComplete", "generationComplete", "interrupted", "inputTranscription", "outputTranscription", "modelTurn"}
    notes += [f"serverContent.{key}" for key in content if key not in known]
    return "; ".join(notes) or "empty message"


def _read_until(socket, deadline: float, done, pieces: list[str], trace=None) -> None:
    """Read server messages until `done(message)`; collect transcription / text pieces on the way."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GeminiError("Live timed out waiting for the model")
        try:
            raw = socket.recv(timeout=remaining)
        except TimeoutError:
            raise GeminiError("Live timed out waiting for the model") from None
        message = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace"))
        content = message.get("serverContent", {})
        if trace is not None:
            trace(_describe(message))
        transcription = content.get("outputTranscription", {}).get("text")
        if transcription:
            pieces.append(transcription)
        else:  # text parts only count when there is no transcription, so nothing is counted twice
            for part in content.get("modelTurn", {}).get("parts", []):
                if part.get("text"):
                    pieces.append(part["text"])
        if done(message):
            return
