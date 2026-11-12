"""Gemini Live API as a long conversation (WebSocket, text and voice in, voice out). Needs the `websockets` package.

One background thread owns the socket: it sends what other threads queue and reads what the model says. Everything the
model does is reported through one callback `on_event(kind, value)`, called on that thread:
  opened (None) | audio (bytes, 16-bit mono 24 kHz) | input_text (str, transcription of the person's voice)
  output_text (str, transcription of the model's voice) | turn_complete (None) | interrupted (None)
  closed (str reason, always the last event) | note (str, something worth logging)
The API key goes in the WebSocket URL, as the docs require; the URL is never logged and the key is scrubbed from errors.
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from typing import Callable

GEMINI_LIVE_INPUT_RATE = 16000
GEMINI_LIVE_OUTPUT_RATE = 24000
_LIVE_URL = ("wss://generativelanguage.googleapis.com/ws/"
             "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}")
_POLL_S = 0.02


class GeminiLiveSession:
    def __init__(self, api_key: str, model_name: str, system_instruction: str,
                 on_event: Callable[[str, object], None], thinking_level: str = "LOW") -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._on_event = on_event
        self._thinking_level = thinking_level
        self._outgoing: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop = threading.Event()
        self._open = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to close the socket; a 'closed' event follows."""
        self._stop.set()

    @property
    def is_open(self) -> bool:
        return self._open

    def send_text(self, text: str) -> None:
        """Queue a text message (it may be sent before the session is open)."""
        self._outgoing.put(("text", text))

    def send_audio(self, pcm16_16k: bytes) -> None:
        """Queue microphone audio. Dropped while the session is not open, because old speech is useless."""
        if self._open:
            self._outgoing.put(("audio", pcm16_16k))

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "<key>") if self._api_key else text

    def _setup_message(self) -> dict:
        generation: dict = {"responseModalities": ["AUDIO"]}
        if "extended-thinking" in self._model_name:  # these models refuse a session without a thinking level
            generation["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        return {"model": f"models/{self._model_name}", "generationConfig": generation,
                "systemInstruction": {"parts": [{"text": self._system_instruction}]},
                "inputAudioTranscription": {}, "outputAudioTranscription": {},
                "contextWindowCompression": {"slidingWindow": {}}}  # a long game must not hit the session length limit

    def _run(self) -> None:
        reason = "closed"
        try:
            from websockets.exceptions import ConnectionClosed, WebSocketException
            from websockets.sync.client import connect
        except ImportError:
            self._on_event("closed", "the 'websockets' package is missing: run  .venv\\Scripts\\python.exe -m pip install websockets")
            return
        try:
            with connect(_LIVE_URL.format(key=self._api_key), open_timeout=20, max_size=None) as socket:
                socket.send(json.dumps({"setup": self._setup_message()}))
                self._wait_for_setup(socket)
                self._open = True
                self._on_event("opened", None)
                while not self._stop.is_set():
                    self._send_queued(socket)
                    try:
                        raw = socket.recv(timeout=_POLL_S)
                    except TimeoutError:
                        continue
                    self._handle(json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")))
                reason = "disconnected"
        except ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else 0
            detail = closed.rcvd.reason if closed.rcvd else ""
            reason = self._scrub(f"Live connection closed (code {code}): {detail or 'no reason given'}")
        except (WebSocketException, OSError, TimeoutError, ValueError) as error:
            reason = self._scrub(f"Live connection failed: {type(error).__name__}: {error}")
        finally:
            self._open = False
            self._on_event("closed", reason)

    def _wait_for_setup(self, socket) -> None:
        deadline = time.monotonic() + 20.0
        while "setupComplete" not in self._read_message(socket, deadline):
            pass

    @staticmethod
    def _read_message(socket, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("no setupComplete from Live")
        raw = socket.recv(timeout=remaining)
        return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace"))

    def _send_queued(self, socket) -> None:
        while True:
            try:
                kind, payload = self._outgoing.get_nowait()
            except queue.Empty:
                return
            if kind == "text":
                socket.send(json.dumps({"realtimeInput": {"text": payload}}))
            else:
                data = base64.b64encode(payload).decode("ascii")
                socket.send(json.dumps({"realtimeInput": {"audio": {"data": data, "mimeType": f"audio/pcm;rate={GEMINI_LIVE_INPUT_RATE}"}}}))

    def _handle(self, message: dict) -> None:
        if "goAway" in message:
            self._on_event("note", "the server will close the session soon (goAway)")
        content = message.get("serverContent", {})
        if content.get("interrupted"):
            self._on_event("interrupted", None)
        for part in content.get("modelTurn", {}).get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                self._on_event("audio", base64.b64decode(inline["data"]))
        spoken_by_person = content.get("inputTranscription", {}).get("text")
        if spoken_by_person:
            self._on_event("input_text", spoken_by_person)
        spoken_by_model = content.get("outputTranscription", {}).get("text")
        if spoken_by_model:
            self._on_event("output_text", spoken_by_model)
        if content.get("turnComplete"):
            self._on_event("turn_complete", None)
