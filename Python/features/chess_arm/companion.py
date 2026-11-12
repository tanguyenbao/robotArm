"""Feature: Gemini Live as the player's companion while the engine plays. No Qt in here.

The companion keeps one Live session open. Game events (text) and the person's voice / typing go in; the model's voice
comes out as audio plus a transcription for the chat view. The UI wires the sound devices through the hooks below.
"""
from __future__ import annotations

from typing import Callable

from adapters.gemini_live_session import GeminiLiveSession
from core.companion_prompt import COMPANION_SYSTEM_PROMPT


def _ignore(*_arguments) -> None:
    return None


class ChessCompanion:
    def __init__(self, post_to_ui: Callable[[Callable[[], None]], None], log: Callable[[str], None],
                 describe_game: Callable[[], str]) -> None:
        self._post_to_ui = post_to_ui
        self._log = log
        self._describe_game = describe_game      # text for the first message of a (re)started session
        self._session: GeminiLiveSession | None = None
        self._answer_pending = False             # an event was sent and its answer has not finished yet
        self._person_words: list[str] = []
        self._model_words: list[str] = []
        # hooks the UI sets
        self.chat_listener: Callable[[str], None] = _ignore          # one finished chat line
        self.status_listener: Callable[[bool], None] = _ignore       # True while connected / connecting
        self.audio_play: Callable[[bytes], None] = _ignore           # model voice, 16-bit mono 24 kHz
        self.audio_stop: Callable[[], None] = _ignore                # the person interrupted: drop queued voice
        self.audio_busy: Callable[[], bool] = lambda: False          # True while the voice is still sounding

    def companion_is_active(self) -> bool:
        """True from pressing Connect until the session closes."""
        return self._session is not None

    def companion_is_speaking(self) -> bool:
        return self.audio_busy()

    def companion_is_busy(self) -> bool:
        """True while an answer is still being generated or sounding (from sending an event until its turn ends)."""
        return self._session is not None and (self._answer_pending or self.audio_busy())

    def companion_connect(self, api_key: str, model_name: str) -> bool:
        """Open the Live session. Returns False (after logging why) when it cannot start."""
        if self._session is not None:
            return True
        if not api_key:
            self._log("Companion: no API key (type it in the field or set GEMINI_API_KEY).")
            return False
        if not model_name:
            self._log("Companion: pick a Live model first (press Refresh models).")
            return False
        session = GeminiLiveSession(api_key, model_name, COMPANION_SYSTEM_PROMPT,
                                    lambda kind, value: self._post_to_ui(lambda: self._on_event(session, kind, value)))
        self._session = session
        self._log(f"Companion: connecting to {model_name} ...")
        session.send_text(self._describe_game())
        self.status_listener(True)
        session.start()
        return True

    def companion_disconnect(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
            self.audio_stop()
            self.status_listener(False)
            self._log("Companion: disconnected.")

    def companion_notify(self, event_text: str) -> None:
        """A game event for the model to react to. Ignored while not connected."""
        if self._session is not None:
            self._answer_pending = True
            self._session.send_text(event_text)

    def companion_say(self, text: str) -> bool:
        """The person typed something to the model. Returns False when there is no session."""
        if self._session is None:
            return False
        self._flush_person()
        self.chat_listener(f"You: {text}")
        self._session.send_text(text)
        return True

    def companion_send_audio(self, pcm16_16k: bytes) -> None:
        """Microphone data. Dropped while the model's voice plays, so it does not hear itself."""
        if self._session is not None and not self.audio_busy():
            self._session.send_audio(pcm16_16k)

    # --------------------------------------------------------------- session
    def _on_event(self, session: GeminiLiveSession, kind: str, value: object) -> None:
        if session is not self._session and kind != "closed":
            return  # an old session still talking
        if kind == "opened":
            self._log("Companion: connected.")
        elif kind == "audio":
            self.audio_play(value)
        elif kind == "input_text":
            self._person_words.append(str(value))
        elif kind == "output_text":
            self._flush_person()
            self._model_words.append(str(value))
        elif kind == "turn_complete":
            self._answer_pending = False
            self._flush_person()
            self._flush_model()
        elif kind == "interrupted":
            self.audio_stop()
            self._flush_model(" ...")
        elif kind == "note":
            self._log(f"Companion: {value}")
        elif kind == "closed":
            self._flush_person()
            self._flush_model()
            if session is self._session:
                self._session = None
                self._answer_pending = False
                self.status_listener(False)
                self.audio_stop()
                self._log(f"Companion: {value}")

    def _flush_person(self) -> None:
        text = "".join(self._person_words).strip()
        self._person_words.clear()
        if text:
            self.chat_listener(f"You (voice): {text}")

    def _flush_model(self, suffix: str = "") -> None:
        text = "".join(self._model_words).strip()
        self._model_words.clear()
        if text:
            self.chat_listener(f"Gemini: {text}{suffix}")
