"""Main window: 3D viewport, Gemini companion / game / arm cards, the log and the chat, on a moonsilver glass layout."""
from __future__ import annotations

import sys
import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
                               QSlider, QVBoxLayout, QWidget)

from adapters.audio_io import AudioError, AudioPlayer, MicCapture
from adapters.gemini_client import GeminiModel
from features.chess_arm.chess_arm_controller import CHESS_ARM_GAME_TYPES, CHESS_ARM_ROBOT_SIDES, ChessArmController
from ui.glass_widgets import GlassCard, MetalButton, MoonsilverBackground
from ui.theme import STYLE_SHEET
from ui.viewport import SceneViewport

_TICK_MS = 33
_BOTTOM_HEIGHT = 190


class _UiPoster(QObject):
    """Lets worker threads run a function on the UI thread."""

    _call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._call.connect(lambda function: function(), Qt.ConnectionType.QueuedConnection)

    def post(self, function: Callable[[], None]) -> None:
        self._call.emit(function)


def _card(title: str) -> tuple[GlassCard, QVBoxLayout]:
    card = GlassCard()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)
    heading = QLabel(title.upper())
    heading.setObjectName("heading")
    layout.addWidget(heading)
    return card, layout


class MainWindow(QMainWindow):
    def __init__(self, controller_class: type[ChessArmController]) -> None:
        super().__init__()
        self.setWindowTitle("RobotArm - Chess companion")
        self.resize(1360, 820)
        self._poster = _UiPoster()
        self.controller = controller_class(self._poster.post, self._append_log)

        background = MoonsilverBackground()
        self.setCentralWidget(background)
        root = QHBoxLayout(background)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)
        header = QHBoxLayout()
        brand = QLabel("RobotArm  ·  6R Chess Player")
        brand.setObjectName("brand")
        self._status = QLabel("")
        self._status.setObjectName("status")
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        game_caption = QLabel("Game")
        game_caption.setObjectName("caption")
        self._game_type_combo = QComboBox()
        self._game_type_combo.addItems(CHESS_ARM_GAME_TYPES)
        self._game_type_combo.setToolTip("Board on the table. Switching starts that game from its initial position.")
        self._game_type_combo.currentTextChanged.connect(self._on_game_type_changed)
        header.addWidget(brand)
        header.addSpacing(18)
        header.addWidget(game_caption)
        header.addWidget(self._game_type_combo)
        header.addWidget(self._status, 1)
        left.addLayout(header)

        self._viewport = SceneViewport(self.controller.chess_arm_scene)
        self._viewport.clicked.connect(self._on_viewport_click)
        left.addWidget(self._viewport, 1)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFixedHeight(_BOTTOM_HEIGHT)
        bottom.addWidget(self._log_view, 1)
        bottom.addWidget(self._build_chat_card(), 1)
        left.addLayout(bottom)
        root.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(14)
        right.addWidget(self._build_gemini_card())
        right.addWidget(self._build_game_card())
        right.addWidget(self._build_arm_card())
        right.addStretch(1)
        side = QWidget()
        side.setFixedWidth(360)
        side.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)
        root.addWidget(side)

        self._player: AudioPlayer | None = None
        self._player_failed = False
        self._mic: MicCapture | None = None
        self.controller.state_listener = self._sync_from_controller
        companion = self.controller.companion
        companion.chat_listener = self._append_chat
        companion.status_listener = self._on_companion_status
        companion.audio_play = self._play_audio
        companion.audio_stop = self._stop_audio
        companion.audio_busy = self._audio_busy

        self._clock = time.perf_counter()
        self._last_texts: tuple = ()
        timer = QTimer(self)
        timer.timeout.connect(self._on_tick)
        timer.start(_TICK_MS)
        self._refresh_readouts()

    # ---------------------------------------------------------------- cards
    def _build_gemini_card(self) -> GlassCard:
        card, layout = _card("Gemini companion")
        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("API key (empty = use GEMINI_API_KEY)")
        self._key_edit.textChanged.connect(lambda text: setattr(self.controller, "api_key", text))
        layout.addWidget(self._key_edit)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.lineEdit().setPlaceholderText("Live model - press Refresh, or type a name")
        self._model_combo.setCurrentIndex(-1)
        self._model_combo.currentTextChanged.connect(lambda text: setattr(self.controller, "model_name", text))
        layout.addWidget(self._model_combo)

        self._refresh_button = MetalButton("Refresh models", "silver")
        self._refresh_button.clicked.connect(self._on_refresh_models)
        layout.addWidget(self._refresh_button)

        buttons = QHBoxLayout()
        self._connect_button = MetalButton("Connect companion", "black")
        self._connect_button.clicked.connect(self.controller.chess_arm_companion_toggle)
        self._mic_button = MetalButton("Mic: off", "silver")
        self._mic_button.clicked.connect(self._on_mic_toggle)
        buttons.addWidget(self._connect_button)
        buttons.addWidget(self._mic_button)
        layout.addLayout(buttons)
        note = QLabel("The engine plays; Gemini Live chats about the game. The key stays in memory only.")
        note.setObjectName("caption")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _build_chat_card(self) -> GlassCard:
        card, layout = _card("Chat with Gemini")
        card.setFixedHeight(_BOTTOM_HEIGHT)
        self._chat_view = QPlainTextEdit()
        self._chat_view.setReadOnly(True)
        layout.addWidget(self._chat_view, 1)
        self._chat_edit = QLineEdit()
        self._chat_edit.setPlaceholderText("Say something to Gemini (Enter), or turn the mic on and talk")
        self._chat_edit.returnPressed.connect(self._on_chat_send)
        layout.addWidget(self._chat_edit)
        return card

    def _build_game_card(self) -> GlassCard:
        card, layout = _card("Game")
        caption = QLabel("Robot (engine) plays")
        caption.setObjectName("caption")
        layout.addWidget(caption)
        self._side_combo = QComboBox()
        self._side_combo.addItems(CHESS_ARM_ROBOT_SIDES)
        self._side_combo.currentTextChanged.connect(self.controller.chess_arm_set_robot_side)
        layout.addWidget(self._side_combo)

        move_row = QHBoxLayout()
        self._move_edit = QLineEdit()
        self._move_edit.setPlaceholderText("Your move: e2e4 or Nf3")
        self._move_edit.returnPressed.connect(self._on_play_move)
        play_button = MetalButton("Play", "black")
        play_button.setFixedWidth(84)
        play_button.clicked.connect(self._on_play_move)
        move_row.addWidget(self._move_edit, 1)
        move_row.addWidget(play_button)
        layout.addLayout(move_row)

        button_row = QHBoxLayout()
        gemini_button = MetalButton("Robot move now", "silver")
        gemini_button.clicked.connect(self.controller.chess_arm_request_robot_move)
        new_button = MetalButton("New game", "black")
        new_button.clicked.connect(self._on_new_game)
        button_row.addWidget(gemini_button)
        button_row.addWidget(new_button)
        layout.addLayout(button_row)
        return card

    def _build_arm_card(self) -> GlassCard:
        card, layout = _card("Arm - 6 revolute joints")
        speed_row = QHBoxLayout()
        speed_caption = QLabel("Speed")
        speed_caption.setObjectName("caption")
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(20, 300)
        self._speed_slider.setValue(100)
        self._speed_slider.valueChanged.connect(lambda value: setattr(self.controller, "speed", value / 100.0))
        speed_row.addWidget(speed_caption)
        speed_row.addWidget(self._speed_slider, 1)
        layout.addLayout(speed_row)
        self._joint_labels = []
        for _ in range(6):
            label = QLabel("")
            label.setObjectName("mono")
            layout.addWidget(label)
            self._joint_labels.append(label)
        self._tool_label = QLabel("")
        self._tool_label.setObjectName("mono")
        self._tool_label.setWordWrap(True)
        layout.addWidget(self._tool_label)
        return card

    # -------------------------------------------------------------- actions
    def _append_log(self, text: str) -> None:
        if hasattr(self, "_log_view"):
            self._log_view.appendPlainText(text)

    def _on_game_type_changed(self, text: str) -> None:
        self.controller.chess_arm_set_game_type(text)
        self._sync_from_controller()

    def _sync_from_controller(self) -> None:
        """Show the game type and robot side the controller holds (the companion can change them by itself)."""
        is_chess = self.controller.chess_arm_is_chess()
        self._game_type_combo.blockSignals(True)
        self._game_type_combo.setCurrentText(self.controller.game_type)
        self._game_type_combo.blockSignals(False)
        self._side_combo.blockSignals(True)  # no signal: the controller already knows
        self._side_combo.setItemText(1, "White" if is_chess else "Red")
        self._side_combo.setCurrentIndex(CHESS_ARM_ROBOT_SIDES.index(self.controller.robot_side))
        self._side_combo.blockSignals(False)
        self._move_edit.setPlaceholderText("Your move: e2e4 or Nf3" if is_chess else "Your move: h2e2 (files a-i, ranks 0-9)")
        self._viewport.refresh()

    def _on_chat_send(self) -> None:
        text = self._chat_edit.text()
        self._chat_edit.clear()
        self.controller.chess_arm_companion_say(text)

    def _append_chat(self, line: str) -> None:
        self._chat_view.appendPlainText(line)

    def _on_companion_status(self, active: bool) -> None:
        self._connect_button.setText("Disconnect companion" if active else "Connect companion")
        if not active:
            self._stop_mic()

    def _on_mic_toggle(self) -> None:
        if self._mic is not None:
            self._stop_mic()
            return
        if not self.controller.chess_arm_companion_ensure_connected():
            return
        try:
            mic = MicCapture(self.controller.chess_arm_companion_audio)
            mic.start()
        except AudioError as failure:
            self._append_log(f"Microphone: {failure}")
            return
        self._mic = mic
        self._mic_button.setText("Mic: on")
        self._append_log("Microphone on. It is muted while Gemini speaks (use headphones to avoid echo).")

    def _stop_mic(self) -> None:
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        self._mic_button.setText("Mic: off")

    def _play_audio(self, pcm16_24k: bytes) -> None:
        if self._player is None and not self._player_failed:
            try:
                self._player = AudioPlayer()
            except AudioError as failure:
                self._player_failed = True
                self._append_log(f"Speaker: {failure}. The words still appear in the chat.")
        if self._player is not None:
            self._player.write(pcm16_24k)

    def _stop_audio(self) -> None:
        if self._player is not None:
            self._player.interrupt()

    def _audio_busy(self) -> bool:
        return self._player is not None and self._player.is_playing()

    def _on_refresh_models(self) -> None:
        self._refresh_button.setEnabled(False)
        self._append_log("Loading the model list from Gemini ...")
        self.controller.chess_arm_refresh_models(self._on_models_loaded)

    def _on_models_loaded(self, models: list[GeminiModel], error: str) -> None:
        self._refresh_button.setEnabled(True)
        if error:
            self._append_log(f"Could not load models: {error}")
            return
        previous = self._model_combo.currentText().strip()
        self._model_combo.clear()
        for model in models:
            self._model_combo.addItem(model.name)
            self._model_combo.setItemData(self._model_combo.count() - 1, model.display_name, Qt.ItemDataRole.ToolTipRole)
        names = [model.name for model in models]
        chatty = [name for name in names if "live" in name and "extended-thinking" not in name]  # thinking only slows a chat
        choice = previous if previous in names else (chatty[0] if chatty else (names[0] if names else ""))
        self._model_combo.setCurrentText(choice)
        self._append_log(f"{len(models)} models support the Live API. Selected: {choice or 'none'}.")

    def _on_play_move(self) -> None:
        if self.controller.chess_arm_submit_move_text(self._move_edit.text()):
            self._move_edit.clear()

    def _on_viewport_click(self, origin, direction) -> None:
        self.controller.chess_arm_click_ray(origin, direction)
        self._viewport.refresh()

    def _on_new_game(self) -> None:
        self.controller.chess_arm_new_game()
        self._viewport.refresh()

    # ----------------------------------------------------------------- tick
    def _on_tick(self) -> None:
        now = time.perf_counter()
        elapsed, self._clock = now - self._clock, now
        changed = self.controller.chess_arm_tick(min(elapsed, 0.1))
        self._viewport.set_fast(self.controller.state == "moving")
        if changed:
            self._viewport.refresh()
        self._refresh_readouts()

    def _refresh_readouts(self) -> None:
        texts = (self.controller.chess_arm_status_text(), *self.controller.chess_arm_joint_text(), self.controller.chess_arm_tool_text())
        if texts == self._last_texts:
            return
        self._last_texts = texts
        self._status.setText(texts[0])
        for label, text in zip(self._joint_labels, texts[1:7]):
            label.setText(text)
        self._tool_label.setText(texts[7])


def main_window_run(controller_class: type[ChessArmController]) -> None:
    """Start the Qt application and block until the window closes."""
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE_SHEET)
    window = MainWindow(controller_class)
    window.show()
    sys.exit(application.exec())
