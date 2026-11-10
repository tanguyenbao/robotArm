"""Feature: the robot arm plays chess, with Gemini choosing the robot's moves. No Qt in here.

The UI gives two callbacks: `post_to_ui` (runs a function on the UI thread, used to bring Gemini answers back from the
worker thread) and `log` (appends a line to the log panel).
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import chess
import numpy as np

from adapters.gemini_client import GeminiError, GeminiModel, gemini_client_generate_text, gemini_client_list_models
from core.arm_model import ARM_JOINT_NAMES, ArmGeometry, arm_model_home_joints
from core.arm_visual import ARM_GRIPPER_OPEN_MM
from core.chess_motion import ChessMotionError, MotionEvent, MotionSegment, chess_motion_plan
from core.chess_world import ChessWorld
from core.gemini_prompt import chess_move_from_text, gemini_prompt_build, gemini_prompt_parse_move
from core.world_scene import WorldScene, world_scene_build, world_scene_tool_position

CHESS_ARM_ROBOT_SIDES = ("Black", "White", "Both (Gemini vs Gemini)", "None (manual)")
_MAX_GEMINI_TRIES = 3
_HIGHLIGHT_RGB = (0.45, 0.62, 0.95)


def _smoothstep(fraction: float) -> float:
    return fraction * fraction * (3.0 - 2.0 * fraction)


class ChessArmController:
    """Game state, arm animation and the Gemini player."""

    def __init__(self, post_to_ui: Callable[[Callable[[], None]], None], log: Callable[[str], None]) -> None:
        self._post_to_ui = post_to_ui
        self._log = log
        self.geometry = ArmGeometry()
        self.world = ChessWorld()
        self.board = chess.Board()
        self.san_history: list[str] = []
        self.api_key = ""
        self.model_name = ""
        self.robot_side = CHESS_ARM_ROBOT_SIDES[0]
        self.speed = 1.0
        self.joints = arm_model_home_joints()
        self.gripper_mm = ARM_GRIPPER_OPEN_MM
        self.state = "idle"                    # idle, thinking or moving
        self._segments: list[MotionSegment] = []
        self._segment_index = 0
        self._segment_time = 0.0
        self._highlights: dict[int, tuple] = {}
        self._generation = 0                   # bumped on a new game so stale Gemini answers are ignored
        self._auto_blocked = False
        self._status_note = ""
        self.chess_arm_new_game()

    # ---------------------------------------------------------------- game
    def chess_arm_new_game(self) -> None:
        """Reset the board and send the arm home. Board orientation follows the robot side chosen now."""
        self._generation += 1
        self.board = chess.Board()
        self.san_history = []
        self.world.reset(self.board, black_near_robot=self.robot_side != "White")
        self.joints = arm_model_home_joints()
        self.gripper_mm = ARM_GRIPPER_OPEN_MM
        self._segments = []
        self._highlights = {}
        self._auto_blocked = False
        self._status_note = ""
        self.state = "idle"
        self._log("New game. White moves first.")
        self._maybe_auto_play()

    def chess_arm_set_robot_side(self, side: str) -> None:
        """Choose which colour(s) Gemini plays."""
        self.robot_side = side
        self._auto_blocked = False
        self._maybe_auto_play()

    def chess_arm_submit_move_text(self, text: str) -> bool:
        """Play a move typed by the user (UCI like e2e4, or SAN like Nf3). Returns True when it was accepted."""
        if self.state != "idle":
            self._log("The arm is busy, wait for it to finish.")
            return False
        if self.board.is_game_over():
            self._log("The game is over. Start a new game.")
            return False
        move = chess_move_from_text(text.strip(), self.board)
        if move is None:
            self._log(f"'{text.strip()}' is not a legal move here.")
            return False
        self._auto_blocked = False
        return self._start_move(move, "You")

    def chess_arm_request_robot_move(self) -> None:
        """Ask Gemini for a move for the side to move, now."""
        if self.state != "idle":
            self._log("The arm is busy, wait for it to finish.")
            return
        if self.board.is_game_over():
            self._log("The game is over. Start a new game.")
            return
        self._auto_blocked = False
        self._start_thinking()

    # -------------------------------------------------------------- gemini
    def chess_arm_refresh_models(self, done: Callable[[list[GeminiModel], str], None]) -> None:
        """Fetch the model list in a worker thread; `done(models, error)` runs on the UI thread."""
        api_key = self._resolve_api_key()

        def work() -> None:
            try:
                models, error = gemini_client_list_models(api_key), ""
            except GeminiError as failure:
                models, error = [], str(failure)
            self._post_to_ui(lambda: done(models, error))

        threading.Thread(target=work, daemon=True).start()

    def _resolve_api_key(self) -> str:
        return self.api_key.strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    def _start_thinking(self) -> None:
        model_name = self.model_name.strip()
        if not model_name:
            self._log("Pick a Gemini model first (press Refresh models).")
            self._auto_blocked = True
            return
        self.state = "thinking"
        generation = self._generation
        board_copy, history, api_key = self.board.copy(), list(self.san_history), self._resolve_api_key()
        side = "White" if board_copy.turn == chess.WHITE else "Black"
        self._log(f"Gemini ({model_name}) is thinking as {side} ...")

        def work() -> None:
            feedback = ""
            for attempt in range(1, _MAX_GEMINI_TRIES + 1):
                try:
                    answer = gemini_client_generate_text(api_key, model_name, gemini_prompt_build(board_copy, history, feedback))
                except GeminiError as failure:
                    self._post_to_ui(lambda message=str(failure): self._on_gemini_failed(generation, message))
                    return
                try:
                    move = gemini_prompt_parse_move(answer, board_copy)
                except ValueError as problem:
                    feedback = str(problem)
                    self._post_to_ui(lambda n=attempt, text=feedback: self._log(f"Gemini answer {n} refused: {text}"))
                    continue
                self._post_to_ui(lambda found=move: self._on_gemini_move(generation, found, model_name))
                return
            self._post_to_ui(lambda: self._on_gemini_failed(generation, f"no legal move after {_MAX_GEMINI_TRIES} tries"))

        threading.Thread(target=work, daemon=True).start()

    def _on_gemini_move(self, generation: int, move: chess.Move, model_name: str) -> None:
        if generation != self._generation or self.state != "thinking":
            return
        self.state = "idle"
        if not self._start_move(move, f"Gemini ({model_name})"):
            self._auto_blocked = True

    def _on_gemini_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.state = "idle"
        self._auto_blocked = True
        self._log(f"Gemini failed: {message}")

    def _maybe_auto_play(self) -> None:
        if self.state != "idle" or self._auto_blocked or self.board.is_game_over():
            return
        turn_is_white = self.board.turn == chess.WHITE
        wanted = (self.robot_side == CHESS_ARM_ROBOT_SIDES[2]
                  or (self.robot_side == "White" and turn_is_white)
                  or (self.robot_side == "Black" and not turn_is_white))
        if wanted:
            self._start_thinking()

    # ---------------------------------------------------------------- moves
    def _start_move(self, move: chess.Move, who: str) -> bool:
        try:
            segments = chess_motion_plan(self.world, self.board, move, self.geometry, self.joints, self.gripper_mm)
        except ChessMotionError as failure:
            self._log(f"Cannot play {move.uci()}: {failure}")
            return False
        san = self.board.san(move)
        number = self.board.fullmove_number
        prefix = f"{number}. " if self.board.turn == chess.WHITE else f"{number}... "
        self.board.push(move)
        self.san_history.append(san)
        self._highlights = {move.from_square: _HIGHLIGHT_RGB, move.to_square: _HIGHLIGHT_RGB}
        self._log(f"{prefix}{san}  ({move.uci()}) by {who}")
        self._segments, self._segment_index, self._segment_time = segments, 0, 0.0
        self.state = "moving"
        return True

    def chess_arm_tick(self, seconds: float) -> bool:
        """Advance the animation. Returns True when the picture changed."""
        if self.state != "moving":
            return False
        segment = self._segments[self._segment_index]
        self._segment_time += seconds * self.speed
        fraction = min(1.0, self._segment_time / segment.duration_s)
        eased = _smoothstep(fraction)
        position = eased * (len(segment.joint_samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(segment.joint_samples) - 1)
        blend = position - lower
        self.joints = segment.joint_samples[lower] * (1.0 - blend) + segment.joint_samples[upper] * blend
        self.gripper_mm = segment.gripper_start_mm + (segment.gripper_end_mm - segment.gripper_start_mm) * eased
        self._status_note = segment.label
        if fraction >= 1.0:
            self.joints = segment.joint_samples[-1].copy()
            self.gripper_mm = segment.gripper_end_mm
            self._apply_event(segment.event)
            self._segment_index += 1
            self._segment_time = 0.0
            if self._segment_index >= len(self._segments):
                self._finish_move()
        return True

    def _apply_event(self, event: MotionEvent | None) -> None:
        if event is None:
            return
        if event.kind == "pick":
            self.world.pick(event.piece_id)
        elif event.kind == "drop":
            self.world.drop(event.piece_id, event.x_mm, event.y_mm, event.square)
        elif event.kind == "promote":
            self.world.promote(event.piece_id, event.new_kind)

    def _finish_move(self) -> None:
        self.state = "idle"
        self._segments = []
        self._status_note = ""
        if self.board.is_game_over():
            outcome = self.board.outcome()
            reason = outcome.termination.name.replace("_", " ").lower() if outcome else "game over"
            self._log(f"Game over: {self.board.result()} ({reason}).")
            return
        if self.board.is_check():
            self._log("Check.")
        self._maybe_auto_play()

    # ----------------------------------------------------------- read-outs
    def chess_arm_scene(self) -> WorldScene:
        """Meshes for the viewport."""
        return world_scene_build(self.world, self.geometry, self.joints, self.gripper_mm, self._highlights)

    def chess_arm_joint_text(self) -> list[str]:
        """One 'Name  +12.3 deg' line per joint."""
        return [f"{name}  {np.degrees(angle):+7.1f}°" for name, angle in zip(ARM_JOINT_NAMES, self.joints)]

    def chess_arm_tool_text(self) -> str:
        """Tool centre point and gripper opening."""
        x_mm, y_mm, z_mm = world_scene_tool_position(self.geometry, self.joints)
        return f"Tool  X {x_mm:6.1f}  Y {y_mm:6.1f}  Z {z_mm:6.1f} mm    Gripper {self.gripper_mm:4.1f} mm"

    def chess_arm_status_text(self) -> str:
        """Who is to move and what the arm is doing."""
        if self.board.is_game_over():
            return f"Game over  {self.board.result()}"
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        if self.state == "thinking":
            return f"{turn} to move  -  Gemini is thinking"
        if self.state == "moving":
            return f"Arm moving  -  {self._status_note}"
        return f"{turn} to move"
