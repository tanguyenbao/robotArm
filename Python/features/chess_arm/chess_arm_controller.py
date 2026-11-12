"""Feature: the robot arm plays chess or xiangqi, with Gemini choosing the robot's moves. No Qt in here.

The UI gives two callbacks: `post_to_ui` (runs a function on the UI thread, used to bring Gemini answers back from the
worker thread) and `log` (appends a line to the log panel).

Squares are integers in both games (chess: python-chess square, xiangqi: rank * 9 + file), so selection, highlights and
the arm animation are shared. "White" in the robot-side list means Red when xiangqi is on the table.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import chess
import numpy as np

from adapters.gemini_client import GeminiError, GeminiModel, gemini_client_generate_text, gemini_client_list_models
from adapters.gemini_live_client import gemini_live_generate_text
from adapters.glyph_bitmap import glyph_bitmap_render
from core.arm_model import ARM_JOINT_NAMES, ArmGeometry, arm_model_home_joints
from core.arm_visual import ARM_GRIPPER_OPEN_MM
from core.chess_motion import ChessMotionError, MotionEvent, MotionSegment, chess_motion_plan, xiangqi_motion_plan
from core.chess_pieces import CHESS_PIECE_HEIGHT_MM
from core.chess_world import ChessWorld
from core.gemini_prompt import (GEMINI_LIVE_FOLLOW_UP, GEMINI_LIVE_SYSTEM_PROMPT, chess_move_from_text,
                                gemini_prompt_build, gemini_prompt_parse_move)
from core.world_scene import WorldScene, world_scene_build, world_scene_tool_position
from core.xiangqi_board import XIANGQI_DISC_HEIGHT_MM, XIANGQI_PIECE_CHARS
from core.xiangqi_prompt import (XIANGQI_LIVE_FOLLOW_UP, XIANGQI_LIVE_SYSTEM_PROMPT, xiangqi_prompt_build,
                                 xiangqi_prompt_parse_move)
from core.xiangqi_rules import (XiangqiMove, XiangqiPosition, xiangqi_rules_in_check, xiangqi_rules_legal_moves,
                                xiangqi_rules_move_from_text, xiangqi_rules_play, xiangqi_rules_start,
                                xiangqi_rules_winner)
from core.xiangqi_scene import xiangqi_scene_build
from core.xiangqi_world import XiangqiWorld

CHESS_ARM_GAME_TYPES = ("Chess", "Xiangqi (Cờ tướng)")
CHESS_ARM_ROBOT_SIDES = ("Black", "White", "Both (Gemini vs Gemini)", "None (manual)")
_MAX_GEMINI_TRIES = 3
_HIGHLIGHT_RGB = (0.45, 0.62, 0.95)
_SELECTED_RGB = (0.96, 0.82, 0.40)
_CAPTURE_RGB = (0.93, 0.50, 0.50)
_PICK_HEIGHTS_MM = (70.0, 45.0, 22.0)
_PICK_RADIUS_MM = 15.0
_XIANGQI_PICK_HEIGHT_MM = XIANGQI_DISC_HEIGHT_MM / 2
_GLYPH_CELLS = 18


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
        self.game_type = CHESS_ARM_GAME_TYPES[0]
        self.xiangqi_world = XiangqiWorld()
        self.xiangqi_position: XiangqiPosition = xiangqi_rules_start()
        self.xiangqi_history: list[str] = []
        self._xiangqi_glyphs: dict[tuple[str, bool], tuple[str, ...]] | None = None
        self.api_key = ""
        self.use_live = False  # True: ask Gemini through the Live API (WebSocket) instead of generateContent
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
        self._last_move_squares: tuple[int, ...] = ()
        self._selected: int | None = None
        self._targets: dict[int, chess.Move | XiangqiMove] = {}
        self._generation = 0                   # bumped on a new game so stale Gemini answers are ignored
        self._auto_blocked = False
        self._status_note = ""
        self.chess_arm_new_game()

    # ---------------------------------------------------------------- game
    def chess_arm_is_chess(self) -> bool:
        """True while the chess set is on the table (False: xiangqi)."""
        return self.game_type == CHESS_ARM_GAME_TYPES[0]

    def chess_arm_set_game_type(self, game_type: str) -> None:
        """Swap the board on the table. Switching starts that game from its initial position."""
        if game_type not in CHESS_ARM_GAME_TYPES or game_type == self.game_type:
            return
        self.game_type = game_type
        self.chess_arm_new_game()

    def _is_game_over(self) -> bool:
        if self.chess_arm_is_chess():
            return self.board.is_game_over()
        return xiangqi_rules_winner(self.xiangqi_position) is not None

    def _red_or_white_to_move(self) -> bool:
        return self.board.turn == chess.WHITE if self.chess_arm_is_chess() else self.xiangqi_position.red_to_move

    def _active_world(self) -> ChessWorld | XiangqiWorld:
        return self.world if self.chess_arm_is_chess() else self.xiangqi_world

    def chess_arm_new_game(self) -> None:
        """Reset the board and send the arm home. The robot's own side ends up nearest to the arm."""
        self._generation += 1
        flipped = self.robot_side != "White"  # black stands nearest to the arm unless the robot plays white / red
        if self.chess_arm_is_chess():
            self.board = chess.Board()
            self.san_history = []
            self.world.reset(self.board, black_near_robot=flipped)
            first_mover = "White"
        else:
            self.xiangqi_position = xiangqi_rules_start()
            self.xiangqi_history = []
            self.xiangqi_world.reset(self.xiangqi_position, flipped=flipped)
            first_mover = "Red"
        self.joints = arm_model_home_joints()
        self.gripper_mm = ARM_GRIPPER_OPEN_MM
        self._segments = []
        self._last_move_squares = ()
        self._clear_selection()
        self._auto_blocked = False
        self._status_note = ""
        self.state = "idle"
        self._log(f"New {'chess' if self.chess_arm_is_chess() else 'xiangqi'} game. {first_mover} moves first.")
        self._maybe_auto_play()

    def chess_arm_set_robot_side(self, side: str) -> None:
        """Choose which colour(s) Gemini plays ('Red' is the xiangqi name of 'White')."""
        self.robot_side = "White" if side == "Red" else side
        self._auto_blocked = False
        self._maybe_auto_play()

    def chess_arm_submit_move_text(self, text: str) -> bool:
        """Play a move typed by the user (chess: UCI e2e4 or SAN Nf3; xiangqi: ICCS h2e2). Returns True when accepted."""
        if self.state != "idle":
            self._log("The arm is busy, wait for it to finish.")
            return False
        if self._is_game_over():
            self._log("The game is over. Start a new game.")
            return False
        if self.chess_arm_is_chess():
            move = chess_move_from_text(text.strip(), self.board)
        else:
            move = xiangqi_rules_move_from_text(text, self.xiangqi_position)
        if move is None:
            self._log(f"'{text.strip()}' is not a legal move here.")
            return False
        self._auto_blocked = False
        return self._start_move(move, "You")

    def chess_arm_click_ray(self, origin: np.ndarray, direction: np.ndarray) -> None:
        """The user clicked the 3D view: find the square (or piece) under the ray and handle it."""
        if direction[2] >= 0:
            return
        ground_hit = origin + (0.0 - origin[2]) / direction[2] * direction
        if not self.chess_arm_is_chess():
            self._click_ray_xiangqi(origin, direction, ground_hit)
            return
        ground_square = self.world.square_at(ground_hit[0], ground_hit[1])
        if self._selected is not None and ground_square in self._targets:
            self.chess_arm_click_square(ground_square)  # a lit tile wins over a piece standing in front of it
            return
        for height in (*_PICK_HEIGHTS_MM, 0.0):
            t = (height - origin[2]) / direction[2]
            hit = origin + t * direction
            square = self.world.square_at(hit[0], hit[1])
            if square is None:
                continue
            piece = self.world.piece_on(square)
            on_body = (piece is not None and CHESS_PIECE_HEIGHT_MM[piece.kind] >= height
                       and np.hypot(hit[0] - piece.x_mm, hit[1] - piece.y_mm) <= _PICK_RADIUS_MM)
            if height == 0.0 or on_body:
                self.chess_arm_click_square(square)
                return

    def _click_ray_xiangqi(self, origin: np.ndarray, direction: np.ndarray, ground_hit: np.ndarray) -> None:
        ground_square = self.xiangqi_world.square_at(ground_hit[0], ground_hit[1])
        if self._selected is not None and ground_square in self._targets:
            self.chess_arm_click_square(ground_square)  # a lit crossing wins over a disc standing in front of it
            return
        t = (_XIANGQI_PICK_HEIGHT_MM - origin[2]) / direction[2]
        hit = origin + t * direction
        square = self.xiangqi_world.square_at(hit[0], hit[1])
        if square is not None:
            self.chess_arm_click_square(square)
        else:
            self._clear_selection()

    def chess_arm_click_square(self, square: int) -> None:
        """First click selects a piece of the side to move, a click on a lit square plays the move."""
        if self.state != "idle" or self._is_game_over():
            return
        if self._selected is not None and square in self._targets:
            move = self._targets[square]
            self._auto_blocked = False
            self._start_move(move, "You")
            return
        if square != self._selected and self._select_piece(square):
            self._refresh_highlights()
        else:
            self._clear_selection()

    def _select_piece(self, square: int) -> bool:
        """Select the piece of the side to move on this square and collect its legal targets."""
        targets: dict[int, chess.Move | XiangqiMove] = {}
        if self.chess_arm_is_chess():
            piece = self.board.piece_at(square)
            if piece is None or piece.color != self.board.turn:
                return False
            for move in self.board.legal_moves:
                if move.from_square == square and move.promotion in (None, chess.QUEEN):
                    targets[move.to_square] = move
        else:
            symbol = self.xiangqi_position.grid[square]
            if symbol is None or symbol.isupper() != self.xiangqi_position.red_to_move:
                return False
            for move in xiangqi_rules_legal_moves(self.xiangqi_position):
                if move.from_square == square:
                    targets[move.to_square] = move
        self._selected = square
        self._targets = targets
        return True

    def _clear_selection(self) -> None:
        self._selected = None
        self._targets = {}
        self._refresh_highlights()

    def _is_capture(self, move: chess.Move | XiangqiMove) -> bool:
        if self.chess_arm_is_chess():
            return self.board.is_capture(move)
        return self.xiangqi_position.grid[move.to_square] is not None

    def _refresh_highlights(self) -> None:
        highlights: dict[int, tuple] = {square: _HIGHLIGHT_RGB for square in self._last_move_squares}
        if self._selected is not None:
            highlights[self._selected] = _SELECTED_RGB
            for square, move in self._targets.items():
                highlights[square] = _CAPTURE_RGB if self._is_capture(move) else _SELECTED_RGB
        self._highlights = highlights

    def chess_arm_request_robot_move(self) -> None:
        """Ask Gemini for a move for the side to move, now."""
        if self.state != "idle":
            self._log("The arm is busy, wait for it to finish.")
            return
        if self._is_game_over():
            self._log("The game is over. Start a new game.")
            return
        self._auto_blocked = False
        self._start_thinking()

    # -------------------------------------------------------------- gemini
    def chess_arm_refresh_models(self, done: Callable[[list[GeminiModel], str], None]) -> None:
        """Fetch the model list in a worker thread; `done(models, error)` runs on the UI thread."""
        api_key = self._resolve_api_key()
        method = "bidiGenerateContent" if self.use_live else "generateContent"

        def work() -> None:
            try:
                models, error = gemini_client_list_models(api_key, method=method), ""
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
        api_key = self._resolve_api_key()
        live = self.use_live
        if self.chess_arm_is_chess():
            board_copy, history = self.board.copy(), list(self.san_history)
            side = "White" if board_copy.turn == chess.WHITE else "Black"
            build_prompt = lambda feedback: gemini_prompt_build(board_copy, history, feedback)  # noqa: E731
            parse_move = lambda text: gemini_prompt_parse_move(text, board_copy)  # noqa: E731
            system_prompt, follow_up = GEMINI_LIVE_SYSTEM_PROMPT, GEMINI_LIVE_FOLLOW_UP
        else:
            position_copy = XiangqiPosition(list(self.xiangqi_position.grid), self.xiangqi_position.red_to_move)
            history = list(self.xiangqi_history)
            side = "Red" if position_copy.red_to_move else "Black"
            build_prompt = lambda feedback: xiangqi_prompt_build(position_copy, history, feedback)  # noqa: E731
            parse_move = lambda text: xiangqi_prompt_parse_move(text, position_copy)  # noqa: E731
            system_prompt, follow_up = XIANGQI_LIVE_SYSTEM_PROMPT, XIANGQI_LIVE_FOLLOW_UP
        self._log(f"Gemini ({model_name}{', Live' if live else ''}) is thinking as {side} ...")

        def has_move(text: str) -> bool:
            try:
                parse_move(text)
                return True
            except ValueError:
                return False

        def ask(prompt: str) -> str:
            if live:
                return gemini_live_generate_text(
                    api_key, model_name, prompt, 120.0, system_instruction=system_prompt, accept=has_move,
                    follow_up=follow_up,
                    trace=lambda text: self._post_to_ui(lambda line=text: self._log(f"Gemini Live: {line}"))
                    if text.startswith("no usable") else None)
            return gemini_client_generate_text(
                api_key, model_name, prompt,
                on_retry=lambda text: self._post_to_ui(lambda line=text: self._log(f"Gemini: {line}")))

        def work() -> None:
            feedback = ""
            for attempt in range(1, _MAX_GEMINI_TRIES + 1):
                try:
                    answer = ask(build_prompt(feedback))
                except GeminiError as failure:
                    self._post_to_ui(lambda message=str(failure): self._on_gemini_failed(generation, message))
                    return
                try:
                    move = parse_move(answer)
                except ValueError as problem:
                    feedback = str(problem)
                    self._post_to_ui(lambda n=attempt, text=feedback: self._log(f"Gemini answer {n} refused: {text}"))
                    continue
                self._post_to_ui(lambda found=move: self._on_gemini_move(generation, found, model_name))
                return
            self._post_to_ui(lambda: self._on_gemini_failed(generation, f"no legal move after {_MAX_GEMINI_TRIES} tries"))

        threading.Thread(target=work, daemon=True).start()

    def _on_gemini_move(self, generation: int, move: chess.Move | XiangqiMove, model_name: str) -> None:
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
        if self.state != "idle" or self._auto_blocked or self._is_game_over():
            return
        turn_is_white = self._red_or_white_to_move()
        wanted = (self.robot_side == CHESS_ARM_ROBOT_SIDES[2]
                  or (self.robot_side == "White" and turn_is_white)
                  or (self.robot_side == "Black" and not turn_is_white))
        if wanted:
            self._start_thinking()

    # ---------------------------------------------------------------- moves
    def _start_move(self, move: chess.Move | XiangqiMove, who: str) -> bool:
        chess_game = self.chess_arm_is_chess()
        try:
            if chess_game:
                segments = chess_motion_plan(self.world, self.board, move, self.geometry, self.joints, self.gripper_mm)
            else:
                segments = xiangqi_motion_plan(self.xiangqi_world, move, self.geometry, self.joints, self.gripper_mm)
        except ChessMotionError as failure:
            self._log(f"Cannot play {move.uci() if chess_game else move.iccs()}: {failure}")
            return False
        if chess_game:
            san = self.board.san(move)
            number = self.board.fullmove_number
            prefix = f"{number}. " if self.board.turn == chess.WHITE else f"{number}... "
            self.board.push(move)
            self.san_history.append(san)
            text = f"{san}  ({move.uci()})"
        else:
            number = len(self.xiangqi_history) // 2 + 1
            prefix = f"{number}. " if self.xiangqi_position.red_to_move else f"{number}... "
            self.xiangqi_position = xiangqi_rules_play(self.xiangqi_position, move)
            self.xiangqi_history.append(move.iccs())
            text = move.iccs()
        self._last_move_squares = (move.from_square, move.to_square)
        self._clear_selection()
        self._log(f"{prefix}{text} by {who}")
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
        world = self._active_world()
        if event.kind == "pick":
            world.pick(event.piece_id)
        elif event.kind == "drop":
            world.drop(event.piece_id, event.x_mm, event.y_mm, event.square)
        elif event.kind == "promote" and isinstance(world, ChessWorld):
            world.promote(event.piece_id, event.new_kind)

    def _finish_move(self) -> None:
        self.state = "idle"
        self._segments = []
        self._status_note = ""
        if self._is_game_over():
            self._log(self._game_over_text())
            return
        if self._in_check():
            self._log("Check.")
        self._maybe_auto_play()

    def _in_check(self) -> bool:
        if self.chess_arm_is_chess():
            return self.board.is_check()
        return xiangqi_rules_in_check(self.xiangqi_position, self.xiangqi_position.red_to_move)

    def _game_over_text(self) -> str:
        if self.chess_arm_is_chess():
            outcome = self.board.outcome()
            reason = outcome.termination.name.replace("_", " ").lower() if outcome else "game over"
            return f"Game over: {self.board.result()} ({reason})."
        return f"Game over: {xiangqi_rules_winner(self.xiangqi_position)} wins (the other side has no legal move)."

    # ----------------------------------------------------------- read-outs
    def chess_arm_scene(self) -> WorldScene:
        """Meshes for the viewport."""
        if not self.chess_arm_is_chess():
            if self._xiangqi_glyphs is None:  # needs the Qt application, so it waits for the first xiangqi frame
                self._xiangqi_glyphs = {key: glyph_bitmap_render(character, _GLYPH_CELLS)
                                        for key, character in XIANGQI_PIECE_CHARS.items()}
            return xiangqi_scene_build(self.xiangqi_world, self.geometry, self.joints, self.gripper_mm,
                                       self._highlights, self._xiangqi_glyphs)
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
        if self._is_game_over():
            if self.chess_arm_is_chess():
                return f"Game over  {self.board.result()}"
            return f"Game over  {xiangqi_rules_winner(self.xiangqi_position)} wins"
        white_name = "White" if self.chess_arm_is_chess() else "Red"
        turn = white_name if self._red_or_white_to_move() else "Black"
        if self.state == "thinking":
            return f"{turn} to move  -  Gemini is thinking"
        if self.state == "moving":
            return f"Arm moving  -  {self._status_note}"
        return f"{turn} to move"
