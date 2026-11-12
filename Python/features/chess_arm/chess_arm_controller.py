"""Feature: the robot arm plays chess or xiangqi with a built-in engine, while Gemini Live chats as the companion. No Qt in here.

The UI gives two callbacks: `post_to_ui` (runs a function on the UI thread, used to bring worker-thread results back)
and `log` (appends a line to the log panel).

Squares are integers in both games (chess: python-chess square, xiangqi: rank * 9 + file), so selection, highlights and
the arm animation are shared. "White" in the robot-side list means Red when xiangqi is on the table.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

import chess
import numpy as np

from adapters.gemini_client import GeminiError, GeminiModel, gemini_client_list_models
from adapters.glyph_bitmap import glyph_bitmap_render
from core.arm_model import ARM_JOINT_NAMES, ArmGeometry, arm_model_home_joints
from core.arm_visual import ARM_GRIPPER_OPEN_MM
from core.chess_motion import ChessMotionError, MotionEvent, MotionSegment, chess_motion_plan, xiangqi_motion_plan
from core.chess_pieces import CHESS_PIECE_HEIGHT_MM
from core.chess_world import ChessWorld
from core.chess_search import CHESS_MATE_SCORE, chess_search_choose
from core.companion_prompt import CompanionTurn, companion_prompt_assessment, companion_prompt_turn
from core.gemini_prompt import chess_move_from_text
from core.world_scene import WorldScene, world_scene_build, world_scene_tool_position
from core.xiangqi_board import XIANGQI_DISC_HEIGHT_MM, XIANGQI_PIECE_CHARS
from core.xiangqi_notation import xiangqi_notation_find_move, xiangqi_notation_piece_name, xiangqi_notation_vietnamese
from core.xiangqi_rules import (XiangqiMove, XiangqiPosition, xiangqi_rules_ascii, xiangqi_rules_in_check,
                                xiangqi_rules_legal_moves, xiangqi_rules_move_from_text, xiangqi_rules_play,
                                xiangqi_rules_start, xiangqi_rules_winner)
from core.xiangqi_scene import xiangqi_scene_build
from core.xiangqi_search import XIANGQI_MATE_SCORE, xiangqi_search_choose
from core.xiangqi_world import XiangqiWorld
from features.chess_arm.companion import ChessCompanion

CHESS_ARM_GAME_TYPES = ("Chess", "Xiangqi (Cờ tướng)")
CHESS_ARM_ROBOT_SIDES = ("Black", "White", "Both (engine vs engine)", "None (manual)")
_AUTO_WAIT_S = 25.0   # engine vs engine: longest the robot waits for the companion to finish talking
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
    """Game state, arm animation, the engine that plays the robot side and the Gemini companion."""

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
        self.model_name = ""
        self.companion = ChessCompanion(post_to_ui, log, self._companion_session_text)
        self.companion.tool_handler = self._companion_tool
        self.state_listener: Callable[[], None] = lambda: None   # the UI re-reads game type and robot side from here
        self._last_user_move_text = ""         # the person's last move in words, told to the companion with the robot's reply
        self._engine_score = 0                 # engine score of the robot's latest move, centipawns for the robot
        self._auto_wait_deadline: float | None = None
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
        self._generation = 0                   # bumped on a new game so stale engine answers are ignored
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
        self._last_user_move_text = ""
        self._auto_wait_deadline = None
        self._log(f"New {'chess' if self.chess_arm_is_chess() else 'xiangqi'} game. {first_mover} moves first.")
        self._maybe_auto_play()
        if self.state == "idle":  # when the robot opens, its first move is announced instead
            self.companion.companion_notify(companion_prompt_turn(self._turn_for_companion("new_game")))

    def chess_arm_set_robot_side(self, side: str) -> None:
        """Choose which colour(s) the robot engine plays ('Red' is the xiangqi name of 'White')."""
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
            move = (xiangqi_rules_move_from_text(text, self.xiangqi_position)
                    or xiangqi_notation_find_move(text, self.xiangqi_position))
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
        """Let the engine play the side to move, now."""
        if self.state != "idle":
            self._log("The arm is busy, wait for it to finish.")
            return
        if self._is_game_over():
            self._log("The game is over. Start a new game.")
            return
        self._auto_blocked = False
        self._start_thinking()

    # ------------------------------------------------------- gemini companion
    def chess_arm_refresh_models(self, done: Callable[[list[GeminiModel], str], None]) -> None:
        """Fetch the Live model list in a worker thread; `done(models, error)` runs on the UI thread."""
        api_key = self._resolve_api_key()

        def work() -> None:
            try:
                models, error = gemini_client_list_models(api_key, method="bidiGenerateContent"), ""
            except GeminiError as failure:
                models, error = [], str(failure)
            self._post_to_ui(lambda: done(models, error))

        threading.Thread(target=work, daemon=True).start()

    def _resolve_api_key(self) -> str:
        return self.api_key.strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    def chess_arm_companion_toggle(self) -> None:
        """Connect the Gemini Live companion, or disconnect it when it is already connected."""
        if self.companion.companion_is_active():
            self.companion.companion_disconnect()
        else:
            self.chess_arm_companion_ensure_connected()

    def chess_arm_companion_ensure_connected(self) -> bool:
        return self.companion.companion_connect(self._resolve_api_key(), self.model_name.strip())

    def chess_arm_companion_say(self, text: str) -> None:
        """The person typed something for the companion (connects first when needed)."""
        text = text.strip()
        if text and self.chess_arm_companion_ensure_connected():
            self.companion.companion_say(text)

    def chess_arm_companion_audio(self, pcm16_16k: bytes) -> None:
        """Microphone data for the companion."""
        self.companion.companion_send_audio(pcm16_16k)

    def _companion_tool(self, name: str, args: dict) -> str:
        """Run a tool the companion called; the answer text goes back to the model."""
        if name == "set_game":
            wanted = {"chess": CHESS_ARM_GAME_TYPES[0], "xiangqi": CHESS_ARM_GAME_TYPES[1]}.get(str(args.get("game", "")).lower())
            if wanted is None:
                return "error: game must be chess or xiangqi"
            if wanted == self.game_type:
                self.chess_arm_new_game()
            else:
                self.chess_arm_set_game_type(wanted)
            result = f"a new {args['game']} game is on the table"
        elif name == "play_move":
            return self._companion_play_move(str(args.get("move", "")))
        elif name == "new_game":
            self.chess_arm_new_game()
            result = "the game was reset to the starting position"
        elif name == "set_robot_side":
            sides = {"white": "White", "black": "Black", "both": CHESS_ARM_ROBOT_SIDES[2], "none": CHESS_ARM_ROBOT_SIDES[3]}
            side = sides.get(str(args.get("side", "")).lower())
            if side is None:
                return "error: side must be white, black, both or none"
            self.chess_arm_set_robot_side(side)
            result = f"the robot now plays {self._robot_side_name()}"
        else:
            return f"error: unknown tool {name}"
        self.state_listener()
        return result

    def _companion_play_move(self, text: str) -> str:
        """The person's move, relayed by the companion. The reply says what happened, or why not."""
        if self._is_game_over():
            return "error: the game is over"
        if self.state != "idle" or self._robot_plays_turn():
            return "error: it is not the person's turn (the arm is moving or the robot is to play)"
        if self.chess_arm_submit_move_text(text):
            return ("the person's move is being played on the board. The robot answers by itself in a moment and you "
                    "will get a [GAME EVENT] for it: do not comment yet")
        return f"error: '{text}' is not one exact legal move here. Legal moves: {'; '.join(self._legal_move_words())}"

    def _robot_side_name(self) -> str:
        if self.robot_side in ("White", "Black"):
            return "Red" if (self.robot_side == "White" and not self.chess_arm_is_chess()) else self.robot_side
        return "both sides (engine against itself)" if self.robot_side == CHESS_ARM_ROBOT_SIDES[2] else "none"

    def _turn_for_companion(self, kind: str) -> CompanionTurn:
        """A turn description filled with the current game state."""
        if self.chess_arm_is_chess():
            board_text, history, name = str(self.board), list(self.san_history), "chess"
        else:
            board_text, history, name = (xiangqi_rules_ascii(self.xiangqi_position), list(self.xiangqi_history), "xiangqi")
        result = self._game_over_text() if self._is_game_over() else ""
        person_moves = [] if result or self._robot_plays_turn() else self._legal_move_words()
        return CompanionTurn(kind, name, self._robot_side_name(), board_text=board_text, history=history, result=result,
                             person_moves=person_moves)

    def _legal_move_words(self) -> list[str]:
        """Every legal move of the side to move, in words (chess: SAN, xiangqi: Vietnamese notation, captures marked)."""
        if self.chess_arm_is_chess():
            return [self.board.san(move) for move in self.board.legal_moves]
        grid = self.xiangqi_position.grid
        words = []
        for move in xiangqi_rules_legal_moves(self.xiangqi_position):
            name, victim = xiangqi_notation_vietnamese(grid, move), grid[move.to_square]
            words.append(f"{name} (ăn {xiangqi_notation_piece_name(victim)})" if victim else name)
        return words

    def _companion_session_text(self) -> str:
        """First message of a new Live session: the position as it is now."""
        turn = self._turn_for_companion("session_start")
        white_name = "White" if self.chess_arm_is_chess() else "Red"
        turn.facts.append(f"{white_name if self._red_or_white_to_move() else 'Black'} is to move")
        return companion_prompt_turn(turn)

    def _describe_move(self, move: chess.Move | XiangqiMove) -> tuple[str, list[str]]:
        """(the move in words, facts about it) for a move on the CURRENT position, before it is pushed."""
        facts: list[str] = []
        if self.chess_arm_is_chess():
            piece = self.board.piece_at(move.from_square)
            spoken = (f"{chess.piece_name(piece.piece_type)} from {chess.square_name(move.from_square)} to "
                      f"{chess.square_name(move.to_square)} (written {self.board.san(move)})")
            if self.board.is_capture(move):
                victim = self.board.piece_at(move.to_square)
                facts.append(f"it captures a {chess.piece_name(victim.piece_type) if victim else 'pawn'}")
            if self.board.is_castling(move):
                facts.append("it is castling")
            if move.promotion:
                facts.append("the pawn promotes to a queen")
        else:
            grid = self.xiangqi_position.grid
            spoken = f"{xiangqi_notation_vietnamese(grid, move)} (ICCS {move.iccs()})"
            victim = grid[move.to_square]
            if victim is not None:
                facts.append(f"it captures a {xiangqi_notation_piece_name(victim)}")
        return spoken, facts

    def _notify_companion_move(self, spoken: str, facts: list[str], robot: bool) -> None:
        """Tell the companion about the move that was just pushed (the position is already the one after it)."""
        if not self.companion.companion_is_active():
            return
        if self._in_check():
            facts = facts + ["it gives check"]
        if robot:
            turn = self._turn_for_companion("robot_move")
            turn.user_move, self._last_user_move_text = self._last_user_move_text, ""
            turn.robot_move, turn.facts = spoken, facts
            mate_score = CHESS_MATE_SCORE if self.chess_arm_is_chess() else XIANGQI_MATE_SCORE
            turn.assessment = companion_prompt_assessment(self._engine_score, mate_score)
            self.companion.companion_notify(companion_prompt_turn(turn))
            return
        text = spoken + (f" ({'; '.join(facts)})" if facts else "")
        if self._robot_plays_turn():
            self._last_user_move_text = text  # told together with the robot's answer, so the model speaks once
            return
        turn = self._turn_for_companion("user_move")
        turn.user_move = text
        self.companion.companion_notify(companion_prompt_turn(turn))

    def _robot_plays_turn(self) -> bool:
        """True when the engine is the one to move now."""
        if self._is_game_over():
            return False
        turn_is_white = self._red_or_white_to_move()
        return (self.robot_side == CHESS_ARM_ROBOT_SIDES[2]
                or (self.robot_side == "White" and turn_is_white)
                or (self.robot_side == "Black" and not turn_is_white))

    # --------------------------------------------------------------- engine
    def _start_thinking(self) -> None:
        self.state = "thinking"
        generation = self._generation
        if self.chess_arm_is_chess():
            board_copy = self.board.copy()
            search = lambda: chess_search_choose(board_copy)  # noqa: E731
        else:
            position_copy = XiangqiPosition(list(self.xiangqi_position.grid), self.xiangqi_position.red_to_move)
            search = lambda: xiangqi_search_choose(position_copy)  # noqa: E731

        def work() -> None:
            choice = search()
            self._post_to_ui(lambda: self._on_engine_move(generation, choice))

        threading.Thread(target=work, daemon=True).start()

    def _on_engine_move(self, generation: int, choice: tuple[chess.Move | XiangqiMove, int] | None) -> None:
        if generation != self._generation or self.state != "thinking":
            return
        self.state = "idle"
        if choice is None:
            self._auto_blocked = True
            self._log("The engine has no legal move.")
            return
        move, self._engine_score = choice
        if not self._start_move(move, "Robot", robot=True):
            self._auto_blocked = True

    def _maybe_auto_play(self) -> None:
        if self.state != "idle" or self._auto_blocked or not self._robot_plays_turn():
            return
        if (self.robot_side == CHESS_ARM_ROBOT_SIDES[2] and self.companion.companion_is_busy()
                and (self._auto_wait_deadline is None or time.monotonic() < self._auto_wait_deadline)):
            if self._auto_wait_deadline is None:  # engine against itself: let the companion finish talking first
                self._auto_wait_deadline = time.monotonic() + _AUTO_WAIT_S
            return
        self._auto_wait_deadline = None
        self._start_thinking()

    # ---------------------------------------------------------------- moves
    def _start_move(self, move: chess.Move | XiangqiMove, who: str, robot: bool = False) -> bool:
        chess_game = self.chess_arm_is_chess()
        try:
            if chess_game:
                segments = chess_motion_plan(self.world, self.board, move, self.geometry, self.joints, self.gripper_mm)
            else:
                segments = xiangqi_motion_plan(self.xiangqi_world, move, self.geometry, self.joints, self.gripper_mm)
        except ChessMotionError as failure:
            self._log(f"Cannot play {move.uci() if chess_game else move.iccs()}: {failure}")
            return False
        spoken, facts = self._describe_move(move)
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
        self._notify_companion_move(spoken, facts, robot)
        return True

    def chess_arm_tick(self, seconds: float) -> bool:
        """Advance the animation. Returns True when the picture changed."""
        if self._auto_wait_deadline is not None and self.state == "idle":
            self._maybe_auto_play()  # engine against itself, waiting for the companion to finish talking
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
            return f"{turn} to move  -  the robot is thinking"
        if self.state == "moving":
            return f"Arm moving  -  {self._status_note}"
        return f"{turn} to move"
