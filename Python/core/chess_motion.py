"""Turns a chess move into arm motion: pick the piece, carry it, put it down (captures go to a graveyard first)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from core.arm_ik import arm_ik_solve
from core.arm_model import ArmGeometry
from core.arm_visual import ARM_GRIPPER_OPEN_MM
from core.chess_pieces import CHESS_PIECE_GRIP_GAP_MM, CHESS_PIECE_GRIP_HEIGHT_MM
from core.chess_world import ChessWorld
from core.world_scene import world_scene_tool_position

_FREE_HEIGHT_MM = 105.0       # tool height when travelling empty
_CARRY_CLEARANCE_MM = 85.0    # a carried piece stays this far above the board
_PATH_STEP_MM = 8.0
_TRAVEL_SPEED_MM_S = 380.0
_SLOW_SPEED_MM_S = 140.0
_GRIPPER_SECONDS = 0.4


class ChessMotionError(Exception):
    """The arm cannot reach a point of the planned path."""


@dataclass
class MotionEvent:
    kind: str                  # "pick", "drop" or "promote"
    piece_id: int
    x_mm: float = 0.0
    y_mm: float = 0.0
    square: int | None = None
    new_kind: str = ""


@dataclass
class MotionSegment:
    joint_samples: np.ndarray  # (n, 6) radians along the path
    duration_s: float
    gripper_start_mm: float
    gripper_end_mm: float
    label: str
    event: MotionEvent | None = None


class _Planner:
    def __init__(self, geometry: ArmGeometry, joints: np.ndarray, gripper_mm: float) -> None:
        self.geometry = geometry
        self.joints = np.asarray(joints, dtype=float)
        self.gripper_mm = gripper_mm
        self.segments: list[MotionSegment] = []

    def tool(self) -> np.ndarray:
        return world_scene_tool_position(self.geometry, self.joints)

    def move_to(self, target: tuple[float, float, float], speed: float, gripper_end_mm: float, label: str,
                event: MotionEvent | None = None) -> None:
        start = self.tool()
        end = np.array(target, dtype=float)
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(distance / _PATH_STEP_MM)) + 1)
        samples = np.empty((count, 6))
        seed = self.joints
        for index in range(count):
            point = start + (end - start) * (index / (count - 1))
            seed, converged = arm_ik_solve(self.geometry, point, seed)
            if not converged:
                raise ChessMotionError(f"{label}: the arm cannot reach ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.0f}) mm")
            samples[index] = seed
        duration = max(0.3, distance / speed * 1.5)
        self._add(samples, duration, gripper_end_mm, label, event)

    def grip(self, gripper_end_mm: float, label: str, event: MotionEvent | None = None) -> None:
        samples = np.vstack([self.joints, self.joints])
        self._add(samples, _GRIPPER_SECONDS, gripper_end_mm, label, event)

    def _add(self, samples: np.ndarray, duration: float, gripper_end_mm: float, label: str, event: MotionEvent | None) -> None:
        self.segments.append(MotionSegment(samples, duration, self.gripper_mm, gripper_end_mm, label, event))
        self.joints = samples[-1].copy()
        self.gripper_mm = gripper_end_mm

    def relocate(self, world: ChessWorld, piece_id: int, destination: tuple[float, float], square: int | None) -> None:
        piece = world.pieces[piece_id]
        grab_z = CHESS_PIECE_GRIP_HEIGHT_MM[piece.kind] - 7.0
        carry_z = grab_z + _CARRY_CLEARANCE_MM
        closed = CHESS_PIECE_GRIP_GAP_MM[piece.kind]
        source = (piece.x_mm, piece.y_mm)
        name = f"{piece.kind.upper() if piece.is_white else piece.kind}"
        self.move_to((*source, _FREE_HEIGHT_MM), _TRAVEL_SPEED_MM_S, ARM_GRIPPER_OPEN_MM, f"{name}: go over the piece")
        self.move_to((*source, grab_z), _SLOW_SPEED_MM_S, ARM_GRIPPER_OPEN_MM, f"{name}: lower")
        self.grip(closed, f"{name}: grip", MotionEvent("pick", piece_id))
        self.move_to((*source, carry_z), _SLOW_SPEED_MM_S, closed, f"{name}: lift")
        self.move_to((*destination, carry_z), _TRAVEL_SPEED_MM_S, closed, f"{name}: carry")
        self.move_to((*destination, grab_z), _SLOW_SPEED_MM_S, closed, f"{name}: set down")
        self.grip(ARM_GRIPPER_OPEN_MM, f"{name}: release", MotionEvent("drop", piece_id, destination[0], destination[1], square))
        self.move_to((*destination, _FREE_HEIGHT_MM), _SLOW_SPEED_MM_S, ARM_GRIPPER_OPEN_MM, f"{name}: retreat")


def chess_motion_plan(world: ChessWorld, board: chess.Board, move: chess.Move, geometry: ArmGeometry,
                      start_joints: np.ndarray, start_gripper_mm: float) -> list[MotionSegment]:
    """Plan the arm motion for a legal move on `board` (the board must not have played the move yet)."""
    planner = _Planner(geometry, start_joints, start_gripper_mm)
    mover = world.piece_on(move.from_square)
    if mover is None:
        raise ChessMotionError("no piece on the source square")

    if board.is_en_passant(move):
        victim_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    else:
        victim_square = move.to_square
    victim = world.piece_on(victim_square) if (board.is_capture(move)) else None
    if victim is not None:
        planner.relocate(world, victim.piece_id, world.next_grave_xy(victim.is_white), None)

    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        planner.relocate(world, mover.piece_id, world.square_xy(move.to_square), move.to_square)
        rook = world.piece_on(rook_from)
        if rook is not None:
            planner.relocate(world, rook.piece_id, world.square_xy(rook_to), rook_to)
        return planner.segments

    planner.relocate(world, mover.piece_id, world.square_xy(move.to_square), move.to_square)
    if move.promotion:
        symbol = chess.piece_symbol(move.promotion)
        planner.grip(ARM_GRIPPER_OPEN_MM, "promotion: swap the pawn", MotionEvent("promote", mover.piece_id, new_kind=symbol))
    return planner.segments
