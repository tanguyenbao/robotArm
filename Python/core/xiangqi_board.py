"""Xiangqi (Chinese chess) board geometry, piece sizes and the starting position, in millimetres, in the robot's world frame.

Pieces stand on the 9 x 10 line crossings. File 0..8 runs left to right, rank 0..9 runs from the red back rank to the
black back rank. Same frame as chess_world: arm base at the origin, +Y towards the board, board top at z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

XIANGQI_FILES = 9
XIANGQI_RANKS = 10
XIANGQI_CELL_MM = 36.0
XIANGQI_BOARD_CENTER_MM = (0.0, 330.0)
XIANGQI_DISC_HEIGHT_MM = 10.0

# kind letters (WXF notation): K general, A advisor, E elephant, H horse, R chariot, C cannon, P soldier
XIANGQI_PIECE_RADIUS_MM = {"K": 14.0, "A": 12.0, "E": 12.0, "H": 13.0, "R": 13.5, "C": 13.0, "P": 11.0}
XIANGQI_PIECE_GRIP_HEIGHT_MM = {kind: 12.0 for kind in XIANGQI_PIECE_RADIUS_MM}
XIANGQI_PIECE_GRIP_GAP_MM = {kind: 2.0 * radius for kind, radius in XIANGQI_PIECE_RADIUS_MM.items()}

# the name written on each piece: (kind, is_red) -> character
XIANGQI_PIECE_CHARS = {
    ("K", True): "帥", ("K", False): "將",
    ("A", True): "仕", ("A", False): "士",
    ("E", True): "相", ("E", False): "象",
    ("H", True): "傌", ("H", False): "馬",
    ("R", True): "俥", ("R", False): "車",
    ("C", True): "炮", ("C", False): "砲",
    ("P", True): "兵", ("P", False): "卒",
}

_BACK_RANK = ("R", "H", "E", "A", "K", "A", "E", "H", "R")
_CANNON_FILES = (1, 7)
_SOLDIER_FILES = (0, 2, 4, 6, 8)


@dataclass(frozen=True)
class XiangqiPiece:
    kind: str       # K A E H R C P
    is_red: bool
    file_index: int
    rank_index: int


def xiangqi_board_point_xy(file_index: int, rank_index: int, flipped: bool = False) -> tuple[float, float]:
    """Table position (x, y) of a line crossing. flipped turns the board 180 degrees (black nearest to the arm)."""
    sign = -1.0 if flipped else 1.0
    x_mm = XIANGQI_BOARD_CENTER_MM[0] + sign * (file_index - (XIANGQI_FILES - 1) / 2) * XIANGQI_CELL_MM
    y_mm = XIANGQI_BOARD_CENTER_MM[1] + sign * (rank_index - (XIANGQI_RANKS - 1) / 2) * XIANGQI_CELL_MM
    return x_mm, y_mm


def xiangqi_board_start_pieces() -> list[XiangqiPiece]:
    """All 32 pieces in the starting position."""
    pieces: list[XiangqiPiece] = []
    for is_red in (True, False):
        back, cannon, soldier = (0, 2, 3) if is_red else (9, 7, 6)
        pieces += [XiangqiPiece(kind, is_red, file_index, back) for file_index, kind in enumerate(_BACK_RANK)]
        pieces += [XiangqiPiece("C", is_red, file_index, cannon) for file_index in _CANNON_FILES]
        pieces += [XiangqiPiece("P", is_red, file_index, soldier) for file_index in _SOLDIER_FILES]
    return pieces
