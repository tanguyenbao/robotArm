"""Physical chess set on the table: where every piece stands, in millimetres, in the robot's world frame.

The arm base is at the origin, +Y points at the board, Z is up, the board top is z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

CHESS_SQUARE_MM = 40.0
CHESS_BOARD_CENTER_MM = (0.0, 330.0)
_GRAVE_ROW_STEP_MM = 34.0
_GRAVE_FIRST_Y_MM = 210.0


@dataclass
class WorldPiece:
    piece_id: int
    kind: str          # p n b r q k
    is_white: bool
    x_mm: float
    y_mm: float
    square: int | None  # None while held or in the graveyard
    held: bool = False


class ChessWorld:
    """All pieces, who stands on which square, and the capture graveyards."""

    def __init__(self) -> None:
        self.pieces: dict[int, WorldPiece] = {}
        self.square_to_piece: dict[int, int] = {}
        self.grave_count = {True: 0, False: 0}
        self.black_near_robot = True

    def reset(self, board: chess.Board, black_near_robot: bool) -> None:
        """Set the pieces up from a python-chess board. black_near_robot rotates the board 180 degrees."""
        self.pieces.clear()
        self.square_to_piece.clear()
        self.grave_count = {True: 0, False: 0}
        self.black_near_robot = black_near_robot
        for index, (square, piece) in enumerate(sorted(board.piece_map().items())):
            x_mm, y_mm = self.square_xy(square)
            self.pieces[index] = WorldPiece(index, piece.symbol().lower(), piece.color == chess.WHITE, x_mm, y_mm, square)
            self.square_to_piece[square] = index

    def square_xy(self, square: int) -> tuple[float, float]:
        """Centre of a square."""
        file_index, rank_index = chess.square_file(square), chess.square_rank(square)
        sign = -1.0 if self.black_near_robot else 1.0
        x_mm = CHESS_BOARD_CENTER_MM[0] + sign * (file_index - 3.5) * CHESS_SQUARE_MM
        y_mm = CHESS_BOARD_CENTER_MM[1] + sign * (rank_index - 3.5) * CHESS_SQUARE_MM
        return x_mm, y_mm

    def next_grave_xy(self, is_white: bool) -> tuple[float, float]:
        """Next free graveyard slot for a captured piece of this colour (white left, black right of the board)."""
        index = self.grave_count[is_white]
        column, row = divmod(index, 8)
        side = -1.0 if is_white else 1.0
        return side * (235.0 + 40.0 * column), _GRAVE_FIRST_Y_MM + _GRAVE_ROW_STEP_MM * row

    def piece_on(self, square: int) -> WorldPiece | None:
        """Piece standing on a square, if any."""
        piece_id = self.square_to_piece.get(square)
        return None if piece_id is None else self.pieces[piece_id]

    def pick(self, piece_id: int) -> None:
        """The gripper takes the piece off its square."""
        piece = self.pieces[piece_id]
        if piece.square is not None:
            self.square_to_piece.pop(piece.square, None)
        piece.square = None
        piece.held = True

    def drop(self, piece_id: int, x_mm: float, y_mm: float, square: int | None) -> None:
        """The gripper puts the piece down on a square (or in the graveyard when square is None)."""
        piece = self.pieces[piece_id]
        piece.held = False
        piece.x_mm, piece.y_mm, piece.square = x_mm, y_mm, square
        if square is not None:
            self.square_to_piece[square] = piece_id
        else:
            self.grave_count[piece.is_white] += 1

    def promote(self, piece_id: int, kind: str) -> None:
        """Swap a pawn for a new piece."""
        self.pieces[piece_id].kind = kind
