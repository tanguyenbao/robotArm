"""Physical xiangqi set on the table: where every piece stands, in millimetres, in the robot's world frame.

Same public methods as ChessWorld (pick, drop, piece_on, square_xy, ...), so the arm animation can use either.
A square is the index rank * 9 + file. WorldPiece.is_white means red here.
"""
from __future__ import annotations

from core.chess_world import WorldPiece
from core.xiangqi_board import XIANGQI_CELL_MM, XIANGQI_FILES, XIANGQI_RANKS, xiangqi_board_point_xy
from core.xiangqi_rules import XiangqiPosition

_GRAVE_FIRST_Y_MM = 210.0
_GRAVE_ROW_STEP_MM = 30.0
_GRAVE_FIRST_X_MM = 205.0
_GRAVE_COLUMN_STEP_MM = 32.0


class XiangqiWorld:
    """All pieces, who stands on which crossing, and the capture graveyards."""

    def __init__(self) -> None:
        self.pieces: dict[int, WorldPiece] = {}
        self.square_to_piece: dict[int, int] = {}
        self.grave_count = {True: 0, False: 0}
        self.flipped = False  # True: black stands nearest to the arm

    def reset(self, position: XiangqiPosition, flipped: bool) -> None:
        """Set the pieces up from a position."""
        self.pieces.clear()
        self.square_to_piece.clear()
        self.grave_count = {True: 0, False: 0}
        self.flipped = flipped
        piece_id = 0
        for square, symbol in enumerate(position.grid):
            if symbol is None:
                continue
            x_mm, y_mm = self.square_xy(square)
            self.pieces[piece_id] = WorldPiece(piece_id, symbol.upper(), symbol.isupper(), x_mm, y_mm, square)
            self.square_to_piece[square] = piece_id
            piece_id += 1

    def square_xy(self, square: int) -> tuple[float, float]:
        """Centre of a crossing."""
        return xiangqi_board_point_xy(square % XIANGQI_FILES, square // XIANGQI_FILES, self.flipped)

    def square_at(self, x_mm: float, y_mm: float) -> int | None:
        """Crossing nearest to a table point, or None when the point is off the board."""
        sign = -1.0 if self.flipped else 1.0
        center_x, center_y = xiangqi_board_point_xy((XIANGQI_FILES - 1) / 2, (XIANGQI_RANKS - 1) / 2)
        file_index = round((XIANGQI_FILES - 1) / 2 + sign * (x_mm - center_x) / XIANGQI_CELL_MM)
        rank_index = round((XIANGQI_RANKS - 1) / 2 + sign * (y_mm - center_y) / XIANGQI_CELL_MM)
        if 0 <= file_index < XIANGQI_FILES and 0 <= rank_index < XIANGQI_RANKS:
            return rank_index * XIANGQI_FILES + file_index
        return None

    def next_grave_xy(self, is_red: bool) -> tuple[float, float]:
        """Next free graveyard slot for a captured piece of this colour (red left, black right of the board)."""
        column, row = divmod(self.grave_count[is_red], 8)
        side = -1.0 if is_red else 1.0
        return side * (_GRAVE_FIRST_X_MM + _GRAVE_COLUMN_STEP_MM * column), _GRAVE_FIRST_Y_MM + _GRAVE_ROW_STEP_MM * row

    def piece_on(self, square: int) -> WorldPiece | None:
        """Piece standing on a crossing, if any."""
        piece_id = self.square_to_piece.get(square)
        return None if piece_id is None else self.pieces[piece_id]

    def pick(self, piece_id: int) -> None:
        """The gripper takes the piece off its crossing."""
        piece = self.pieces[piece_id]
        if piece.square is not None:
            self.square_to_piece.pop(piece.square, None)
        piece.square = None
        piece.held = True

    def drop(self, piece_id: int, x_mm: float, y_mm: float, square: int | None) -> None:
        """The gripper puts the piece down on a crossing (or in the graveyard when square is None)."""
        piece = self.pieces[piece_id]
        piece.held = False
        piece.x_mm, piece.y_mm, piece.square = x_mm, y_mm, square
        if square is not None:
            self.square_to_piece[square] = piece_id
        else:
            self.grave_count[piece.is_white] += 1
