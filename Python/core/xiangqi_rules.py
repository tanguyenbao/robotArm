"""Xiangqi rules in pure python: positions, legal moves, check and the end of the game.

A square is an index rank * 9 + file (rank 0 = red back rank, file 0 = column 'a'). A grid cell holds None or a piece
symbol: uppercase = red, lowercase = black, letters K A E H R C P. Moves are written in ICCS: 'h2e2' = from h2 to e2.
Not modelled: draw by repetition / perpetual check rules (the game simply goes on).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.xiangqi_board import XIANGQI_FILES, XIANGQI_RANKS, xiangqi_board_start_pieces

_ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_FILE_LETTERS = "abcdefghi"
_ICCS_PATTERN = re.compile(r"^([a-i])([0-9])\s*-?\s*([a-i])([0-9])$")


@dataclass(frozen=True)
class XiangqiMove:
    from_square: int
    to_square: int

    def iccs(self) -> str:
        """The move as text, for example 'h2e2'."""
        return xiangqi_rules_square_text(self.from_square) + xiangqi_rules_square_text(self.to_square)


@dataclass
class XiangqiPosition:
    grid: list[str | None]
    red_to_move: bool


def xiangqi_rules_start() -> XiangqiPosition:
    """The initial position, red to move."""
    grid: list[str | None] = [None] * (XIANGQI_FILES * XIANGQI_RANKS)
    for piece in xiangqi_board_start_pieces():
        grid[piece.rank_index * XIANGQI_FILES + piece.file_index] = piece.kind if piece.is_red else piece.kind.lower()
    return XiangqiPosition(grid, True)


def xiangqi_rules_square_text(square: int) -> str:
    """Square name such as 'e0'."""
    return f"{_FILE_LETTERS[square % XIANGQI_FILES]}{square // XIANGQI_FILES}"


def _on_board(file_index: int, rank_index: int) -> bool:
    return 0 <= file_index < XIANGQI_FILES and 0 <= rank_index < XIANGQI_RANKS


def _in_palace(file_index: int, rank_index: int, is_red: bool) -> bool:
    return 3 <= file_index <= 5 and (0 <= rank_index <= 2 if is_red else 7 <= rank_index <= 9)


def _pseudo_targets(grid: list[str | None], square: int) -> list[int]:
    """Squares the piece can reach by its movement rules, ignoring whether its own general ends up in danger."""
    symbol = grid[square]
    is_red = symbol.isupper()
    kind = symbol.upper()
    file_index, rank_index = square % XIANGQI_FILES, square // XIANGQI_FILES
    targets: list[int] = []

    def add_if_allowed(target_file: int, target_rank: int) -> None:
        if _on_board(target_file, target_rank):
            occupant = grid[target_rank * XIANGQI_FILES + target_file]
            if occupant is None or occupant.isupper() != is_red:
                targets.append(target_rank * XIANGQI_FILES + target_file)

    if kind == "K":
        for delta_file, delta_rank in _ORTHOGONAL:
            if _in_palace(file_index + delta_file, rank_index + delta_rank, is_red):
                add_if_allowed(file_index + delta_file, rank_index + delta_rank)
    elif kind == "A":
        for delta_file, delta_rank in _DIAGONAL:
            if _in_palace(file_index + delta_file, rank_index + delta_rank, is_red):
                add_if_allowed(file_index + delta_file, rank_index + delta_rank)
    elif kind == "E":
        for delta_file, delta_rank in _DIAGONAL:
            eye_file, eye_rank = file_index + delta_file, rank_index + delta_rank
            target_file, target_rank = file_index + 2 * delta_file, rank_index + 2 * delta_rank
            own_half = target_rank <= 4 if is_red else target_rank >= 5
            if _on_board(target_file, target_rank) and own_half and grid[eye_rank * XIANGQI_FILES + eye_file] is None:
                add_if_allowed(target_file, target_rank)
    elif kind == "H":
        for delta_file, delta_rank in _ORTHOGONAL:
            leg_file, leg_rank = file_index + delta_file, rank_index + delta_rank
            if not _on_board(leg_file, leg_rank) or grid[leg_rank * XIANGQI_FILES + leg_file] is not None:
                continue  # the horse is hobbled by a piece on its leg square
            for side_file, side_rank in ((delta_rank, delta_file), (-delta_rank, -delta_file)):
                add_if_allowed(file_index + 2 * delta_file + side_file, rank_index + 2 * delta_rank + side_rank)
    elif kind == "R":
        for delta_file, delta_rank in _ORTHOGONAL:
            step_file, step_rank = file_index + delta_file, rank_index + delta_rank
            while _on_board(step_file, step_rank):
                add_if_allowed(step_file, step_rank)
                if grid[step_rank * XIANGQI_FILES + step_file] is not None:
                    break
                step_file, step_rank = step_file + delta_file, step_rank + delta_rank
    elif kind == "C":
        for delta_file, delta_rank in _ORTHOGONAL:
            step_file, step_rank = file_index + delta_file, rank_index + delta_rank
            screen_found = False
            while _on_board(step_file, step_rank):
                occupant = grid[step_rank * XIANGQI_FILES + step_file]
                if not screen_found:
                    if occupant is None:
                        targets.append(step_rank * XIANGQI_FILES + step_file)
                    else:
                        screen_found = True
                elif occupant is not None:
                    if occupant.isupper() != is_red:
                        targets.append(step_rank * XIANGQI_FILES + step_file)
                    break
                step_file, step_rank = step_file + delta_file, step_rank + delta_rank
    elif kind == "P":
        forward = 1 if is_red else -1
        add_if_allowed(file_index, rank_index + forward)
        if (rank_index >= 5) if is_red else (rank_index <= 4):  # across the river a soldier also steps sideways
            add_if_allowed(file_index - 1, rank_index)
            add_if_allowed(file_index + 1, rank_index)
    return targets


def xiangqi_rules_in_check(position: XiangqiPosition, is_red: bool) -> bool:
    """True when the general of the given colour is attacked (including the 'flying general' face-off)."""
    return _grid_in_check(position.grid, is_red)


def _grid_in_check(grid: list[str | None], is_red: bool) -> bool:
    own_general, enemy_general = ("K", "k") if is_red else ("k", "K")
    if own_general not in grid:
        return True
    general_square = grid.index(own_general)
    if enemy_general in grid:
        enemy_square = grid.index(enemy_general)
        if enemy_square % XIANGQI_FILES == general_square % XIANGQI_FILES:
            low, high = sorted((general_square, enemy_square))
            if all(grid[between] is None for between in range(low + XIANGQI_FILES, high, XIANGQI_FILES)):
                return True
    for square, symbol in enumerate(grid):
        if symbol is not None and symbol.isupper() != is_red and general_square in _pseudo_targets(grid, square):
            return True
    return False


def xiangqi_rules_legal_moves(position: XiangqiPosition) -> list[XiangqiMove]:
    """Every legal move of the side to move."""
    moves: list[XiangqiMove] = []
    for square, symbol in enumerate(position.grid):
        if symbol is None or symbol.isupper() != position.red_to_move:
            continue
        for target in _pseudo_targets(position.grid, square):
            trial = list(position.grid)
            trial[target], trial[square] = symbol, None
            if not _grid_in_check(trial, position.red_to_move):
                moves.append(XiangqiMove(square, target))
    return moves


def xiangqi_rules_play(position: XiangqiPosition, move: XiangqiMove) -> XiangqiPosition:
    """The position after a legal move (the original is left untouched)."""
    grid = list(position.grid)
    grid[move.to_square], grid[move.from_square] = grid[move.from_square], None
    return XiangqiPosition(grid, not position.red_to_move)


def xiangqi_rules_winner(position: XiangqiPosition) -> str | None:
    """'Red' or 'Black' when the side to move has no legal move (checkmate or stalemate both lose), else None."""
    if xiangqi_rules_legal_moves(position):
        return None
    return "Black" if position.red_to_move else "Red"


def xiangqi_rules_move_from_text(text: str, position: XiangqiPosition) -> XiangqiMove | None:
    """Parse an ICCS move ('h2e2' or 'h2-e2'). None when the text is not a legal move here."""
    match = _ICCS_PATTERN.match(text.strip().lower())
    if match is None:
        return None
    from_square = int(match.group(2)) * XIANGQI_FILES + _FILE_LETTERS.index(match.group(1))
    to_square = int(match.group(4)) * XIANGQI_FILES + _FILE_LETTERS.index(match.group(3))
    candidate = XiangqiMove(from_square, to_square)
    return candidate if candidate in xiangqi_rules_legal_moves(position) else None


def xiangqi_rules_ascii(position: XiangqiPosition) -> str:
    """Text diagram, black on top. Red = uppercase, black = lowercase, '.' = empty."""
    lines = []
    for rank_index in range(XIANGQI_RANKS - 1, -1, -1):
        cells = " ".join(position.grid[rank_index * XIANGQI_FILES + file_index] or "." for file_index in range(XIANGQI_FILES))
        lines.append(f"{rank_index}  {cells}")
    lines.append("   " + " ".join(_FILE_LETTERS))
    return "\n".join(lines)
