"""Traditional xiangqi move names, in Vietnamese, as a player says them aloud (pure text).

'Pháo 2 bình 5' = the cannon on file 2 slides sideways to file 5. Files are counted 1-9 from the mover's own right
hand side. Straight movers (xe, pháo, tốt, tướng) give the number of steps after tiến (forward) / thoái (back); the
diagonal and L-shaped movers (mã, tượng, sĩ) always give the file they land on. Two equal pieces on one file are told
apart with 'trước' (the front one) and 'sau' (the back one).
"""
from __future__ import annotations

import re
import unicodedata

from core.xiangqi_board import XIANGQI_FILES
from core.xiangqi_rules import XiangqiMove, XiangqiPosition, xiangqi_rules_legal_moves

_NAMES_RED = {"K": "tướng", "A": "sĩ", "E": "tượng", "H": "mã", "R": "xe", "C": "pháo", "P": "binh"}
_NAMES_BLACK = {"K": "tướng", "A": "sĩ", "E": "tượng", "H": "mã", "R": "xe", "C": "pháo", "P": "tốt"}
_STRAIGHT_KINDS = ("K", "R", "C", "P")


def xiangqi_notation_piece_name(symbol: str) -> str:
    """Vietnamese name of a piece symbol ('H' -> 'mã')."""
    return (_NAMES_RED if symbol.isupper() else _NAMES_BLACK)[symbol.upper()]


def xiangqi_notation_vietnamese(grid: list[str | None], move: XiangqiMove) -> str:
    """The move as a player would announce it, judged on the position BEFORE the move, e.g. 'pháo 2 bình 5'."""
    symbol = grid[move.from_square]
    is_red = symbol.isupper()
    from_rank, from_file = divmod(move.from_square, XIANGQI_FILES)
    to_rank, to_file = divmod(move.to_square, XIANGQI_FILES)

    def file_number(file_index: int) -> int:
        return XIANGQI_FILES - file_index if is_red else file_index + 1

    forward = (to_rank - from_rank) if is_red else (from_rank - to_rank)
    action = "tiến" if forward > 0 else "thoái" if forward < 0 else "bình"
    target = abs(forward) if symbol.upper() in _STRAIGHT_KINDS and action != "bình" else file_number(to_file)
    twins = [square for square, other in enumerate(grid)
             if other == symbol and square != move.from_square and square % XIANGQI_FILES == from_file]
    if twins:
        twin_ahead = (twins[0] // XIANGQI_FILES > from_rank) if is_red else (twins[0] // XIANGQI_FILES < from_rank)
        where = "sau" if twin_ahead else "trước"
    else:
        where = str(file_number(from_file))
    return f"{xiangqi_notation_piece_name(symbol)} {where} {action} {target}"


_NUMBER_WORDS = {"một": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4", "năm": "5", "sáu": "6", "bảy": "7", "tám": "8",
                 "chín": "9"}


def _normalise(text: str, strip_accents: bool) -> str:
    lowered = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r"[^\W_]+", lowered)
    words = [_NUMBER_WORDS.get(word, word) for word in words if word not in ("số", "so")]
    joined = " ".join("binh" if word == "tốt" else word for word in words)
    if strip_accents:
        decomposed = unicodedata.normalize("NFD", joined.replace("đ", "d"))
        joined = "".join(char for char in decomposed if not unicodedata.combining(char))
    return joined


def xiangqi_notation_find_move(text: str, position: XiangqiPosition) -> XiangqiMove | None:
    """The legal move a player named in words ('pháo 2 bình 5', 'Pháo hai bình năm'), or None.

    Exact words are tried first. Without accents (speech to text loses them) the answer must be unique,
    because 'tướng' and 'tượng' only differ by an accent."""
    legal = xiangqi_rules_legal_moves(position)
    for strip_accents in (False, True):
        wanted = _normalise(text, strip_accents)
        found = [move for move in legal
                 if _normalise(xiangqi_notation_vietnamese(position.grid, move), strip_accents) == wanted]
        if len(found) == 1:
            return found[0]
    return None
