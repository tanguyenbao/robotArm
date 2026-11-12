"""Small xiangqi search engine (pure python): material alpha-beta, used to take blunders out of Gemini's choices.

Gemini reads a board diagram badly and gives pieces away, so the controller shows it only the moves this search rates
within `margin` of the best one. Search nodes use pseudo-legal moves; a captured general is a huge material swing and
generals facing each other count as a win for the side to move, so mates inside the horizon are found.
"""
from __future__ import annotations

from dataclasses import dataclass

import random

from core.xiangqi_board import XIANGQI_FILES, XIANGQI_RANKS
from core.xiangqi_rules import (XiangqiMove, XiangqiPosition, xiangqi_rules_in_check, xiangqi_rules_legal_moves,
                                xiangqi_rules_play, xiangqi_rules_pseudo_moves)

XIANGQI_SEARCH_DEPTH = 3
XIANGQI_SEARCH_MARGIN = 150     # centipawns: moves rated within this of the best stay in the list
XIANGQI_SEARCH_LIMIT = 12
XIANGQI_SEARCH_TOLERANCE = 25   # the engine plays any move rated within this of the best one, so games differ
XIANGQI_MATE_SCORE = 50000      # scores from this up mean a forced win inside the search horizon
_QUIET_DEPTH = 2
_MATE = XIANGQI_MATE_SCORE
_INFINITY = 10 ** 9
_VALUE = {"K": 10000, "R": 900, "C": 450, "H": 400, "E": 200, "A": 200, "P": 100}
_KIND_NAMES = {"K": "general", "R": "chariot", "C": "cannon", "H": "horse", "E": "elephant", "A": "advisor", "P": "soldier"}


@dataclass(frozen=True)
class XiangqiCandidate:
    move: XiangqiMove
    score: int      # centipawns for the side to move, after the search
    note: str       # short remark such as 'captures chariot, check'


def _evaluate(grid: list[str | None], red: bool) -> int:
    """Material (a soldier across the river is worth double) from the point of view of `red`."""
    total = 0
    for square, symbol in enumerate(grid):
        if symbol is None:
            continue
        kind = symbol.upper()
        value = _VALUE[kind]
        if kind == "P" and ((square >= 5 * XIANGQI_FILES) if symbol.isupper() else (square < 5 * XIANGQI_FILES)):
            value += 100
        if kind in ("R", "H", "C"):  # a few centipawns for activity, so equal-material moves are not all alike
            rank_index, file_index = divmod(square, XIANGQI_FILES)
            progress = rank_index if symbol.isupper() else XIANGQI_RANKS - 1 - rank_index
            value += 3 * progress + (3 * (4 - abs(file_index - 4)) if kind != "R" else 0)
        total += value if symbol.isupper() == red else -value
    return total


def _generals_face(grid: list[str | None]) -> bool:
    if "K" not in grid or "k" not in grid:
        return False
    red_square, black_square = grid.index("K"), grid.index("k")
    if red_square % XIANGQI_FILES != black_square % XIANGQI_FILES:
        return False
    low, high = sorted((red_square, black_square))
    return all(grid[between] is None for between in range(low + XIANGQI_FILES, high, XIANGQI_FILES))


def _ordered_moves(grid: list[str | None], red: bool, captures_only: bool = False) -> list[tuple[int, int]]:
    moves = xiangqi_rules_pseudo_moves(grid, red)
    if captures_only:
        moves = [move for move in moves if grid[move[1]] is not None]
    moves.sort(key=lambda move: -_VALUE[grid[move[1]].upper()] if grid[move[1]] is not None else 0)
    return moves


def _quiesce(grid: list[str | None], red: bool, alpha: int, beta: int, depth: int) -> int:
    stand_pat = _evaluate(grid, red)
    if depth == 0 or stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    for source, target in _ordered_moves(grid, red, captures_only=True):
        captured = grid[target]
        grid[target], grid[source] = grid[source], None
        score = -_quiesce(grid, not red, -beta, -alpha, depth - 1)
        grid[source], grid[target] = grid[target], captured
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def _negamax(grid: list[str | None], red: bool, depth: int, alpha: int, beta: int) -> int:
    if ("K" if red else "k") not in grid:
        return -(_MATE + depth)
    if _generals_face(grid):
        return _MATE + depth  # the side to move simply takes the opposing general
    if depth == 0:
        return _quiesce(grid, red, alpha, beta, _QUIET_DEPTH)
    moves = _ordered_moves(grid, red)
    if not moves:
        return -_MATE
    best = -_INFINITY
    for source, target in moves:
        captured = grid[target]
        grid[target], grid[source] = grid[source], None
        score = -_negamax(grid, not red, depth - 1, -beta, -alpha)
        grid[source], grid[target] = grid[target], captured
        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def _note(position: XiangqiPosition, move: XiangqiMove) -> str:
    notes = []
    victim = position.grid[move.to_square]
    if victim is not None:
        notes.append(f"captures {_KIND_NAMES[victim.upper()]}")
    after = xiangqi_rules_play(position, move)
    if xiangqi_rules_in_check(after, after.red_to_move):
        notes.append("check")
    return ", ".join(notes)


def xiangqi_search_candidates(position: XiangqiPosition, depth: int = XIANGQI_SEARCH_DEPTH,
                              margin: int = XIANGQI_SEARCH_MARGIN, limit: int = XIANGQI_SEARCH_LIMIT) -> list[XiangqiCandidate]:
    """Legal moves rated within `margin` of the best one, best first (at most `limit`). Empty when there is no legal move."""
    legal = xiangqi_rules_legal_moves(position)
    grid = list(position.grid)
    red = position.red_to_move
    legal.sort(key=lambda move: -_VALUE[grid[move.to_square].upper()] if grid[move.to_square] is not None else 0)
    scored: list[tuple[XiangqiMove, int]] = []
    best = -_INFINITY
    for move in legal:
        captured = grid[move.to_square]
        grid[move.to_square], grid[move.from_square] = grid[move.from_square], None
        # moves already worse than best - margin are cut off early: the score returned for them is only a bound
        score = -_negamax(grid, not red, depth - 1, -_INFINITY, -(best - margin) if best > -_INFINITY else _INFINITY)
        grid[move.from_square], grid[move.to_square] = grid[move.to_square], captured
        scored.append((move, score))
        best = max(best, score)
    kept = sorted((item for item in scored if item[1] > best - margin), key=lambda item: -item[1])[:limit]
    return [XiangqiCandidate(move, score, _note(position, move)) for move, score in kept]


def xiangqi_search_choose(position: XiangqiPosition, depth: int = XIANGQI_SEARCH_DEPTH,
                          tolerance: int = XIANGQI_SEARCH_TOLERANCE) -> tuple[XiangqiMove, int] | None:
    """The engine's move (one of the best, picked at random among those within `tolerance`) and its score in centipawns
    for the side to move. None when there is no legal move."""
    candidates = xiangqi_search_candidates(position, depth, tolerance, XIANGQI_SEARCH_LIMIT)
    if not candidates:
        return None
    chosen = random.choice([item for item in candidates if item.score >= candidates[0].score - tolerance])
    return chosen.move, chosen.score
