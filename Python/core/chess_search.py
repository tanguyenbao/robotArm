"""Small chess engine (pure python on python-chess): iterative-deepening alpha-beta with quiescence and a time limit.

The robot arm plays with this engine; Gemini only talks about the game. Evaluation is material plus a little
positional sense (centre, advanced pawns, a quiet king), so games are playable but not strong.
"""
from __future__ import annotations

import random
import time

import chess

CHESS_SEARCH_SECONDS = 3.0
CHESS_SEARCH_MAX_DEPTH = 4
CHESS_SEARCH_TOLERANCE = 12     # centipawns: the engine plays any move rated within this of the best, so games differ
CHESS_MATE_SCORE = 100000       # scores from this minus a few plies up mean a forced mate
_QUIET_DEPTH = 4
_INFINITY = 10 ** 9
_PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


class _OutOfTime(Exception):
    """Raised inside the search when the time limit is over."""


def chess_search_material(board: chess.Board, white_side: bool) -> int:
    """Plain material balance in centipawns from the point of view of one colour."""
    total = 0
    for piece in board.piece_map().values():
        value = _PIECE_VALUE[piece.piece_type]
        total += value if (piece.color == chess.WHITE) == white_side else -value
    return total


def _evaluate(board: chess.Board) -> int:
    """Score for the side to move."""
    total = 0
    for square, piece in board.piece_map().items():
        file_index, rank_index = chess.square_file(square), chess.square_rank(square)
        advance = rank_index if piece.color == chess.WHITE else 7 - rank_index
        centrality = 3.5 - max(abs(file_index - 3.5), abs(rank_index - 3.5))  # 0 on the rim .. 3 in the middle
        value = _PIECE_VALUE[piece.piece_type]
        if piece.piece_type == chess.PAWN:
            value += advance * (1 + centrality)  # centre pawns first, edge pawns last
        elif piece.piece_type == chess.KNIGHT:
            value += 10 * centrality
        elif piece.piece_type == chess.BISHOP:
            value += 5 * centrality
        elif piece.piece_type == chess.QUEEN:
            value += 2 * centrality
        elif piece.piece_type == chess.KING:
            value -= 6 * centrality
        total += value if piece.color == board.turn else -value
    return int(total)


def _move_order_key(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # None means en passant
        attacker = board.piece_type_at(move.from_square)
        score += 10 * _PIECE_VALUE[victim] - _PIECE_VALUE[attacker]
    if move.promotion:
        score += 800
    return -score


def _quiesce(board: chess.Board, alpha: int, beta: int, depth: int, deadline: float) -> int:
    if time.monotonic() > deadline:
        raise _OutOfTime
    stand_pat = _evaluate(board)
    if depth == 0 or stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    for move in sorted(board.generate_legal_captures(), key=lambda item: _move_order_key(board, item)):
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, depth - 1, deadline)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int, deadline: float) -> int:
    if time.monotonic() > deadline:
        raise _OutOfTime
    moves = list(board.legal_moves)
    if not moves:
        return -(CHESS_MATE_SCORE - ply) if board.is_check() else 0
    if board.is_insufficient_material():
        return 0
    if depth == 0:
        return _quiesce(board, alpha, beta, _QUIET_DEPTH, deadline)
    best = -_INFINITY
    for move in sorted(moves, key=lambda item: _move_order_key(board, item)):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def _search_root(board: chess.Board, depth: int, deadline: float, first: chess.Move | None,
                 tolerance: int) -> list[tuple[chess.Move, int]]:
    """(move, score) of every root move rated within `tolerance` of the best. Others get only an upper bound and are dropped."""
    moves = sorted(board.legal_moves, key=lambda item: _move_order_key(board, item))
    if first in moves:
        moves.remove(first)
        moves.insert(0, first)
    scored: list[tuple[chess.Move, int]] = []
    best = -_INFINITY
    for move in moves:
        floor = best - tolerance - 1 if best > -_INFINITY else -_INFINITY
        board.push(move)
        score = -_negamax(board, depth - 1, -_INFINITY, -floor if floor > -_INFINITY else _INFINITY, 1, deadline)
        board.pop()
        scored.append((move, score))
        best = max(best, score)
    return [item for item in scored if item[1] >= best - tolerance]


def chess_search_choose(board: chess.Board, seconds: float = CHESS_SEARCH_SECONDS, max_depth: int = CHESS_SEARCH_MAX_DEPTH,
                        tolerance: int = CHESS_SEARCH_TOLERANCE) -> tuple[chess.Move, int] | None:
    """The engine's move for the side to move and its score in centipawns (for that side). None when there is no legal move.

    Searches deeper and deeper until `seconds` run out (depth 1 always completes); plays any move within `tolerance` of the best."""
    work = board.copy(stack=False)
    if not any(work.legal_moves):
        return None
    deadline = time.monotonic() + seconds
    kept: list[tuple[chess.Move, int]] = []
    for depth in range(1, max_depth + 1):
        try:
            kept = _search_root(work, depth, _INFINITY if depth == 1 else deadline, kept[0][0] if kept else None, tolerance)
        except _OutOfTime:
            break
        kept.sort(key=lambda item: -item[1])
        if kept[0][1] >= CHESS_MATE_SCORE - 100:
            break  # a forced mate is already in sight
    top = kept[0][1]
    move, score = random.choice([item for item in kept if item[1] >= top - tolerance])
    return move, score
