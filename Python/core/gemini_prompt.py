"""Builds the prompt asking Gemini for a chess move and reads the answer back (pure text and python-chess)."""
from __future__ import annotations

import re

import chess

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9=+#\-]+")


def gemini_prompt_build(board: chess.Board, san_history: list[str], feedback: str = "") -> str:
    """Prompt for the side to move. `feedback` explains why the previous answer was refused."""
    side = "White" if board.turn == chess.WHITE else "Black"
    legal = " ".join(move.uci() for move in board.legal_moves)
    history = " ".join(san_history) if san_history else "(none yet)"
    lines = [
        f"You are a strong chess engine playing {side}. A robot arm will play your move on a real board.",
        f"Position (FEN): {board.fen()}",
        f"Moves so far (SAN): {history}",
        f"Legal moves (UCI): {legal}",
        "Choose the best move. Answer with EXACTLY ONE move taken from the legal list, in UCI notation (for example e2e4 or e7e8q), and nothing else.",
    ]
    if feedback:
        lines.append(f"Your previous answer was refused: {feedback} Pick again from the legal list.")
    return "\n".join(lines)


def gemini_prompt_parse_move(text: str, board: chess.Board) -> chess.Move:
    """First legal move found in the answer, written in UCI or SAN. Raises ValueError when there is none."""
    for token in _TOKEN_PATTERN.findall(text):
        move = chess_move_from_text(token, board)
        if move is not None:
            return move
    raise ValueError(f"no legal move found in the answer: {text.strip()[:80]!r}")


def chess_move_from_text(token: str, board: chess.Board) -> chess.Move | None:
    """Parse one UCI or SAN token. A pawn reaching the last rank without a letter becomes a queen. None if illegal."""
    candidate = token.strip()
    for parser in (board.parse_uci, board.parse_san):
        try:
            return parser(candidate)
        except ValueError:
            continue
    try:  # UCI pawn move to the last rank with the promotion letter left out
        move = chess.Move.from_uci(candidate.lower())
    except ValueError:
        return None
    queen_move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
    return queen_move if queen_move in board.legal_moves else None
