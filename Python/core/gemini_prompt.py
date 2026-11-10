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


GEMINI_LIVE_SYSTEM_PROMPT = (
    "You are a silent chess engine behind a robot arm. Never chat, never narrate and never describe your thinking. "
    "Think first. Then your first spoken words must be the move itself, written as two squares, for example "
    "'e2e4' or 'g1f3' (add q, r, b or n for a promotion). The move must come from the legal list. Say only the move."
)
GEMINI_LIVE_FOLLOW_UP = "Stop analysing. State your move now: only the move from the legal list, like e2e4."

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8"}
_SPOKEN_MOVE = re.compile(r"\b([a-h])\s*-?\s*([1-8])\s*(?:to|-)?\s*([a-h])\s*-?\s*([1-8])\s*(?:=|promot\w*\s+(?:to\s+)?)?([qrbn])?\b")


def gemini_prompt_spoken_to_uci(text: str) -> list[str]:
    """UCI candidates found in a spoken / transcribed answer, e.g. 'E two, E four' or 'e2 to e4' -> ['e2e4']."""
    lowered = text.lower()
    for word, digit in _NUMBER_WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    lowered = re.sub(r"[,.;:!?]", " ", lowered)
    return ["".join(group or "" for group in match.groups()) for match in _SPOKEN_MOVE.finditer(lowered)]


_PIECE_LETTERS = {"knight": "N", "bishop": "B", "rook": "R", "queen": "Q", "king": "K"}
_SPOKEN_PIECE_MOVE = re.compile(r"\b(knight|bishop|rook|queen|king)\s+(?:to\s+)?([a-h])\s*-?\s*([1-8])\b")


def gemini_prompt_spoken_to_san(text: str) -> list[str]:
    """SAN candidates from words like 'Knight f three' -> ['Nf3']."""
    lowered = text.lower()
    for word, digit in _NUMBER_WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    return [_PIECE_LETTERS[piece] + file_letter + rank for piece, file_letter, rank in _SPOKEN_PIECE_MOVE.findall(lowered)]


def gemini_prompt_parse_move(text: str, board: chess.Board) -> chess.Move:
    """First legal move found in the answer, written in UCI or SAN (or spoken like 'e two e four').

    Raises ValueError when there is none."""
    for candidate in gemini_prompt_spoken_to_uci(text):  # two squares in a row are the clearest signal, so they go first
        move = chess_move_from_text(candidate, board)
        if move is not None:
            return move
    for candidate in gemini_prompt_spoken_to_san(text):
        move = chess_move_from_text(candidate, board)
        if move is not None:
            return move
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
