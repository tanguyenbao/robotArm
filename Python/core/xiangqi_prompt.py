"""Builds the prompt asking Gemini for a xiangqi move and reads the answer back (pure text)."""
from __future__ import annotations

import re

from core.xiangqi_rules import (XiangqiMove, XiangqiPosition, xiangqi_rules_ascii, xiangqi_rules_legal_moves,
                                xiangqi_rules_move_from_text)

XIANGQI_LIVE_SYSTEM_PROMPT = (
    "You are a silent xiangqi (Chinese chess) engine behind a robot arm. Never chat, never narrate and never describe your "
    "thinking. Think first. Then your first spoken words must be the move itself, written as two squares in ICCS notation, "
    "for example 'h2e2' or 'b9c7'. The move must come from the legal list. Say only the move."
)
XIANGQI_LIVE_FOLLOW_UP = "Stop analysing. State your move now: only the move from the legal list, like h2e2."

_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9"}
_SPOKEN_MOVE = re.compile(r"\b([a-i])\s*-?\s*([0-9])\s*(?:to|-)?\s*([a-i])\s*-?\s*([0-9])\b")


def xiangqi_prompt_build(position: XiangqiPosition, history: list[str], feedback: str = "") -> str:
    """Prompt for the side to move. `feedback` explains why the previous answer was refused."""
    side = "Red" if position.red_to_move else "Black"
    legal = " ".join(move.iccs() for move in xiangqi_rules_legal_moves(position))
    lines = [
        f"You are a strong xiangqi (Chinese chess) engine playing {side}. A robot arm will play your move on a real board.",
        "Board (files a-i left to right, ranks 0-9; Red is uppercase and starts at ranks 0-4, Black is lowercase and starts at ranks 5-9; "
        "K general, A advisor, E elephant, H horse, R chariot, C cannon, P soldier; '.' is empty):",
        xiangqi_rules_ascii(position),
        f"Moves so far (ICCS): {' '.join(history) if history else '(none yet)'}",
        f"Legal moves (ICCS): {legal}",
        "Choose the best move. Answer with EXACTLY ONE move taken from the legal list, in ICCS notation (for example h2e2), and nothing else.",
    ]
    if feedback:
        lines.append(f"Your previous answer was refused: {feedback} Pick again from the legal list.")
    return "\n".join(lines)


def xiangqi_prompt_parse_move(text: str, position: XiangqiPosition) -> XiangqiMove:
    """First legal move found in the answer, written like 'h2e2', 'h2-e2' or spoken like 'h two e two'.

    Raises ValueError when there is none."""
    lowered = text.lower()
    for word, digit in _NUMBER_WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    lowered = re.sub(r"[,.;:!?]", " ", lowered)
    for match in _SPOKEN_MOVE.finditer(lowered):
        move = xiangqi_rules_move_from_text("".join(match.groups()), position)
        if move is not None:
            return move
    raise ValueError(f"no legal move found in the answer: {text.strip()[:80]!r}")
