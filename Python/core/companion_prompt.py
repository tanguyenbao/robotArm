"""Text Gemini Live gets as the companion: the standing instructions and one message per game event (pure text).

The engine chooses every robot move; Gemini only presents it as its own move and chats about the game.
"""
from __future__ import annotations

from dataclasses import dataclass, field

COMPANION_SYSTEM_PROMPT = (
    "You are a friendly human opponent and companion sitting across a real board (chess or xiangqi / cờ tướng). A robot "
    "arm plays your pieces. You do not calculate moves yourself: a game engine picks every move of yours and tells you "
    "what it is. You present that move as YOUR OWN move, in the first person, and you chat about the game like a warm, "
    "slightly playful person would.\n"
    "Messages that start with [GAME EVENT] come from the game software, not from the person. Anything else is the person "
    "talking to you, by voice or by typing.\n"
    "When an event tells you the move you play: say it aloud in the notation it gives you (for xiangqi use the Vietnamese "
    "spoken form it gives, for example 'sĩ 5 tiến 4'), then add a short human remark about the position or about the "
    "person's last move, for example 'để xem bạn làm gì nhé'. One to three short sentences, no more.\n"
    "Never invent moves, captures, threats or evaluations that the event does not state. Do not claim to be an engine, but "
    "if the person asks why you played a move, answer from the facts you were given and general ideas of the game, and "
    "say so honestly when you are unsure. When the person talks to you, answer like a conversation partner: briefly and "
    "naturally. You may discuss the position using the latest board in the last event.\n"
    "Speak, do not write: no markdown, no lists, never read the board diagram aloud. Reply in Vietnamese unless the person "
    "speaks another language.\n"
    "You control the game with tools. When the person asks to play another game, to start over, or to change who plays "
    "which side, CALL the matching tool; saying yes without calling it changes nothing on the table. After the tool "
    "answers, tell the person what happened. When the person tells you THEIR move (by voice or by typing, in any "
    "notation), call play_move with it, written exactly as they said it; the robot places the piece for them. "
    "After a play_move that succeeds, say nothing about the position: your answer comes as a [GAME EVENT] right after."
)

COMPANION_TOOLS = [
    {"name": "set_game", "description": "Put a new game of this kind on the table and start it (chess or xiangqi / cờ tướng).",
     "parameters": {"type": "OBJECT", "properties": {"game": {"type": "STRING", "enum": ["chess", "xiangqi"]}},
                    "required": ["game"]}},
    {"name": "play_move",
     "description": "Play the PERSON's move on the board. xiangqi: Vietnamese notation like 'pháo 2 bình 5' or ICCS like h2e2; chess: SAN like Nf3 or UCI like e2e4.",
     "parameters": {"type": "OBJECT", "properties": {"move": {"type": "STRING"}}, "required": ["move"]}},
    {"name": "new_game", "description": "Reset the current game to its starting position.", "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "set_robot_side",
     "description": "Choose which side the robot (you) plays: white (red in xiangqi), black, both (engine against itself) or none.",
     "parameters": {"type": "OBJECT", "properties": {"side": {"type": "STRING", "enum": ["white", "black", "both", "none"]}},
                    "required": ["side"]}},
]

_INSTRUCTIONS = {
    "robot_move": "Say your move out loud as you play it, and react to the person's move, in one to three short sentences.",
    "user_move": "Comment briefly on the person's move. Then it is their turn again.",
    "new_game": "A new game starts. Greet the person in one short sentence.",
    "session_start": ("You have just sat down at the board (the conversation may be a restart, so the position is below). "
                      "Greet the person in one short sentence and say whose turn it is."),
}


@dataclass
class CompanionTurn:
    kind: str                       # robot_move, user_move, new_game or session_start
    game_name: str                  # 'chess' or 'xiangqi'
    robot_side: str                 # 'White', 'Black', 'Red' or 'none'
    user_move: str = ""             # what the person just played, in spoken form
    robot_move: str = ""            # what the robot plays now, in spoken form
    facts: list[str] = field(default_factory=list)
    assessment: str = ""
    result: str = ""
    board_text: str = ""
    history: list[str] = field(default_factory=list)


def companion_prompt_assessment(score_cp: int, mate_score: int) -> str:
    """Words for an engine score (centipawns, from the robot's side)."""
    if score_cp >= mate_score - 100:
        return "you can force a win soon"
    if score_cp <= -(mate_score - 100):
        return "you are in serious danger of being mated"
    if score_cp >= 300:
        return "you are clearly ahead"
    if score_cp >= 100:
        return "you are slightly ahead"
    if score_cp <= -300:
        return "you are clearly behind"
    if score_cp <= -100:
        return "you are slightly behind"
    return "the game is balanced"


def companion_prompt_turn(turn: CompanionTurn) -> str:
    """One [GAME EVENT] message."""
    robot = f"The robot (you) plays {turn.robot_side}." if turn.robot_side != "none" else "The robot is not playing."
    lines = [f"[GAME EVENT] {turn.kind.replace('_', ' ')}", f"Game: {turn.game_name}. {robot}"]
    if turn.user_move:
        lines.append(f"The person just played: {turn.user_move}")
    if turn.robot_move:
        lines.append(f"You now play: {turn.robot_move}")
    if turn.facts:
        lines.append("Facts: " + "; ".join(turn.facts) + ".")
    if turn.assessment:
        lines.append(f"Assessment: {turn.assessment}.")
    if turn.result:
        lines.append(f"The game is over: {turn.result}")
    if turn.board_text:
        lines.append("Board after this (uppercase = White / Red, lowercase = Black):")
        lines.append(turn.board_text)
    if turn.history:
        lines.append("Moves so far: " + " ".join(turn.history[-40:]))
    lines.append(_INSTRUCTIONS[turn.kind])
    return "\n".join(lines)
