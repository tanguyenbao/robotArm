"""Terminal check: can a Gemini / Gemma model play chess for the arm?

Uses the same client, prompt and parser as the GUI, so a PASS here means the GUI will work with that model.
The API key is read from --key, GEMINI_API_KEY, or a hidden prompt. It is never printed.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python"))

import chess  # noqa: E402

from adapters.gemini_client import GeminiError, gemini_client_generate_text, gemini_client_list_models  # noqa: E402
from adapters.gemini_live_client import gemini_live_generate_text  # noqa: E402
from core.gemini_prompt import (GEMINI_LIVE_FOLLOW_UP, GEMINI_LIVE_SYSTEM_PROMPT, gemini_prompt_build,  # noqa: E402
                                gemini_prompt_parse_move)

# name, FEN, SAN history shown to the model
CHECK_POSITIONS = (
    ("opening, white to move", chess.STARTING_FEN, []),
    ("opening, black to move", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", ["e4"]),
    ("middlegame, white can castle", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Nf3", "Nf6"]),
    ("promotion available", "8/4P3/8/8/8/8/k6K/8 w - - 0 1", []),
)


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _log(text: str) -> None:
    print(f"[{_stamp()}] {text}", flush=True)


def _resolve_key(argument: str) -> str:
    key = argument or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        key = getpass.getpass("Gemini API key (hidden): ")
    cleaned = "".join(key.split())  # a paste can carry spaces, tabs or a line break
    _log(f"key: {len(cleaned)} characters, starts with AIza: {cleaned.startswith('AIza')}"
         f"{' (whitespace removed)' if cleaned != key else ''}")
    return cleaned


_TRACE = False


def _ask(api_key: str, model_name: str, prompt: str, temperature: float | None, live: bool, board: chess.Board,
        timeout_s: float = 90.0) -> str:
    """One question to the model, over the normal API or the Live API."""
    if live:
        def has_move(text: str) -> bool:
            try:
                gemini_prompt_parse_move(text, board)
                return True
            except ValueError:
                return False

        def trace(line: str) -> None:
            if _TRACE or line.startswith("no usable"):
                _log(f"      live> {line}")

        return gemini_live_generate_text(api_key, model_name, prompt, max(timeout_s, 120.0), trace=trace,
                                         system_instruction=GEMINI_LIVE_SYSTEM_PROMPT, accept=has_move,
                                         follow_up=GEMINI_LIVE_FOLLOW_UP)
    return gemini_client_generate_text(api_key, model_name, prompt, temperature, timeout_s,
                                       on_retry=lambda text: _log(f"      {text}"))


def _list_models(api_key: str, name_filter: str, live: bool) -> int:
    _log(f"Asking the API for models that support {'the Live API (bidiGenerateContent)' if live else 'text generation'} ...")
    try:
        models = gemini_client_list_models(api_key, method="bidiGenerateContent" if live else "generateContent")
    except GeminiError as error:
        _log(f"FAIL  could not list models: {error}")
        return 1
    shown = [model for model in models if name_filter.lower() in model.name.lower()]
    for model in shown:
        print(f"  {model.name:<44} {model.display_name}")
    _log(f"{len(shown)} of {len(models)} models shown (filter: {name_filter or 'none'}).")
    if not shown:
        _log("Nothing matches. You can still type a name by hand; the list is only a hint.")
    return 0


def _check_position(api_key: str, model_name: str, title: str, fen: str, history: list[str], tries: int,
                    temperature: float | None, live: bool) -> bool:
    board = chess.Board(fen)
    legal_count = board.legal_moves.count()
    _log(f"--- {title}  ({'White' if board.turn else 'Black'} to move, {legal_count} legal moves)")
    feedback = ""
    for attempt in range(1, tries + 1):
        prompt = gemini_prompt_build(board, history, feedback)
        _log(f"try {attempt}/{tries}: sending {len(prompt)} characters to {model_name} ...")
        started = time.perf_counter()
        try:
            answer = _ask(api_key, model_name, prompt, temperature, live, board)
        except GeminiError as error:
            _log(f"      request failed: {error}")
            return False
        elapsed = time.perf_counter() - started
        shown = " ".join(answer.split())
        _log(f"      answer in {elapsed:.1f}s: {shown[:200]}{' ...' if len(shown) > 200 else ''}")
        try:
            move = gemini_prompt_parse_move(answer, board)
        except ValueError as error:
            _log(f"      not usable: {error}")
            feedback = "Your last answer had no legal move. Reply with one legal move only."
            continue
        _log(f"      PASS  move {move.uci()} ({board.san(move)}) is legal")
        return True
    _log(f"      FAIL  no legal move after {tries} tries")
    return False


def _sweep(api_key: str, name_filter: str, temperature: float | None, live: bool) -> int:
    """Ask every matching model for one opening move and show which ones answer."""
    try:
        listed = gemini_client_list_models(api_key, method="bidiGenerateContent" if live else "generateContent")
        models = [model for model in listed if name_filter.lower() in model.name.lower()]
    except GeminiError as error:
        _log(f"FAIL  could not list models: {error}")
        return 1
    _log(f"Sweeping {len(models)} models (filter: {name_filter or 'none'}), 1 position, 1 try each.")
    board = chess.Board()
    rows = []
    for model in models:
        started = time.perf_counter()
        try:
            answer = _ask(api_key, model.name, gemini_prompt_build(board, []), temperature, live, board, 60.0)
            move = gemini_prompt_parse_move(answer, board)
            verdict = f"PASS  {board.san(move)}"
        except GeminiError as error:
            verdict = f"FAIL  {error}"
        except ValueError as error:
            verdict = f"FAIL  unusable answer: {error}"
        rows.append((model.name, verdict))
        _log(f"{model.name:<44} {verdict[:90]}  ({time.perf_counter() - started:.1f}s)")
    passed = [name for name, verdict in rows if verdict.startswith("PASS")]
    _log(f"RESULT  {len(passed)}/{len(rows)} models answered with a legal move.")
    for name in passed:
        print(f"  usable: {name}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a Gemini / Gemma model can play chess for the arm.")
    parser.add_argument("--model", default="", help="model id, e.g. a Gemma name from --list")
    parser.add_argument("--list", nargs="?", const="", default=None, metavar="FILTER",
                        help="list models that can generate text (optional name filter, e.g. gemma) and exit")
    parser.add_argument("--key", default="", help="API key (default: GEMINI_API_KEY, else asked)")
    parser.add_argument("--tries", type=int, default=3, help="tries per position, like the GUI (default 3)")
    parser.add_argument("--sweep", nargs="?", const="", default=None, metavar="FILTER",
                        help="quick test of every listed model (optional name filter, e.g. gemma): 1 position, 1 try each")
    parser.add_argument("--live", action="store_true",
                        help="use the Live API (WebSocket) instead of generateContent; works with --list, --sweep, --model")
    parser.add_argument("--trace", action="store_true",
                        help="with --live: print one line per server message (flags, transcription, text parts; no audio)")
    parser.add_argument("--bare", action="store_true",
                        help="send only the prompt, no generationConfig (isolates HTTP 500 caused by a setting)")
    arguments = parser.parse_args()
    temperature = None if arguments.bare else 0.4
    global _TRACE
    _TRACE = arguments.trace

    api_key = _resolve_key(arguments.key)
    if not api_key:
        _log("FAIL  no API key")
        return 1
    if arguments.list is not None:
        return _list_models(api_key, arguments.list, arguments.live)
    if arguments.sweep is not None:
        return _sweep(api_key, arguments.sweep, temperature, arguments.live)
    if not arguments.model:
        _log("FAIL  give --model <id>, or run with --list gemma to see the names")
        return 1

    _log(f"Checking model '{arguments.model}' on {len(CHECK_POSITIONS)} positions.")
    results = [(title, _check_position(api_key, arguments.model, title, fen, history, arguments.tries, temperature, arguments.live))
               for title, fen, history in CHECK_POSITIONS]
    print()
    for title, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {title}")
    passed_count = sum(passed for _, passed in results)
    _log(f"{passed_count}/{len(results)} positions passed.")
    if passed_count == len(results):
        _log("RESULT  OK: this model can drive the arm.")
        return 0
    _log("RESULT  NOT RELIABLE: use a larger model, or send me the log above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
