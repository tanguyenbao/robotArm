"""Gemini REST client (standard library only). The API key is passed in per call and never stored or logged."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(Exception):
    """Any failure talking to the Gemini API (the message never contains the key)."""


@dataclass(frozen=True)
class GeminiModel:
    name: str          # id to use in requests, e.g. gemini-2.5-flash
    display_name: str


def _request(url: str, api_key: str, payload: dict | None, timeout_s: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST")
    request.add_header("x-goog-api-key", api_key)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = error.reason
        raise GeminiError(f"HTTP {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise GeminiError(f"network error: {error}") from None


def gemini_client_list_models(api_key: str, timeout_s: float = 20.0) -> list[GeminiModel]:
    """Every model of this key that can generate text, sorted by name."""
    if not api_key:
        raise GeminiError("no API key (type it in the field or set GEMINI_API_KEY)")
    models: list[GeminiModel] = []
    page_token = ""
    while True:
        url = f"{_API_BASE}/models?pageSize=200" + (f"&pageToken={page_token}" if page_token else "")
        answer = _request(url, api_key, None, timeout_s)
        for entry in answer.get("models", []):
            if "generateContent" in entry.get("supportedGenerationMethods", []):
                name = entry["name"].removeprefix("models/")
                models.append(GeminiModel(name, entry.get("displayName", name)))
        page_token = answer.get("nextPageToken", "")
        if not page_token:
            break
    return sorted(models, key=lambda model: model.name)


def gemini_client_generate_text(api_key: str, model_name: str, prompt: str, temperature: float = 0.4,
                                timeout_s: float = 90.0) -> str:
    """One prompt in, the text of the first answer out."""
    if not api_key:
        raise GeminiError("no API key (type it in the field or set GEMINI_API_KEY)")
    if not model_name:
        raise GeminiError("no model selected")
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": temperature}}
    answer = _request(f"{_API_BASE}/models/{model_name}:generateContent", api_key, payload, timeout_s)
    candidates = answer.get("candidates", [])
    if not candidates:
        reason = answer.get("promptFeedback", {}).get("blockReason", "empty answer")
        raise GeminiError(f"no answer ({reason})")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise GeminiError(f"no text in the answer (finish reason {candidates[0].get('finishReason', '?')})")
    return text
