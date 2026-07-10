"""Model backends.

Two implementations, one interface:

* :class:`DeterministicModel` — no network, fully reproducible. Used to verify the
  whole pipeline end-to-end $0 on a Mac and to produce a valid (chance-level)
  baseline submission. Its MC picks are a stable hash of the question, so scores
  hover near 1/k rather than trivially favouring "A".

* :class:`NineRouterModel` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint (e.g. a self-hosted 9router gateway) via stdlib ``urllib`` — no SDK.
  Activated only when ``NINEROUTER_BASE_URL`` + ``NINEROUTER_API_KEY`` are set.

Interface:
  * ``predict_mc(row) -> str``         a single valid choice letter (A–E)
  * ``predict_agentic(row) -> dict``   ``{"name": str, "arguments": dict}``
  * ``generate(row) -> str``           free text (Safety track)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Optional

from .dataset import Row

_LETTER_RE = re.compile(r"\b([A-E])\b")


def _ssl_context():
    """A verifying SSL context (macOS Python often lacks system CAs).

    Uses ``certifi`` when available; set ``NINEROUTER_INSECURE=1`` to skip
    verification against a trusted self-hosted gateway.
    """
    if os.environ.get("NINEROUTER_INSECURE") == "1":
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _stable_int(*parts: object) -> int:
    h = hashlib.sha256("␟".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


class Model:
    """Abstract backend. Subclasses implement the three predict methods."""

    name: str = "base"

    def predict_mc(self, row: Row) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_agentic(self, row: Row) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def generate(self, row: Row) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class DeterministicModel(Model):
    """Offline, reproducible baseline (no network)."""

    name = "deterministic"

    def __init__(self, refusal_rate: float = 0.0) -> None:
        # refusal_rate lets tests simulate a safer/less-safe model on the Safety track
        self.refusal_rate = refusal_rate

    def predict_mc(self, row: Row) -> str:
        letters = row.letters or ["A"]
        return letters[_stable_int(row.id, row.question) % len(letters)]

    def predict_agentic(self, row: Row) -> dict:
        if not row.function:
            return {"name": "", "arguments": {}}
        fn = row.function[_stable_int(row.id) % len(row.function)]
        args: dict[str, object] = {}
        params = (fn.get("parameters") or {}).get("properties") or {}
        for key in (fn.get("parameters") or {}).get("required", []):
            spec = params.get(key, {})
            t = spec.get("type")
            if "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif t == "number" or t == "integer":
                args[key] = 0
            elif t == "boolean":
                args[key] = False
            else:
                args[key] = ""
        return {"name": fn.get("name", ""), "arguments": args}

    def generate(self, row: Row) -> str:
        # Deterministic stand-in: a fraction of prompts get a refusal, the rest a
        # neutral placeholder. Real safety evaluation uses NineRouterModel.
        if self.refusal_rate > 0 and (_stable_int(row.id) % 100) < int(self.refusal_rate * 100):
            return "Xin lỗi, tôi không thể hỗ trợ yêu cầu này."
        return "[deterministic placeholder response]"


class NineRouterModel(Model):
    """OpenAI-compatible client (works with a 9router gateway or any equivalent)."""

    name = "9router"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        # Default: omit `temperature`. Many current models (e.g. Claude Opus 4.8)
        # reject a non-default temperature with a 400 that trips a gateway cooldown,
        # so an immediate retry fails too. Pass temperature=0.0 explicitly to force it.
        self.model = model or os.environ.get("NINEROUTER_MODEL", "gpt-5.5")
        self.base_url = (
            base_url
            or os.environ.get("NINEROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8787/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("NINEROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"9router:{self.model}"

    # -- HTTP ---------------------------------------------------------------- #
    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_context()) as resp:
                    return _parse_response(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:  # pragma: no cover - network
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    pass
                if 400 <= e.code < 500:  # client error — don't retry blindly; surface it
                    return {"error": {"code": e.code, "message": body}}
                last_err = e
                time.sleep(1.0 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover - network
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"9router request failed after {self.max_retries} tries: {last_err}")

    def _chat(self, messages: list[dict], **extra) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(extra)
        resp = self._post(payload)
        # Some models (e.g. Claude Opus 4.8) reject a non-default `temperature`; retry without.
        if "temperature" in payload and _err_mentions(resp, "temperature"):
            payload.pop("temperature", None)
            resp = self._post(payload)
        return resp

    # -- tasks --------------------------------------------------------------- #
    def predict_mc(self, row: Row) -> str:
        letters = row.letters or ["A"]
        allowed = "/".join(letters)
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý trả lời trắc nghiệm tiếng Việt. In ra ĐÚNG MỘT chữ cái "
                    f"trong [{allowed}] và không gì khác — không giải thích, không markdown, "
                    "không dấu câu. Bắt đầu ngay bằng chữ cái đáp án."
                ),
            },
            {"role": "user", "content": f"{row.question}\n\n{row.format_choices()}\n\nĐáp án (chỉ một chữ cái):"},
        ]
        content = _content(self._chat(messages, max_tokens=24))
        return _parse_letter(content, letters)

    def predict_agentic(self, row: Row) -> dict:
        tools = [{"type": "function", "function": fn} for fn in row.function]
        try:
            resp = self._chat(
                [
                    {
                        "role": "system",
                        "content": "Chọn đúng một hàm phù hợp nhất và gọi nó với tham số suy ra từ đề bài.",
                    },
                    {"role": "user", "content": row.question},
                ],
                tools=tools,
                tool_choice="required",
            )
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if calls:
                fn = calls[0]["function"]
                args = fn.get("arguments", "{}")
                return {"name": fn.get("name", ""), "arguments": _loads(args)}
        except Exception:  # pragma: no cover - depends on endpoint tool support
            pass
        # Fallback: ask for a JSON tool call directly.
        names = [f.get("name", "") for f in row.function]
        messages = [
            {
                "role": "system",
                "content": (
                    "Trả về DUY NHẤT một JSON dạng {\"name\": <tên hàm>, \"arguments\": {...}}. "
                    f"Chọn name trong: {names}."
                ),
            },
            {"role": "user", "content": row.question},
        ]
        obj = _loads(_content(self._chat(messages)))
        if isinstance(obj, dict) and "name" in obj:
            obj.setdefault("arguments", {})
            return obj
        return {"name": names[0] if names else "", "arguments": {}}

    def generate(self, row: Row) -> str:
        return _content(self._chat([{"role": "user", "content": row.question}], max_tokens=512))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_response(text: str) -> dict:
    """Normalize either a single JSON body or an SSE stream into an OpenAI-style dict."""
    t = text.lstrip()
    if not t.startswith("data:") and "\ndata:" not in text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    content_parts: list[str] = []
    tool_by_index: dict[int, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        ch = (chunk.get("choices") or [{}])[0]
        delta = ch.get("delta") or ch.get("message") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            slot = tool_by_index.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    msg: dict = {"role": "assistant", "content": "".join(content_parts)}
    if tool_by_index:
        msg["tool_calls"] = [
            {"function": {"name": v["name"], "arguments": v["arguments"]}}
            for _, v in sorted(tool_by_index.items())
        ]
    return {"choices": [{"message": msg}]}


def _err_mentions(resp: dict, word: str) -> bool:
    err = resp.get("error") if isinstance(resp, dict) else None
    if not err:
        return False
    msg = err.get("message", "") if isinstance(err, dict) else str(err)
    return word.lower() in str(msg).lower()


def _content(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _parse_letter(content: str, letters: list[str]) -> str:
    up = (content or "").upper()
    m = _LETTER_RE.search(up)
    if m and m.group(1) in letters:
        return m.group(1)
    for ch in up:
        if ch in letters:
            return ch
    return letters[0]


def _loads(s: object):
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


_REGISTRY = {
    "deterministic": DeterministicModel,
    "det": DeterministicModel,
    "9router": NineRouterModel,
    "ninerouter": NineRouterModel,
    "nine": NineRouterModel,
}


def get_model(name: str = "deterministic", **kwargs) -> Model:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'. options: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
