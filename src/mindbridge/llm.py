from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """No provider configured, or it could not be reached after retries."""


class LLM(Protocol):
    name: str

    def generate(
        self, system: str, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> str: ...


def merge_turns(messages: list[Message]) -> list[Message]:
    """Providers want alternating roles that start with the user. Merge runs of the same
    role and drop any leading assistant turns."""
    merged: list[Message] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {
                "role": m["role"],
                "content": merged[-1]["content"] + "\n\n" + m["content"],
            }
        else:
            merged.append(dict(m))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


class _HttpLLM:
    name = "http"
    retry_statuses = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 3,
    ):
        self._client = client or httpx.Client(timeout=timeout_s)
        self._sleep = sleep
        self._attempts = attempts

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last = exc
            else:
                if resp.status_code in self.retry_statuses:
                    last = LLMError(f"{self.name}: HTTP {resp.status_code}")
                elif resp.is_success:
                    return resp.json()
                else:
                    # a 4xx other than 408/429 will not get better by retrying
                    raise LLMError(f"{self.name}: HTTP {resp.status_code}")
            if attempt < self._attempts - 1:
                self._sleep(0.5 * (3**attempt))
        raise LLMUnavailable(f"{self.name}: gave up after {self._attempts} attempts ({last})")


class AnthropicLLM(_HttpLLM):
    def __init__(self, api_key: str, model: str, **kw: Any):
        super().__init__(**kw)
        self.name = f"anthropic:{model}"
        self._model = model
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def generate(self, system, messages, *, max_tokens, temperature):
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            self._headers,
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": merge_turns(messages),
            },
        )
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        if not text.strip():
            raise LLMError("anthropic: empty completion")
        return text.strip()


class OpenAILLM(_HttpLLM):
    def __init__(self, api_key: str, model: str, base_url: str, **kw: Any):
        super().__init__(**kw)
        self.name = f"openai:{model}"
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}

    def generate(self, system, messages, *, max_tokens, temperature):
        data = self._post(
            self._url,
            self._headers,
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *merge_turns(messages)],
            },
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("openai: unexpected response shape") from exc
        if not text.strip():
            raise LLMError("openai: empty completion")
        return text.strip()


class OllamaLLM(_HttpLLM):
    def __init__(self, base_url: str, model: str, **kw: Any):
        super().__init__(**kw)
        self.name = f"ollama:{model}"
        self._model = model
        self._url = base_url.rstrip("/") + "/api/chat"

    def generate(self, system, messages, *, max_tokens, temperature):
        data = self._post(
            self._url,
            {"content-type": "application/json"},
            {
                "model": self._model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, *merge_turns(messages)],
            },
        )
        text = (data.get("message") or {}).get("content", "")
        if not text.strip():
            raise LLMError("ollama: empty completion")
        return text.strip()


class NullLLM:
    """Used when no provider is configured. The pipeline answers extractively instead."""

    name = "none"

    def generate(self, system, messages, *, max_tokens, temperature):
        raise LLMUnavailable("no LLM provider configured")


def make_llm(settings: Settings, client: httpx.Client | None = None) -> LLM:
    kw: dict[str, Any] = {"timeout_s": settings.llm_timeout_s, "client": client}
    provider = settings.llm_provider
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model, **kw)
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        return OpenAILLM(
            settings.openai_api_key, settings.openai_model, settings.openai_base_url, **kw
        )
    if provider == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model, **kw)
    return NullLLM()
