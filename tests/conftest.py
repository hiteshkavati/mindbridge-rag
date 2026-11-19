from __future__ import annotations

from pathlib import Path

import pytest

from mindbridge.config import Settings
from mindbridge.embeddings import LsaEmbedder
from mindbridge.index import HybridRetriever
from mindbridge.kb import build_chunks
from mindbridge.llm import LLMError, LLMUnavailable
from mindbridge.pipeline import Pipeline
from mindbridge.safety import load_helplines

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Records calls and returns a canned reply (or raises)."""

    name = "fake"

    def __init__(self, reply: str = "Try slow breathing [1]. What feels hardest right now?"):
        self.reply = reply
        self.calls: list[dict] = []
        self.error: Exception | None = None

    def generate(self, system, messages, *, max_tokens, temperature):
        self.calls.append({"system": system, "messages": messages})
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, embedding_backend="lsa", llm_provider="none")


@pytest.fixture(scope="session")
def retriever(settings) -> HybridRetriever:
    return HybridRetriever(build_chunks(ROOT / "data" / "kb"), LsaEmbedder())


@pytest.fixture(scope="session")
def helplines():
    return load_helplines(ROOT / "data" / "helplines.json")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(retriever, fake_llm, helplines, settings) -> Pipeline:
    return Pipeline(retriever, fake_llm, helplines, settings)


__all__ = ["FakeLLM", "LLMError", "LLMUnavailable", "ROOT"]
