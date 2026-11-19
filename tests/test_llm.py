import json

import httpx
import pytest

from mindbridge.config import Settings
from mindbridge.llm import (
    AnthropicLLM,
    LLMError,
    LLMUnavailable,
    NullLLM,
    OllamaLLM,
    OpenAILLM,
    make_llm,
    merge_turns,
)

MSGS = [{"role": "user", "content": "hello"}]


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_merge_turns_joins_runs_and_drops_leading_assistant():
    merged = merge_turns(
        [
            {"role": "assistant", "content": "welcome"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
    )
    assert [m["role"] for m in merged] == ["user", "assistant"]
    assert merged[0]["content"] == "a\n\nb"


def test_anthropic_request_shape_and_parsing():
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": " hi there "}]})

    llm = AnthropicLLM("k", "m1", client=client(handler))
    assert llm.generate("sys", MSGS, max_tokens=50, temperature=0.2) == "hi there"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["messages"][0]["role"] == "user"
    assert seen["body"]["max_tokens"] == 50


def test_openai_puts_system_prompt_first():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAILLM("k", "gpt", "https://example.test/v1/", client=client(handler))
    assert llm.generate("sys", MSGS, max_tokens=10, temperature=0) == "ok"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_parses_message_content():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"content": "local reply"}})

    llm = OllamaLLM("http://localhost:11434", "llama", client=client(handler))
    assert llm.generate("sys", MSGS, max_tokens=10, temperature=0) == "local reply"


def test_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "finally"}}]})

    sleeps = []
    llm = OpenAILLM("k", "m", "https://x/v1", client=client(handler), sleep=sleeps.append)
    assert llm.generate("s", MSGS, max_tokens=5, temperature=0) == "finally"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.5]


def test_gives_up_after_repeated_server_errors():
    llm = OpenAILLM(
        "k", "m", "https://x/v1", client=client(lambda r: httpx.Response(503)), sleep=lambda s: None
    )
    with pytest.raises(LLMUnavailable):
        llm.generate("s", MSGS, max_tokens=5, temperature=0)


def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401)

    llm = OpenAILLM("k", "m", "https://x/v1", client=client(handler), sleep=lambda s: None)
    with pytest.raises(LLMError) as exc:
        llm.generate("s", MSGS, max_tokens=5, temperature=0)
    assert not isinstance(exc.value, LLMUnavailable)
    assert calls["n"] == 1


def test_network_failure_becomes_unavailable():
    def handler(request):
        raise httpx.ConnectError("boom")

    llm = OllamaLLM("http://x", "m", client=client(handler), sleep=lambda s: None)
    with pytest.raises(LLMUnavailable):
        llm.generate("s", MSGS, max_tokens=5, temperature=0)


def test_empty_completion_is_an_error():
    llm = OpenAILLM(
        "k",
        "m",
        "https://x/v1",
        client=client(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        ),
    )
    with pytest.raises(LLMError):
        llm.generate("s", MSGS, max_tokens=5, temperature=0)


def test_factory_picks_provider_and_validates_keys():
    base = {"_env_file": None}
    assert isinstance(make_llm(Settings(**base)), NullLLM)
    assert isinstance(make_llm(Settings(**base, llm_provider="ollama")), OllamaLLM)
    with pytest.raises(LLMError):
        make_llm(Settings(**base, llm_provider="openai"))
    ok = make_llm(Settings(**base, llm_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(ok, AnthropicLLM)


def test_null_llm_always_unavailable():
    with pytest.raises(LLMUnavailable):
        NullLLM().generate("s", MSGS, max_tokens=1, temperature=0)
