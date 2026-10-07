"""OpenAI JSON client without a network."""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import openai
import pytest

from cutter.llm import EndpointUnreachable, OpenAIJsonClient

_SCHEMA = {"type": "object"}


class _Completions:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        message = SimpleNamespace(content=step)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _install(monkeypatch: pytest.MonkeyPatch, completions: _Completions) -> list[dict[str, object]]:
    constructed: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr("cutter.llm.OpenAI", _FakeOpenAI)
    return constructed


def _client(
    monkeypatch: pytest.MonkeyPatch, steps: list[object]
) -> tuple[OpenAIJsonClient, _Completions, list[dict[str, object]]]:
    completions = _Completions(steps)
    constructed = _install(monkeypatch, completions)
    client = OpenAIJsonClient(base_url="http://localhost:1234/v1", model="qwen", timeout_s=5)
    return client, completions, constructed


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "http://localhost/v1/chat/completions")


def _bad_request() -> openai.BadRequestError:
    response = httpx2.Response(400, request=_request())
    return openai.BadRequestError("rejected schema", response=response, body=None)


def _status(code: int) -> openai.APIStatusError:
    response = httpx2.Response(code, request=_request())
    return openai.APIStatusError("status", response=response, body=None)


def test_structured_output_returns_the_object(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, constructed = _client(monkeypatch, ['{"choice": "B", "confidence": 0.9}'])

    assert client.complete_json("sys", "user", _SCHEMA) == {"choice": "B", "confidence": 0.9}
    assert constructed == [
        {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio", "timeout": 5}
    ]
    call = completions.calls[0]
    assert call["model"] == "qwen"
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "retake_verdict", "strict": True, "schema": _SCHEMA},
    }


def test_bad_request_retries_plain_json_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(
        monkeypatch, [_bad_request(), '{"choice": "A", "confidence": 1}']
    )

    assert client.complete_json("sys", "judge these", _SCHEMA) == {"choice": "A", "confidence": 1}
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert completions.calls[1]["messages"][1]["content"] == (
        "judge these\nreply with a single JSON object matching the schema, no markdown."
    )


def test_status_400_retries_plain_json_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(monkeypatch, [_status(400), '{"ok": true}'])

    assert client.complete_json("sys", "user", _SCHEMA) == {"ok": True}
    assert len(completions.calls) == 2
    assert "response_format" not in completions.calls[1]


def test_plain_retry_does_not_retry_again(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(monkeypatch, [_bad_request(), _bad_request()])

    with pytest.raises(openai.BadRequestError):
        client.complete_json("sys", "user", _SCHEMA)
    assert len(completions.calls) == 2


def test_status_500_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(monkeypatch, [_status(500)])

    with pytest.raises(openai.APIStatusError):
        client.complete_json("sys", "user", _SCHEMA)
    assert len(completions.calls) == 1


def test_plain_json_that_is_not_json_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(monkeypatch, [_bad_request(), "not json"])

    assert client.complete_json("sys", "user", _SCHEMA) is None
    assert len(completions.calls) == 2


def test_json_that_is_not_an_object_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _completions, _constructed = _client(monkeypatch, ["[1, 2]"])

    assert client.complete_json("sys", "user", _SCHEMA) is None


def test_connection_error_raises_endpoint_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    client, completions, _constructed = _client(
        monkeypatch, [openai.APIConnectionError(request=_request())]
    )

    with pytest.raises(EndpointUnreachable):
        client.complete_json("sys", "user", _SCHEMA)
    assert len(completions.calls) == 1


def test_timeout_raises_endpoint_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _completions, _constructed = _client(
        monkeypatch, [openai.APITimeoutError(request=_request())]
    )

    with pytest.raises(EndpointUnreachable):
        client.complete_json("sys", "user", _SCHEMA)
