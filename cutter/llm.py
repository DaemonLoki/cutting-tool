"""JSON chat completions against a local OpenAI-compatible model."""

from __future__ import annotations

import json
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, OpenAI

_PLAIN_JSON = "reply with a single JSON object matching the schema, no markdown."


class LlmError(Exception):
    """The local model request failed."""


class EndpointUnreachable(LlmError):
    """The judge endpoint could not be reached."""


class JsonClient(Protocol):
    """One JSON completion from the local judge model."""

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return a JSON object, or None when the reply is not one."""


class OpenAIJsonClient:
    """Chat completion that asks for one JSON object.

    The API key is a placeholder for LM Studio, which does not check it.
    """

    def __init__(self, *, base_url: str, model: str, timeout_s: float) -> None:
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout_s)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the parsed object. A rejected schema is retried once as plain JSON."""
        try:
            content = self._create(system, user, schema)
        except BadRequestError:
            content = self._create(system, _plain_user(user), None)
        except APIStatusError as exc:
            if exc.status_code != 400:
                raise
            content = self._create(system, _plain_user(user), None)
        return _parse_json_object(content)

    def _create(self, system: str, user: str, schema: dict[str, Any] | None) -> str | None:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "retake_verdict",
                    "strict": True,
                    "schema": schema,
                },
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            raise EndpointUnreachable(str(exc)) from exc
        content = response.choices[0].message.content
        if content is None:
            return None
        if not isinstance(content, str):
            return None
        return content


def _plain_user(user: str) -> str:
    return f"{user}\n{_PLAIN_JSON}"


def _parse_json_object(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value
