from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from aivgen.providers.openai import ProviderError


@dataclass(frozen=True)
class OllamaProvider:
    """Native Ollama provider using official ollama Python SDK."""

    name: str
    host: str | None
    _client: Any

    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> OllamaProvider:
        host = config.get("host")

        if host is not None and (not isinstance(host, str) or not host):
            raise ProviderError(f"Provider {name!r}: host must be a string")

        sdk_module_name = str(config.get("_sdk", "ollama"))
        ollama_mod = importlib.import_module(sdk_module_name)

        # Get Client class from ollama module
        client_cls = cast(Any, getattr(ollama_mod, "Client", None))
        if client_cls is None:
            raise ProviderError(f"Provider {name!r}: ollama SDK missing Client class")

        # Create client with optional host
        client_kwargs: dict[str, Any] = {}
        if host:
            client_kwargs["host"] = host
        client = client_cls(**client_kwargs)

        return cls(name=name, host=host, _client=client)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Ollama format."""
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role not in ("user", "assistant", "system", "tool"):
                role = "user"

            if isinstance(content, str):
                out.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        continue
                text = "\n".join(text_parts) if text_parts else ""
                out.append({"role": role, "content": text})
            else:
                out.append({"role": role, "content": str(content)})

        return out

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        stream = bool(kwargs.get("stream", False))
        converted_messages = self._convert_messages(messages)
        options: dict[str, Any] = {}

        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]

        try:
            if stream:
                response = self._client.chat(
                    model=model,
                    messages=converted_messages,
                    stream=True,
                    options=options if options else None,
                )
                return OllamaStreamIterator(response)

            response = self._client.chat(
                model=model,
                messages=converted_messages,
                stream=False,
                options=options if options else None,
            )
            return self._convert_response(response)
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}") from e

    def _convert_response(self, response: Any) -> dict[str, Any]:
        """Convert Ollama response to OpenAI-compatible format."""
        message = getattr(response, "message", None)
        if message is None:
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }

        content = getattr(message, "content", "") or ""
        reasoning_content = getattr(message, "reasoning", None)

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def list_models(self) -> Any:
        """List available Ollama models."""
        try:
            models_response = self._client.list()
            models = getattr(models_response, "models", [])
            return {
                "data": [
                    {
                        "id": getattr(m, "model", str(m)),
                        "object": "model",
                        "owned_by": "ollama",
                    }
                    for m in models
                ]
            }
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}") from e


class OllamaStreamIterator:
    """Converts Ollama stream chunks to OpenAI-compatible format."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self._chunk_id = "ollama-chunk-0"
        self._iter: Iterator[Any] = iter(cast(Any, self._response))

    def __iter__(self) -> OllamaStreamIterator:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iter)
        except StopIteration:
            raise

        message = getattr(chunk, "message", None)
        if message is None:
            if getattr(chunk, "done", False):
                raise StopIteration
            return OllamaChunk(
                id=self._chunk_id,
                choices=[OllamaChoice(delta=OllamaDelta(), finish_reason=None)],
            )

        delta: dict[str, Any] = {}

        content = getattr(message, "content", "")
        if content:
            delta["content"] = content

        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            delta["reasoning_content"] = reasoning

        done = getattr(chunk, "done", False)
        finish_reason = "stop" if done else None

        return OllamaChunk(
            id=self._chunk_id,
            choices=[
                OllamaChoice(delta=OllamaDelta(**delta), finish_reason=finish_reason)
            ],
        )


@dataclass
class OllamaChunk:
    id: str
    choices: list[OllamaChoice]


@dataclass
class OllamaChoice:
    delta: OllamaDelta
    finish_reason: str | None


@dataclass
class OllamaDelta:
    content: str | None = None
    reasoning_content: str | None = None
