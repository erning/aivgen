from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from aivgen.providers.contracts import ProviderCapabilities
from aivgen.providers.errors import ProviderConfigError, ProviderRequestError
from aivgen.providers.images import parse_data_url


@dataclass(frozen=True)
class OllamaProvider:
    """Native Ollama provider using official ollama Python SDK."""

    name: str
    host: str | None
    _client: Any

    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        supports_images=True,
        supports_tools=False,
        supports_reasoning_stream=True,
        supports_model_list=True,
        supports_stream=True,
    )

    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> OllamaProvider:
        host = config.get("host")

        if host is not None and (not isinstance(host, str) or not host):
            raise ProviderConfigError(f"Provider {name!r}: host must be a string")

        sdk_module_name = str(config.get("_sdk", "ollama"))
        ollama_mod = importlib.import_module(sdk_module_name)

        # Get Client class from ollama module
        client_cls = cast(Any, getattr(ollama_mod, "Client", None))
        if client_cls is None:
            raise ProviderConfigError(
                f"Provider {name!r}: ollama SDK missing Client class"
            )

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
                images: list[bytes] = []
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url")
                        if not isinstance(image_url, Mapping):
                            raise ProviderRequestError(
                                "Ollama error: invalid image_url payload",
                                retryable=False,
                            )
                        url = image_url.get("url")
                        if not isinstance(url, str) or not url:
                            raise ProviderRequestError(
                                "Ollama error: missing image_url.url",
                                retryable=False,
                            )

                        if url.startswith("data:"):
                            mime_type, data = parse_data_url(url)
                            if not mime_type.startswith("image/"):
                                msg = (
                                    "Ollama error: unsupported image type "
                                    f"{mime_type!r}"
                                )
                                raise ProviderRequestError(
                                    msg,
                                    retryable=False,
                                )
                            images.append(data)
                        elif url.startswith("http://") or url.startswith("https://"):
                            msg = (
                                "Ollama error: image_url.url must be a data URL "
                                "(remote URLs are not supported)"
                            )
                            raise ProviderRequestError(
                                msg,
                                retryable=False,
                            )
                        else:
                            raise ProviderRequestError(
                                "Ollama error: unsupported image_url.url format",
                                retryable=False,
                            )
                text = "\n".join(text_parts) if text_parts else ""
                msg_out: dict[str, Any] = {"role": role, "content": text}
                if images:
                    msg_out["images"] = images
                out.append(msg_out)
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
            chat_kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
            }
            if options:
                chat_kwargs["options"] = options

            # Enable thinking for models that support it (e.g., qwen3, deepseek-r1)
            chat_kwargs["think"] = True

            if stream:
                response = self._client.chat(stream=True, **chat_kwargs)
                return OllamaStreamIterator(response)

            response = self._client.chat(stream=False, **chat_kwargs)
            return self._convert_response(response)
        except Exception as e:  # noqa: BLE001
            raise ProviderRequestError(
                f"Ollama error: {e}",
                retryable=False,
                cause=e,
            ) from e

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
        reasoning_content = getattr(message, "thinking", None)

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
        except Exception as e:  # noqa: BLE001
            raise ProviderRequestError(
                f"Ollama error: {e}",
                retryable=False,
                cause=e,
            ) from e


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

        thinking = getattr(message, "thinking", None)
        if thinking:
            delta["reasoning_content"] = thinking

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
