from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from aivgen.providers.openai import ProviderError


@dataclass(frozen=True)
class AnthropicProvider:
    name: str
    api_key: str
    base_url: str | None
    default_max_tokens: int
    thinking_enabled: bool
    thinking_budget: int
    _client: Any

    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> AnthropicProvider:
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        max_tokens = config.get("max_tokens", 2048)

        thinking_cfg = config.get("thinking")
        thinking_enabled = True
        thinking_budget = 1024

        if thinking_cfg is None:
            pass
        elif isinstance(thinking_cfg, Mapping):
            if "enabled" in thinking_cfg:
                thinking_enabled = bool(thinking_cfg.get("enabled", True))
            if "budget_tokens" in thinking_cfg:
                thinking_budget = thinking_cfg.get("budget_tokens", 1024)
        else:
            raise ProviderError(f"Provider {name!r}: thinking must be a mapping")

        if not isinstance(api_key, str) or not api_key:
            raise ProviderError(f"Provider {name!r}: missing or invalid api_key")
        if base_url is not None and (not isinstance(base_url, str) or not base_url):
            raise ProviderError(f"Provider {name!r}: base_url must be a string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ProviderError(f"Provider {name!r}: max_tokens must be a positive int")
        if not isinstance(thinking_budget, int) or thinking_budget <= 0:
            raise ProviderError(
                f"Provider {name!r}: thinking.budget_tokens must be a positive int"
            )
        if thinking_budget > max_tokens:
            raise ProviderError(
                f"Provider {name!r}: thinking.budget_tokens cannot exceed max_tokens"
            )

        sdk_module_name = str(config.get("_sdk", "anthropic"))
        anthropic_mod = importlib.import_module(sdk_module_name)
        client_attr = str(config.get("_client_class", "Anthropic"))
        client_cls = cast(Any, getattr(anthropic_mod, client_attr, None))
        if client_cls is None:
            raise ProviderError(f"Provider {name!r}: anthropic SDK missing Anthropic")
        client = client_cls(api_key=api_key, base_url=base_url or None)

        return cls(
            name=name,
            api_key=api_key,
            base_url=base_url,
            default_max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            _client=client,
        )

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    if content:
                        system_parts.append(content)
                elif isinstance(content, list):
                    text_bits: list[str] = []
                    for part in content:
                        if isinstance(part, Mapping) and part.get("type") == "text":
                            text_bits.append(str(part.get("text", "")))
                    if text_bits:
                        system_parts.append("".join(text_bits))
                else:
                    system_parts.append(str(content))
                continue

            if role not in ("user", "assistant"):
                role = "user"

            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") == "text":
                        blocks.append(
                            {"type": "text", "text": str(part.get("text", ""))}
                        )
                    elif part.get("type") == "image_url":
                        # MVP: skip images
                        continue
            else:
                blocks = [{"type": "text", "text": str(content)}]

            out.append({"role": role, "content": blocks})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, out

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: Mapping[str, str] | None = None,  # ignored
        **kwargs: Any,
    ) -> Any:
        headers_items = list(headers.items()) if headers else []
        if headers_items:
            pass
        stream = bool(kwargs.get("stream", False))

        system, converted_messages = self._convert_messages(messages)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ProviderError("Anthropic error: max_tokens must be a positive int")
        if self.thinking_enabled and self.thinking_budget > max_tokens:
            raise ProviderError(
                "Anthropic error: thinking budget cannot exceed max_tokens"
            )

        thinking: dict[str, Any] | None = None
        if self.thinking_enabled:
            thinking = {"type": "enabled", "budget_tokens": self.thinking_budget}

        try:
            if stream:
                response = self._client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=converted_messages,
                    thinking=thinking,
                )
                return AnthropicStreamIterator(response)

            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=converted_messages,
                thinking=thinking,
            )
            return self._convert_response(response)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Anthropic error: {e}") from e

    def _convert_response(self, response: Any) -> dict[str, Any]:
        content = ""
        reasoning_content = ""

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content += getattr(block, "text", "")
            elif block_type == "thinking":
                reasoning_content += getattr(block, "thinking", "")

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content or None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def list_models(self) -> Any:
        return {
            "data": [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "object": "model",
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-3-7-sonnet-20250219",
                    "object": "model",
                    "owned_by": "anthropic",
                },
            ]
        }


class AnthropicStreamIterator:
    """Converts Anthropic stream events to OpenAI-compatible chunks."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self._chunk_id = "anthropic-chunk-0"
        self._current_content = ""
        self._current_thinking = ""

        self._stream: Any
        self._iter: Iterator[Any]
        self._closed = False

        self._stream = self._response.__enter__()
        self._iter = iter(cast(Any, self._stream))

    def __iter__(self) -> AnthropicStreamIterator:
        return self

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def __next__(self) -> Any:
        while True:
            try:
                event = next(self._iter)
            except StopIteration:
                self._close()
                raise

            if getattr(event, "type", None) == "message_stop":
                self._close()
                raise StopIteration

            if getattr(event, "type", None) == "content_block_start":
                continue
            if getattr(event, "type", None) == "content_block_stop":
                continue
            if getattr(event, "type", None) != "content_block_delta":
                continue

            delta_obj = getattr(event, "delta", None)
            delta: dict[str, Any] = {}

            if getattr(delta_obj, "type", None) == "text_delta":
                text = getattr(delta_obj, "text", "")
                if text:
                    self._current_content += text
                    delta["content"] = text
            elif getattr(delta_obj, "type", None) == "thinking_delta":
                thinking = getattr(delta_obj, "thinking", "")
                if thinking:
                    self._current_thinking += thinking
                    delta["reasoning_content"] = thinking
            else:
                continue

            return AnthropicChunk(
                id=self._chunk_id,
                choices=[
                    AnthropicChoice(
                        delta=AnthropicDelta(**delta),
                        finish_reason=None,
                    )
                ],
            )


@dataclass
class AnthropicChunk:
    id: str
    choices: list[AnthropicChoice]


@dataclass
class AnthropicChoice:
    delta: AnthropicDelta
    finish_reason: str | None


@dataclass
class AnthropicDelta:
    content: str | None = None
    reasoning_content: str | None = None
