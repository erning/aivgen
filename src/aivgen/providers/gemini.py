from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from google import genai
from google.genai import types

from aivgen.providers.contracts import ProviderCapabilities
from aivgen.providers.errors import ProviderConfigError, ProviderRequestError
from aivgen.providers.images import parse_data_url


def _parse_data_url(url: str) -> tuple[str, bytes]:
    """Parse a data URL and return (mime_type, decoded_bytes)."""
    return parse_data_url(url)


def _fetch_https_image(url: str) -> tuple[str, bytes]:
    import mimetypes
    import urllib.request

    with urllib.request.urlopen(url) as resp:
        data = resp.read()
        mime_type = ""
        try:
            mime_type = resp.headers.get_content_type()
        except Exception:  # noqa: BLE001
            mime_type = ""

    if not mime_type or mime_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        mime_type = guessed or mime_type or "application/octet-stream"

    if not mime_type.startswith("image/"):
        raise ValueError(f"URL does not look like an image (Content-Type: {mime_type})")

    return mime_type, data


@dataclass(frozen=True)
class GeminiProvider:
    """Native Gemini provider using google-genai SDK.

    Supports thinking/reasoning content for Gemini 2.5+ and 3.0+ models.
    """

    name: str
    api_key: str
    _client: Any

    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        supports_images=True,
        supports_tools=False,
        supports_reasoning_stream=True,
        supports_model_list=True,
        supports_stream=True,
    )

    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> GeminiProvider:
        api_key = config.get("api_key")

        if not isinstance(api_key, str) or not api_key:
            raise ProviderConfigError(f"Provider {name!r}: missing or invalid api_key")

        client = genai.Client(api_key=api_key)

        return cls(name=name, api_key=api_key, _client=client)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[types.Content]:
        """Convert OpenAI-style messages to Gemini Content format."""
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map OpenAI roles to Gemini roles
            # Gemini uses: "user", "model" (assistant is mapped to model)
            gemini_role = "model" if role in ("assistant", "model") else "user"

            if isinstance(content, str):
                parts = [types.Part(text=content)]
            elif isinstance(content, list):
                # Handle multimodal content (text + images)
                parts: list[types.Part] = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(types.Part(text=part.get("text", "")))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:"):
                            # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                            mime_type, b64_data = _parse_data_url(url)
                            parts.append(
                                types.Part.from_bytes(
                                    data=b64_data, mime_type=mime_type
                                )
                            )
                        elif url.startswith("gs://"):
                            parts.append(types.Part.from_uri(file_uri=url))
                        elif url.startswith(
                            "https://generativelanguage.googleapis.com/"
                        ):
                            mime_hint = image_url.get("mime_type")
                            mime_type = (
                                str(mime_hint)
                                if isinstance(mime_hint, str) and mime_hint
                                else "application/octet-stream"
                            )
                            parts.append(
                                types.Part.from_uri(file_uri=url, mime_type=mime_type)
                            )
                        elif url.startswith("http://") or url.startswith("https://"):
                            try:
                                mime_type, data = _fetch_https_image(url)
                            except Exception as e:  # noqa: BLE001
                                raise ProviderRequestError(
                                    f"Gemini error: failed to fetch image URL: {e}",
                                    retryable=False,
                                    cause=e,
                                ) from e
                            parts.append(
                                types.Part.from_bytes(data=data, mime_type=mime_type)
                            )
            else:
                parts = [types.Part(text=str(content))]

            contents.append(types.Content(role=gemini_role, parts=parts))

        return contents

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate chat completion with streaming support and thinking content."""
        stream = kwargs.get("stream", False)
        contents = self._convert_messages(messages)

        # Build config
        config_kwargs: dict[str, Any] = {}

        # Handle temperature
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs["temperature"]

        # Handle max_tokens
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_tokens"]

        # Handle thinking/reasoning for Gemini 2.5+ and 3.0+
        # Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget
        # include_thoughts=True is required to get thinking content in response
        if "gemini-3" in model:
            thinking_config = kwargs.get("thinking_config", {})
            level = thinking_config.get("thinking_level", "MEDIUM")
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=getattr(
                    types.ThinkingLevel, level, types.ThinkingLevel.MEDIUM
                ),
                include_thoughts=True,
            )
        elif "gemini-2.5" in model:
            thinking_config = kwargs.get("thinking_config", {})
            budget = thinking_config.get("thinking_budget", 1024)
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=True,
            )

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if stream:
            return self._stream_response(model, contents, config)

        # Non-streaming response
        response = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )

        # Convert to OpenAI-compatible response format
        return self._convert_response(response)

    def _stream_response(
        self,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig | None,
    ) -> Any:
        """Stream response with thinking content support."""
        response = self._client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        # Return a generator that yields OpenAI-compatible chunks
        return GeminiStreamIterator(response)

    def _convert_response(self, response: Any) -> dict[str, Any]:
        """Convert Gemini response to OpenAI-compatible format."""
        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }

        content_parts = []
        reasoning_content = ""

        for part in candidate.content.parts:
            if getattr(part, "thought", False):
                reasoning_content += part.text
            else:
                content_parts.append(part.text)

        content = "".join(content_parts)

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content
                        if reasoning_content
                        else None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def list_models(self) -> Any:
        """List available Gemini models."""
        models = self._client.models.list()
        # Convert to OpenAI-compatible format
        return {
            "data": [
                {"id": model.name, "object": "model", "owned_by": "google"}
                for model in models
            ]
        }


class GeminiStreamIterator:
    """Iterator that converts Gemini stream chunks to OpenAI-compatible format."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self._chunk_id = "gemini-chunk-0"

    def __iter__(self) -> GeminiStreamIterator:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._response)
        except StopIteration:
            raise

        # Extract content and thinking from chunk
        content = ""
        reasoning_content = ""

        if chunk.candidates and chunk.candidates[0].content.parts:
            for part in chunk.candidates[0].content.parts:
                # Check if this is a thinking part
                is_thought = getattr(part, "thought", False)
                if is_thought:
                    reasoning_content += part.text
                else:
                    content += part.text

        # Build OpenAI-compatible chunk
        delta: dict[str, Any] = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content

        return GeminiChunk(
            id=self._chunk_id,
            choices=[
                GeminiChoice(
                    delta=GeminiDelta(**delta),
                    finish_reason=None,
                )
            ],
        )


@dataclass
class GeminiChunk:
    """OpenAI-compatible chunk structure."""

    id: str
    choices: list[GeminiChoice]


@dataclass
class GeminiChoice:
    """OpenAI-compatible choice structure."""

    delta: GeminiDelta
    finish_reason: str | None


@dataclass
class GeminiDelta:
    """OpenAI-compatible delta structure."""

    content: str | None = None
    reasoning_content: str | None = None
