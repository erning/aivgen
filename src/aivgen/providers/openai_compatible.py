from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from openai import DefaultHttpxClient, OpenAI


class ProviderError(RuntimeError):
    pass


class _ChatCompletionsProtocol(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401,E501
        ...


class _ChatProtocol(Protocol):
    completions: _ChatCompletionsProtocol


class _OpenAIClientProtocol(Protocol):
    chat: _ChatProtocol


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    name: str
    base_url: str
    api_key: str
    headers: dict[str, str]
    _client: _OpenAIClientProtocol

    @classmethod
    def from_config(
        cls, *, name: str, config: Mapping[str, Any]
    ) -> "OpenAICompatibleProvider":
        base_url = config.get("base_url")
        api_key = config.get("api_key")
        headers_raw = config.get("headers")

        if not isinstance(base_url, str) or not base_url:
            raise ProviderError(f"Provider {name!r}: missing or invalid base_url")
        if not isinstance(api_key, str) or not api_key:
            raise ProviderError(f"Provider {name!r}: missing or invalid api_key")

        headers: dict[str, str] = {}
        if headers_raw is None:
            headers = {}
        elif isinstance(headers_raw, dict):
            for k, v in headers_raw.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ProviderError(
                        f"Provider {name!r}: headers must be string:string"
                    )
                headers[k] = v
        else:
            raise ProviderError(f"Provider {name!r}: headers must be a mapping")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(headers=headers),
        )

        return cls(
            name=name,
            base_url=base_url,
            api_key=api_key,
            headers=headers,
            _client=client,
        )

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        extra_headers = dict(headers or {})
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=extra_headers if extra_headers else None,
            **kwargs,
        )
