from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from aivgen.providers.openai import OpenAIProvider, ProviderError


@dataclass
class _FakeCompletions:
    seen: dict[str, Any]

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.seen["model"] = model
        self.seen["messages"] = messages
        self.seen["extra_headers"] = dict(extra_headers or {})
        self.seen["kwargs"] = dict(kwargs)
        return {"ok": True}


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeClient:
    chat: _FakeChat


def test_from_config_requires_base_url_and_api_key() -> None:
    with pytest.raises(ProviderError):
        OpenAIProvider.from_config(name="zhipu", config={})


def test_headers_validation() -> None:
    with pytest.raises(ProviderError):
        OpenAIProvider.from_config(
            name="zhipu",
            config={"base_url": "https://x", "api_key": "k", "headers": []},
        )


def test_call_merges_per_request_headers_over_provider_headers() -> None:
    seen: dict[str, Any] = {}
    fake_client = _FakeClient(chat=_FakeChat(completions=_FakeCompletions(seen=seen)))

    provider = OpenAIProvider(
        name="zhipu",
        base_url="https://example",
        api_key="secret",
        headers={"X-A": "1", "X-B": "2"},
        _client=fake_client,
    )

    resp = provider.chat_completions(
        model="glm-4",
        messages=[{"role": "user", "content": "hi"}],
        headers={"X-B": "override", "X-C": "3"},
        temperature=0.1,
    )
    assert resp == {"ok": True}
    assert seen["model"] == "glm-4"
    assert seen["messages"][0]["content"] == "hi"
    assert seen["extra_headers"] == {"X-B": "override", "X-C": "3"}
    assert seen["kwargs"]["temperature"] == 0.1
