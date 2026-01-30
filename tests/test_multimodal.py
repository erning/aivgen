from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

from aivgen.providers.anthropic import AnthropicProvider
from aivgen.providers.errors import ProviderRequestError
from aivgen.providers.gemini import GeminiProvider
from aivgen.providers.ollama import OllamaProvider


def test_anthropic_converts_data_url_image_to_image_block() -> None:
    raw = b"img-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    provider = AnthropicProvider(
        name="claude",
        api_key="k",
        base_url=None,
        default_max_tokens=8,
        thinking_enabled=False,
        thinking_budget=1,
        _client=object(),
    )

    system, out = provider._convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    )

    assert system is None
    assert out[0]["role"] == "user"
    blocks = out[0]["content"]
    assert any(b.get("type") == "image" for b in blocks)
    img = next(b for b in blocks if b.get("type") == "image")
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"


def test_ollama_converts_data_url_image_to_images_bytes() -> None:
    raw = b"img-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    provider = OllamaProvider(name="ollama", host=None, _client=object())
    out = provider._convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    )

    assert out[0]["role"] == "user"
    assert out[0]["images"] == [raw]


def test_ollama_errors_on_remote_image_url() -> None:
    provider = OllamaProvider(name="ollama", host=None, _client=object())

    with pytest.raises(ProviderRequestError, match="remote URLs"):
        provider._convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}}
                    ],
                }
            ]
        )


def test_gemini_downloads_generic_https_images(monkeypatch) -> None:  # noqa: ANN001
    provider = GeminiProvider(name="gemini", api_key="k", _client=MagicMock())

    called = {"ok": False}

    def fake_fetch(url: str) -> tuple[str, bytes]:
        called["ok"] = True
        return "image/jpeg", b"img"

    monkeypatch.setattr("aivgen.providers.gemini._fetch_https_image", fake_fetch)

    provider._convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/i.jpg"},
                    }
                ],
            }
        ]
    )

    assert called["ok"] is True


def test_gemini_does_not_download_files_api_uri(monkeypatch) -> None:  # noqa: ANN001
    provider = GeminiProvider(name="gemini", api_key="k", _client=MagicMock())

    def fail_fetch(url: str) -> tuple[str, bytes]:
        raise AssertionError("_fetch_https_image should not be called")

    monkeypatch.setattr("aivgen.providers.gemini._fetch_https_image", fail_fetch)

    provider._convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://generativelanguage.googleapis.com/v1beta/files/abc"
                        },
                    }
                ],
            }
        ]
    )
