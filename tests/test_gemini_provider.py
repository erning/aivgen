from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aivgen.providers.gemini import GeminiProvider, _parse_data_url
from aivgen.providers.openai_compatible import ProviderError


def test_parse_data_url_base64() -> None:
    import base64

    image_data = b"fake-image-data"
    b64_data = base64.b64encode(image_data).decode()
    url = f"data:image/jpeg;base64,{b64_data}"

    mime_type, decoded = _parse_data_url(url)

    assert mime_type == "image/jpeg"
    assert decoded == image_data


def test_parse_data_url_no_mime_type() -> None:
    import base64

    text_data = b"hello world"
    b64_data = base64.b64encode(text_data).decode()
    url = f"data:;base64,{b64_data}"

    mime_type, decoded = _parse_data_url(url)

    assert mime_type == "application/octet-stream"
    assert decoded == text_data


def test_from_config_requires_api_key() -> None:
    with pytest.raises(ProviderError, match="missing or invalid api_key"):
        GeminiProvider.from_config(name="gemini", config={})


def test_from_config_with_valid_api_key() -> None:
    with patch("aivgen.providers.gemini.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        provider = GeminiProvider.from_config(
            name="gemini", config={"api_key": "test-key"}
        )

        assert provider.name == "gemini"
        assert provider.api_key == "test-key"
        mock_client.assert_called_once_with(api_key="test-key")


def test_convert_messages_text_only() -> None:
    with patch("aivgen.providers.gemini.genai.Client"):
        provider = GeminiProvider(name="gemini", api_key="test", _client=MagicMock())

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        contents = provider._convert_messages(messages)

        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"  # assistant mapped to model


def test_convert_response_with_thinking() -> None:
    with patch("aivgen.providers.gemini.genai.Client"):
        provider = GeminiProvider(name="gemini", api_key="test", _client=MagicMock())

        # Mock response with thinking parts
        mock_part_thought = MagicMock()
        mock_part_thought.thought = True
        mock_part_thought.text = "Let me think about this..."

        mock_part_response = MagicMock()
        mock_part_response.thought = False
        mock_part_response.text = "The answer is 42."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part_thought, mock_part_response]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = provider._convert_response(mock_response)

        assert result["choices"][0]["message"]["content"] == "The answer is 42."
        assert (
            result["choices"][0]["message"]["reasoning_content"]
            == "Let me think about this..."
        )


def test_convert_response_without_thinking() -> None:
    with patch("aivgen.providers.gemini.genai.Client"):
        provider = GeminiProvider(name="gemini", api_key="test", _client=MagicMock())

        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.text = "Hello!"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = provider._convert_response(mock_response)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["message"]["reasoning_content"] is None


def test_list_models() -> None:
    with patch("aivgen.providers.gemini.genai.Client"):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.name = "gemini-2.5-pro"
        mock_client.models.list.return_value = [mock_model]

        provider = GeminiProvider(name="gemini", api_key="test", _client=mock_client)

        result = provider.list_models()

        assert result["data"][0]["id"] == "gemini-2.5-pro"
        assert result["data"][0]["owned_by"] == "google"
