from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aivgen.providers.anthropic import AnthropicStreamIterator


@dataclass
class _FakeDelta:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class _FakeEvent:
    type: str
    delta: Any | None = None


class _FakeResponse:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events
        self.exit_calls = 0

    def __enter__(self):  # noqa: ANN001
        return iter(self._events)

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.exit_calls += 1


def test_anthropic_stream_iterator_closes_on_exhaust() -> None:
    resp = _FakeResponse(
        [
            _FakeEvent(
                type="content_block_delta",
                delta=_FakeDelta(type="text_delta", text="he"),
            ),
            _FakeEvent(type="message_stop"),
        ]
    )

    it = AnthropicStreamIterator(resp)
    chunks = list(it)
    assert len(chunks) == 1
    assert resp.exit_calls == 1


def test_anthropic_stream_iterator_close_method() -> None:
    resp = _FakeResponse(
        [
            _FakeEvent(
                type="content_block_delta",
                delta=_FakeDelta(type="text_delta", text="he"),
            ),
            _FakeEvent(
                type="content_block_delta",
                delta=_FakeDelta(type="text_delta", text="llo"),
            ),
        ]
    )

    it = AnthropicStreamIterator(resp)
    next(it)
    it.close()
    assert resp.exit_calls == 1
