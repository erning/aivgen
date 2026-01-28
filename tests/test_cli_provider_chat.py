from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typer.testing import CliRunner

import aivgen.cli as cli


runner = CliRunner()


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list[_Choice]


class _FakeProvider:
    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def chat_completions(
        self, *, model: str, messages: list[dict[str, str]], **kwargs: Any
    ) -> _Resp:  # noqa: ANN401,E501
        self.seen["model"] = model
        self.seen["messages"] = messages
        self.seen["kwargs"] = kwargs
        return _Resp(choices=[_Choice(message=_Msg(content="hello"))])


def test_provider_chat_joins_prompts_and_prints_content(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeProvider()

    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return object()

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = fake

        assert name == "zhipu"
        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(
        cli.app,
        [
            "provider",
            "chat",
            "--provider",
            "zhipu",
            "--model",
            "glm-4",
            "--prompt",
            "first",
            "--prompt",
            "second",
        ],
    )

    assert res.exit_code == 0, res.stdout
    assert res.stdout.strip() == "hello"
    assert fake.seen["model"] == "glm-4"
    assert fake.seen["messages"] == [{"role": "user", "content": "first\n\nsecond"}]
