from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from typer.testing import CliRunner

import aivgen.cli as cli
from aivgen.config import AivConfig, AivGenConfig, ProviderConfig, ScriptConfig

runner = CliRunner()


class _FakeProvider:
    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def chat_completions(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.seen["model"] = model
        self.seen["messages"] = messages
        self.seen["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "script result"}}]}


def test_script_requires_image(monkeypatch) -> None:  # noqa: ANN001
    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return AivConfig(
            aivgen=AivGenConfig(providers={}),
            loaded_from=(),
        )

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    res = runner.invoke(cli.app, ["script", "--provider", "zhipu", "--model", "glm-4"])

    assert res.exit_code != 0


def test_script_uses_config_defaults(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fake = _FakeProvider()

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake-jpeg")

    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return AivConfig(
            aivgen=AivGenConfig(
                providers={
                    "zhipu": ProviderConfig(
                        data={
                            "type": "openai-compatible",
                            "base_url": "...",
                            "api_key": "...",
                        }
                    )
                },
                script=ScriptConfig(
                    provider="zhipu",
                    model="glm-4v",
                    system_prompt=["You are a script writer."],
                    prompt=["Write a script for this image."],
                ),
            ),
            loaded_from=(),
        )

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = fake

        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(cli.app, ["script", "--image", str(img), "--no-stream"])

    assert res.exit_code == 0, res.stdout
    assert res.stdout.strip() == "script result"
    assert fake.seen["model"] == "glm-4v"
    messages = fake.seen["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a script writer."
    assert messages[1]["role"] == "user"


def test_script_cli_overrides_config(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fake = _FakeProvider()

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake-jpeg")

    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return AivConfig(
            aivgen=AivGenConfig(
                providers={
                    "zhipu": ProviderConfig(data={}),
                    "gemini": ProviderConfig(data={}),
                },
                script=ScriptConfig(
                    provider="zhipu",
                    model="glm-4v",
                ),
            ),
            loaded_from=(),
        )

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = fake

        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(
        cli.app,
        [
            "script",
            "--image",
            str(img),
            "--provider",
            "gemini",
            "--model",
            "gemini-pro",
        ],
    )

    assert res.exit_code == 0, res.stdout
    assert fake.seen["model"] == "gemini-pro"


def test_script_merges_prompts(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fake = _FakeProvider()

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake-jpeg")

    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return AivConfig(
            aivgen=AivGenConfig(
                providers={"zhipu": ProviderConfig(data={})},
                script=ScriptConfig(
                    provider="zhipu",
                    model="glm-4v",
                    system_prompt=["Config system prompt."],
                    prompt=["Config prompt."],
                ),
            ),
            loaded_from=(),
        )

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = fake

        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(
        cli.app,
        [
            "script",
            "--image",
            str(img),
            "--system-prompt",
            "CLI system prompt.",
            "--prompt",
            "CLI prompt.",
        ],
    )

    assert res.exit_code == 0, res.stdout
    messages = fake.seen["messages"]
    assert messages[0]["content"] == "CLI system prompt.\n\nConfig system prompt."
    user_content = messages[1]["content"]
    assert user_content[0]["text"] == "CLI prompt.\n\nConfig prompt."


def test_script_writes_to_file(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fake = _FakeProvider()

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake-jpeg")
    output_file = tmp_path / "output.txt"

    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return AivConfig(
            aivgen=AivGenConfig(
                providers={"zhipu": ProviderConfig(data={})},
                script=ScriptConfig(provider="zhipu", model="glm-4v"),
            ),
            loaded_from=(),
        )

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = fake

        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(
        cli.app,
        ["script", "--image", str(img), "--output", str(output_file), "--no-stream"],
    )

    assert res.exit_code == 0, res.stdout
    assert output_file.read_text() == "script result"
    assert res.stdout.strip() == ""
