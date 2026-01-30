from __future__ import annotations

from pathlib import Path

import pytest

from aivgen.config import AivConfig, AivGenConfig, ScriptConfig, load_config


def test_load_config_allows_missing_api_key_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPEN_API_KEY", raising=False)

    # Isolate away from repo's real aivgen.yaml.
    cfg = load_config(cwd=Path("__nonexistent__"))
    assert cfg.aivgen.providers == {}


def test_precedence_config_over_project_over_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_cfg = tmp_path / ".config" / "aivgen" / "aivgen.yaml"
    global_cfg.parent.mkdir(parents=True, exist_ok=True)
    global_cfg.write_text(
        """
aivgen:
  providers:
    openai:
      base_url: https://global.example/v1
      apk_key: sk-global
""".lstrip(),
        encoding="utf-8",
    )

    project_cfg = tmp_path / "aivgen.yaml"
    project_cfg.write_text(
        """
aivgen:
  providers:
    openai:
      base_url: https://project.example/v1
      apk_key: sk-project
""".lstrip(),
        encoding="utf-8",
    )

    explicit_cfg = tmp_path / "explicit.yaml"
    explicit_cfg.write_text(
        """
aivgen:
  providers:
    openai:
      base_url: https://explicit.example/v1
      apk_key: sk-explicit
""".lstrip(),
        encoding="utf-8",
    )

    cfg1 = load_config(cwd=tmp_path)
    assert (
        cfg1.aivgen.providers["openai"].data["base_url"] == "https://project.example/v1"
    )

    cfg2 = load_config(cwd=tmp_path, config_path=str(explicit_cfg))
    assert (
        cfg2.aivgen.providers["openai"].data["base_url"]
        == "https://explicit.example/v1"
    )


def test_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_API_KEY", "sk-test-1234567890")

    # Ensure we don't read the repo's aivgen.yaml in this unit test.
    cfg = load_config(cwd=Path("__nonexistent__"))
    d = cfg.to_dict(redact_secrets=True)
    assert d["aivgen"]["providers"] == {}


def test_redaction_with_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_API_KEY", "sk-test-1234567890")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
aivgen:
  providers:
    zhipu:
      apk_key: "${OPEN_API_KEY}"
      base_url: https://example.invalid
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(config_path=str(cfg_path), cwd=tmp_path)
    d = cfg.to_dict(redact_secrets=True)
    assert d["aivgen"]["providers"]["zhipu"]["apk_key"] != "sk-test-1234567890"
    assert "..." in d["aivgen"]["providers"]["zhipu"]["apk_key"]


def test_to_dict_includes_script() -> None:
    cfg = AivConfig(
        aivgen=AivGenConfig(
            providers={},
            script=ScriptConfig(
                provider="zhipu",
                model="glm-4.6v",
                system_prompt=["@prompts/base.md"],
                prompt=["Write a script"],
            ),
        ),
        loaded_from=(),
    )

    d = cfg.to_dict(redact_secrets=True)
    assert d["aivgen"]["script"]["provider"] == "zhipu"
    assert d["aivgen"]["script"]["model"] == "glm-4.6v"
    assert d["aivgen"]["script"]["system-prompt"] == ["@prompts/base.md"]
    assert d["aivgen"]["script"]["prompt"] == ["Write a script"]
