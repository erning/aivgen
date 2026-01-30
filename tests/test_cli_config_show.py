from __future__ import annotations

import aivgen.cli as cli


def test_config_show_warns_on_unresolved_env_vars(
    tmp_path, monkeypatch, capsys
) -> None:  # noqa: ANN001
    monkeypatch.delenv("MISSING_VAR", raising=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
aivgen:
  providers:
    zhipu:
      type: openai
      base_url: https://example.invalid/v1
      api_key: ${MISSING_VAR}
""".lstrip(),
        encoding="utf-8",
    )

    cli.config_show(config_path=str(cfg_path), redact=True, as_json=True)

    out = capsys.readouterr()
    assert "MISSING_VAR" in out.err
