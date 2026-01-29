from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

import aivgen.cli as cli

runner = CliRunner()


@dataclass
class _Model:
    id: str


@dataclass
class _ModelsResp:
    data: list[_Model]


class _FakeProvider:
    def list_models(self) -> _ModelsResp:
        return _ModelsResp(data=[_Model(id="m1"), _Model(id="m2")])


def test_provider_models_prints_ids_in_order(monkeypatch) -> None:  # noqa: ANN001
    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return object()

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = _FakeProvider()

        assert name == "zhipu"
        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(cli.app, ["models", "--provider", "zhipu"])
    assert res.exit_code == 0, res.stdout
    assert res.stdout.splitlines() == ["m1", "m2"]


def test_provider_models_errors_if_unsupported(monkeypatch) -> None:  # noqa: ANN001
    def fake_load_config(*, config_path=None, cwd=None):  # noqa: ANN001
        return object()

    def fake_build_provider(cfg, *, name: str):  # noqa: ANN001
        class _Ref:
            provider = object()

        return _Ref()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "build_provider", fake_build_provider)

    res = runner.invoke(cli.app, ["models", "--provider", "x"])
    assert res.exit_code != 0
    assert '"ok": false' in res.stdout
