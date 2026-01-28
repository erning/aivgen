from __future__ import annotations

import json

import typer
from rich.console import Console

from aivgen.config import ConfigError, load_config


app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
provider_app = typer.Typer(no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(provider_app, name="provider")

console = Console()


@config_app.command("show")
def config_show(
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Path to a YAML config file (overrides auto-discovery).",
    ),
    redact: bool = typer.Option(
        True, "--redact/--no-redact", help="Redact secrets in output."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
    except ConfigError as e:
        raise typer.Exit(code=_print_error(str(e)))

    if as_json:
        typer.echo(cfg.to_json(redact_secrets=redact))
        return

    data = cfg.to_dict(redact_secrets=redact)
    console.print(json.dumps(data, indent=2, sort_keys=True))


@provider_app.command("list")
def provider_list(
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
    except ConfigError as e:
        raise typer.Exit(code=_print_error(str(e)))

    for name in sorted(cfg.aivgen.providers.keys()):
        typer.echo(name)


def _print_error(message: str) -> int:
    console.print(json.dumps({"ok": False, "error": message}, indent=2, sort_keys=True))
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
