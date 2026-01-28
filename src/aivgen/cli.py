from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console

from aivgen.config import ConfigError, load_config
from aivgen.providers.openai_compatible import ProviderError
from aivgen.providers.registry import build_provider


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


@provider_app.command("chat")
def provider_chat(
    provider: str = typer.Option(..., "--provider", help="Provider name."),
    model: str = typer.Option(..., "--model", help="Model name (passed at call time)."),
    prompt: list[str] = typer.Option(
        ...,
        "--prompt",
        help="Prompt text. Repeat the flag to append multiple prompts.",
    ),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
        ref = build_provider(cfg, name=provider)
    except (ConfigError, ProviderError) as e:
        raise typer.Exit(code=_print_error(str(e)))

    text = "\n\n".join(prompt)
    messages = [{"role": "user", "content": text}]

    try:
        resp = ref.provider.chat_completions(model=model, messages=messages)
    except Exception as e:  # noqa: BLE001
        raise typer.Exit(code=_print_error(str(e)))

    content = _extract_chat_content(resp)
    typer.echo(content)


@provider_app.command("models")
def provider_models(
    provider: str = typer.Option(..., "--provider", help="Provider name."),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
        ref = build_provider(cfg, name=provider)
    except (ConfigError, ProviderError) as e:
        raise typer.Exit(code=_print_error(str(e)))

    list_models = getattr(ref.provider, "list_models", None)
    if list_models is None:
        raise typer.Exit(
            code=_print_error(f"Provider {provider!r} does not support models listing")
        )

    try:
        models = list_models()
    except Exception as e:  # noqa: BLE001
        raise typer.Exit(code=_print_error(str(e)))

    for mid in _extract_model_ids(models):
        typer.echo(mid)


def _extract_model_ids(models: object) -> list[str]:
    data = getattr(models, "data", None)
    items: Any = data if data is not None else models

    out: list[str] = []
    try:
        iterator = iter(items)
    except TypeError:
        iterator = iter(())

    for item in iterator:
        mid = getattr(item, "id", None)
        if isinstance(mid, str) and mid:
            out.append(mid)
            continue
        if isinstance(item, dict):
            mid2 = item.get("id")
            if isinstance(mid2, str) and mid2:
                out.append(mid2)
                continue

    if not out:
        raise ValueError("Unexpected provider response shape: missing model ids")
    return out


def _extract_chat_content(resp: object) -> str:
    # OpenAI SDK returns a typed object. Keep this function forgiving so
    # provider implementations can vary while CLI remains stable.
    try:
        choices = getattr(resp, "choices")
        choice0 = choices[0]
        msg = getattr(choice0, "message")
        content = getattr(msg, "content")
        if isinstance(content, str):
            return content
    except Exception:  # noqa: BLE001
        pass

    if isinstance(resp, dict):
        try:
            content = resp["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
        except Exception:  # noqa: BLE001
            pass

    raise ValueError(
        "Unexpected provider response shape: missing choices[0].message.content"
    )


def _print_error(message: str) -> int:
    console.print(json.dumps({"ok": False, "error": message}, indent=2, sort_keys=True))
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
