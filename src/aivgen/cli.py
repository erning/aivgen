from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from aivgen.config import ConfigError, load_config
from aivgen.providers.openai_compatible import ProviderError
from aivgen.providers.registry import build_provider

app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)

app.add_typer(config_app, name="config")

console = Console()
trace_console = Console(stderr=True)


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
        raise typer.Exit(code=_print_error(str(e))) from e

    if as_json:
        typer.echo(cfg.to_json(redact_secrets=redact))
        return

    data = cfg.to_dict(redact_secrets=redact)
    console.print(json.dumps(data, indent=2, sort_keys=True))


@app.command("providers")
def providers_list(
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
    except ConfigError as e:
        raise typer.Exit(code=_print_error(str(e))) from e

    for name in sorted(cfg.aivgen.providers.keys()):
        typer.echo(name)


@app.command("chat")
def chat(
    provider: str = typer.Option(..., "--provider", help="Provider name."),
    model: str = typer.Option(..., "--model", help="Model name (passed at call time)."),
    prompt: list[str] = typer.Option(
        ...,
        "--prompt",
        help="Prompt text. Repeat the flag to append multiple prompts.",
    ),
    system_prompt: list[str] = typer.Option(
        [],
        "--system-prompt",
        help=(
            "System prompt text. Supports @file and -. "
            "Repeat the flag to append multiple chunks."
        ),
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream output to stdout (default: stream).",
    ),
    reasoning: bool = typer.Option(
        True,
        "--reasoning/--no-reasoning",
        help="Print reasoning content to stderr if present (default: enabled).",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Print request/response trace to stderr (redacted).",
    ),
    image: list[str] = typer.Option(
        [],
        "--image",
        help=(
            "Image URL, data URL, or local file path. "
            "Repeat the flag for multiple images."
        ),
    ),
    image_detail: str = typer.Option(
        "auto",
        "--image-detail",
        help="Image detail hint: auto, low, or high.",
    ),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
        ref = build_provider(cfg, name=provider)
    except (ConfigError, ProviderError) as e:
        raise typer.Exit(code=_print_error(str(e))) from e

    try:
        stdin_state: dict[str, Any] = {"used": False, "text": ""}

        system_chunks = _resolve_prompts(system_prompt, stdin_state)
        prompt_chunks = _resolve_prompts(prompt, stdin_state)
        system_text = "\n\n".join(system_chunks).strip()
        text = "\n\n".join(prompt_chunks)
    except Exception as e:  # noqa: BLE001
        raise typer.Exit(code=_print_error(str(e))) from e

    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append(
        {
            "role": "user",
            "content": _build_user_content(
                text=text, images=image, image_detail=image_detail
            ),
        }
    )

    if trace:
        _trace_request(provider=provider, model=model, messages=messages)

    if not stream:
        try:
            resp = ref.provider.chat_completions(model=model, messages=messages)
        except Exception as e:  # noqa: BLE001
            raise typer.Exit(code=_print_error(str(e))) from e

        content = _extract_chat_content(resp)
        typer.echo(content)
        return

    try:
        resp = ref.provider.chat_completions(
            model=model, messages=messages, stream=True
        )
    except Exception as e:  # noqa: BLE001
        raise typer.Exit(code=_print_error(str(e))) from e

    _stream_chat_response(resp, trace=trace, reasoning=reasoning)


@app.command("models")
def models(
    provider: str = typer.Option(..., "--provider", help="Provider name."),
    config_path: str | None = typer.Option(
        None, "--config", help="Path to a YAML config file."
    ),
) -> None:
    try:
        cfg = load_config(config_path=config_path)
        ref = build_provider(cfg, name=provider)
    except (ConfigError, ProviderError) as e:
        raise typer.Exit(code=_print_error(str(e))) from e

    list_models = getattr(ref.provider, "list_models", None)
    if list_models is None:
        raise typer.Exit(
            code=_print_error(f"Provider {provider!r} does not support models listing")
        )

    try:
        models = list_models()
    except Exception as e:  # noqa: BLE001
        raise typer.Exit(code=_print_error(str(e))) from e

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
    if hasattr(resp, "choices"):
        try:
            r = cast(Any, resp)
            choice0 = r.choices[0]
            content = choice0.message.content
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


def _stream_chat_response(resp: object, *, trace: bool, reasoning: bool) -> None:
    try:
        iterator = iter(cast(Any, resp))
    except TypeError as e:
        raise TypeError("Provider did not return a stream iterator") from e

    saw_reasoning = False
    last_finish: str | None = None
    last_id: str | None = None

    for chunk in iterator:
        if not getattr(chunk, "choices", None):
            continue

        choice0 = chunk.choices[0]
        delta = getattr(choice0, "delta", None)
        if delta is None:
            continue

        last_id = getattr(chunk, "id", last_id)
        finish = getattr(choice0, "finish_reason", None)
        if isinstance(finish, str) and finish:
            last_finish = finish

        if reasoning:
            r = getattr(delta, "reasoning_content", None)
            if isinstance(r, str) and r:
                if not saw_reasoning:
                    _stderr_write("\n[thinking]\n", dim=True)
                    saw_reasoning = True
                _stderr_write(r, dim=True)

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            sys.stdout.write(content)
            sys.stdout.flush()

    if saw_reasoning:
        _stderr_write("\n", dim=True)

    sys.stdout.write("\n")
    sys.stdout.flush()

    if trace:
        _trace_response_summary(finish_reason=last_finish, response_id=last_id)


def _trace_request(
    *, provider: str, model: str, messages: list[dict[str, Any]]
) -> None:
    payload = {
        "event": "request",
        "provider": provider,
        "model": model,
        "messages": _sanitize_messages(messages),
    }
    _stderr_write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", dim=True)


def _trace_response_summary(
    *, finish_reason: str | None, response_id: str | None
) -> None:
    payload = {
        "event": "response.done",
        "id": response_id,
        "finish_reason": finish_reason,
    }
    _stderr_write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", dim=True)


def _stderr_should_style() -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    return sys.stderr.isatty()


def _stderr_write(text: str, *, dim: bool) -> None:
    if _stderr_should_style():
        trace_console.print(
            text,
            style="dim" if dim else None,
            end="",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return
    sys.stderr.write(text)
    sys.stderr.flush()


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            sanitized_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    sanitized_parts.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                    continue
                if part.get("type") == "image_url":
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url")
                        detail = iu.get("detail")
                        sanitized_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _truncate_data_url(url),
                                    "detail": detail,
                                },
                            }
                        )
                    continue
            out.append({"role": role, "content": sanitized_parts})
            continue

        out.append({"role": role, "content": content})
    return out


def _truncate_data_url(url: object) -> str:
    if not isinstance(url, str):
        return ""
    if url.startswith("data:image/"):
        prefix_end = url.find("base64,")
        if prefix_end != -1:
            prefix_end += len("base64,")
            prefix = url[:prefix_end]
            b64 = url[prefix_end:]
            head = b64[:64]
            return f"{prefix}{head}...(len={len(b64)})"
        return url[:96] + "..."
    return url


def _build_user_content(*, text: str, images: list[str], image_detail: str) -> Any:
    if not images:
        return text

    if image_detail not in {"auto", "low", "high"}:
        raise ValueError("--image-detail must be one of: auto, low, high")

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        url = _normalize_image_url(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": url, "detail": image_detail},
            }
        )
    return content


def _normalize_image_url(value: str) -> str:
    if value.startswith("data:image/"):
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value

    path = Path(value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image file not found: {value}")

    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path.name}")

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resolve_prompts(values: list[str], stdin_state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for v in values:
        if v == "-":
            if bool(stdin_state.get("used")):
                raise ValueError("stdin (-) can only be used once")
            stdin_state["used"] = True
            stdin_state["text"] = sys.stdin.read()
            out.append(str(stdin_state.get("text", "")))
            continue

        if v.startswith("@"):  # @path/to/file
            path = Path(v[1:])
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            out.append(path.read_text(encoding="utf-8"))
            continue

        out.append(v)

    return out


def _print_error(message: str) -> int:
    console.print(
        json.dumps(
            {"ok": False, "error": message},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 1


def main() -> None:
    app()


if __name__ == "__main__":
    main()
