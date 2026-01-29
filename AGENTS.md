# aivgen Agent Notes

CLI + config + provider plumbing for AI inference. Early-stage—keep changes small, typed, and testable.

## Ground Rules

- Do not commit secrets. Never add `.env` or API keys to git.
- Do not add commit trailers for coding agents (e.g., `Co-authored-by: ...`, "Ultraworked with ...").
- Prefer minimal, explicit abstractions over broad frameworks.
- Keep config loading dumb: parse/merge/interpolate only. Validate provider-specific requirements inside provider implementations.

## Structure

```
.
├── src/aivgen/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entry point
│   ├── config.py               # Config loading (excluded from type checking)
│   ├── config.default.yaml     # Built-in defaults
│   └── providers/              # Provider implementations
│       ├── __init__.py
│       ├── gemini.py           # Native Gemini provider with thinking support
│       ├── registry.py         # Provider dispatch
│       └── openai_compatible.py
├── tests/                      # Unit tests (excluded from type checking)
├── pyproject.toml              # uv, ruff, pytest config
├── pyrightconfig.json          # basedpyright settings
└── aivgen.yaml                 # Project-local config
```

## Where to Look

| Task | Location |
|------|----------|
| Add CLI command | `src/aivgen/cli.py` |
| Add provider type | `src/aivgen/providers/` + `registry.py` |
| Config loading | `src/aivgen/config.py` |
| Provider interface | `src/aivgen/providers/openai_compatible.py` |
| Tests | `tests/test_*.py` |

## Config

Precedence (later overrides earlier):
1. Built-in: `src/aivgen/config.default.yaml`
2. Global: `~/.config/aivgen/aivgen.yaml`
3. Project: `./aivgen.yaml`
4. Explicit: `--config <path>`

Interpolation: `${ENV_VAR}` → value (or `""` if unset).

Shape:
```yaml
aivgen:
  providers:
    <provider-name>:
      type: <provider-type>
      # provider-specific fields
  
  # Optional: script command defaults
  script:
    provider: <default-provider>
    model: <default-model>
    system-prompt:
      - "base system prompt"
    prompt:
      - "base prompt"
```

## Providers

Provider configuration is dynamic:
- Provider names are keys under `aivgen.providers`.
- Type dispatch happens in `src/aivgen/providers/registry.py`.

Supported types:
- `openai-compatible` - Any OpenAI-compatible API
- `gemini` - Native Google Gemini (supports thinking/reasoning)

See `src/aivgen/providers/AGENTS.md` for provider implementation details.

## CLI

Entry point: `aiv` (`pyproject.toml` → `aivgen.cli:main`)

Commands:
- `aiv providers` - List configured providers
- `aiv chat --provider <p> --model <m> --prompt "..."` - Chat with model
- `aiv models --provider <p>` - List available models
- `aiv script --image <path> [--output <file>]` - Generate video script from image

Global options:
- `--trace` (default: on) - Print request/response trace to stderr
- `--no-trace` - Disable trace output
- `--config <path>` - Use specific config file

## Local Dev

```bash
uv sync --dev
uv run ruff check .
uv run ruff format
uv run pytest -q
uv run basedpyright
```

## Type Checking

We use `basedpyright`.

Policy (keep signal high):
- `tests/` excluded.
- `src/aivgen/config.py` excluded (intentionally untyped/dynamic).
- `reportCallInDefaultInitializer = none` (Typer pattern).
- `reportExplicitAny = none` at boundaries.

## Testing

- Keep tests unit-level by default.
- Avoid real network calls; use fake clients/stubs.
