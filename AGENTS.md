# aivgen Agent Notes

This repo is intentionally early-stage. Keep changes small, typed, and testable.

## Ground Rules

- Do not commit secrets. Never add `.env` or API keys to git.
- Do not add commit trailers for coding agents (e.g. `Co-authored-by: ...`, "Ultraworked with ...").
- Prefer minimal, explicit abstractions over broad frameworks.
- Keep config loading dumb: parse/merge/interpolate only. Validate provider-specific requirements inside provider implementations.

## Local Dev

```bash
uv sync --dev
uv run ruff check .
uv run ruff format
uv run pytest -q
```

## Config

Precedence (later overrides earlier):
1. Built-in defaults: `src/aivgen/config.default.yaml`
2. Global: `~/.config/aivgen/aivgen.yaml`
3. Project: `./aivgen.yaml`
4. Explicit: `--config <path>`

Interpolation:
- Any string value may include `${ENV_VAR}` placeholders.

Shape:

```yaml
aivgen:
  providers:
    <provider-name>:
      type: <provider-type>
      # other fields are provider-dependent
```

## Providers

Provider configuration is dynamic:
- Provider names are keys under `aivgen.providers`.
- Provider type dispatch happens in `src/aivgen/providers/registry.py`.

### OpenAI Compatible Provider

Implementation: `src/aivgen/providers/openai_compatible.py`

`type: openai-compatible`
- Uses the OpenAI Python SDK.
- Requires (validated in provider code):
  - `base_url`
  - `api_key`
- Optional:
  - `headers` (mapping of string -> string)

Header rules:
- Provider-level `headers` apply to every request.
- If a call site provides request headers later, they override provider headers on key conflicts.

## CLI

Entry point: `aiv` (`pyproject.toml` -> `aivgen.cli:main`)

Commands:
- `aiv config show` prints the effective config (optionally redacted).
- `aiv provider list` prints provider names, one per line.
- `aiv provider chat` sends prompts to a provider.
- `aiv provider models` lists provider models.

## Testing

- Keep tests unit-level by default.
- Avoid real network calls; use fake clients/stubs.

## Type Checking

We use `basedpyright` for type checking.

```bash
uv run basedpyright
```

Current policy (to keep signal high in an early-stage, dynamic-config CLI):
- `tests/` is excluded from type checking.
- `src/aivgen/config.py` is excluded (config is intentionally untyped/dynamic).
- Typer parameter default initializers are allowed (`reportCallInDefaultInitializer = none`).
- Explicit `Any` is allowed at boundaries (`reportExplicitAny = none`).
