# aivgen Agent Notes

This repo is intentionally early-stage. Keep changes small, typed, and testable.

## Ground Rules

- Do not commit secrets. Never add `.env` or API keys to git.
- Prefer minimal, explicit abstractions over broad frameworks.
- Keep config loading dumb: parse/merge/interpolate only. Validate provider-specific requirements inside provider implementations.

## Local Dev

```bash
uv sync --dev
./.venv/bin/python -m pytest -q
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

## Testing

- Keep tests unit-level by default.
- Avoid real network calls; use fake clients/stubs.
