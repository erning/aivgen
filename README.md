## aivgen

CLI + config + provider plumbing for a short-video generation pipeline.

Current scope (intentionally minimal):
- Config loading/merging with env var interpolation
- Provider registry + an OpenAI-compatible provider implementation (OpenAI Python SDK)
- CLI helpers to inspect config and list configured providers

### Install

This repo uses `uv`.

```bash
uv sync --dev
```

### Config

Config precedence (later overrides earlier):
1. Built-in defaults: `src/aivgen/config.default.yaml`
2. `~/.config/aivgen/aivgen.yaml`
3. `./aivgen.yaml`
4. `--config <path>`

Environment variable interpolation is supported for any string values:
- `${ENV_VAR}` will be replaced with the value of `ENV_VAR` (or `""` if unset).

Example `aivgen.yaml`:

```yaml
aivgen:
  providers:
    zhipu:
      type: openai-compatible
      base_url: "https://open.bigmodel.cn/api/paas/v4"
      api_key: "${AIVGEN_ZHIPU_API_KEY}"
      # Optional. Only needed for special cases.
      headers:
        User-Agent: "aivgen/0.1.0"
```

Provider config notes:
- `type` selects the implementation.
- `type: openai-compatible` uses the OpenAI Python SDK with `base_url` pointing to an OpenAI-compatible endpoint.
- `headers` are provider-level defaults used on every request.
- Per-request headers (if the call site provides any later) should override provider headers on key conflicts.
- `model` is passed at call time (not stored in config).

### CLI

```bash
# Show effective config (secrets redacted)
aiv config show --json

# List configured provider names (one per line)
aiv provider list
```

### Development

```bash
uv sync --dev
./.venv/bin/python -m pytest -q
```
