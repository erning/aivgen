## aivgen

CLI + config + provider plumbing for a short-video generation pipeline.

### Config

Config precedence (later overrides earlier):
1. Built-in defaults: `src/aivgen/config.default.yaml`
2. `~/.config/aivgen/aivgen.yaml`
3. `./aivgen.yaml`
4. `--config <path>`

Interpolation:
- `${ENV_VAR}` will be replaced with the value of `ENV_VAR` (or `""` if unset).

### CLI

```bash
aiv config show --json
aiv provider list

aiv provider chat --provider <provider> --model <model> --prompt "..." --prompt "..."
aiv provider models --provider <provider>
```

### Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format
uv run pytest -q
```
