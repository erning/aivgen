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

### Example

```bash
uv run aiv chat \
    --provider zhipu --model glm-4.6v \
    --system-prompt @prompts/base.md \
    --prompt "Audio Voice 用西班牙语，其他内容用中文" \
    --image ~/Downloads/product-image.jpg \
```

```bash
uv run aiv script \
    --image ~/Downloads/product-image.jpg \
    --prompt "TikTok 德国市场" \
    --prompt "Audio Voice 用德语，其他内容用中文" \
```

```bash
uv run aiv chat \
    --provider zhipu --model glm-4.7 \
    --system-prompt "summary the following content in English, less than 300 words" \
    --prompt @prompts/script-fr.md
```
