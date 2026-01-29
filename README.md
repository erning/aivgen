## aivgen

CLI + config + provider plumbing for AI inference and short-video generation.

### Config

Config precedence (later overrides earlier):
1. Built-in defaults: `src/aivgen/config.default.yaml`
2. `~/.config/aivgen/aivgen.yaml`
3. `./aivgen.yaml`
4. `--config <path>`

Interpolation:
- `${ENV_VAR}` will be replaced with the value of `ENV_VAR` (or `""` if unset).

Example config (`aivgen.yaml`):
```yaml
aivgen:
  providers:
    zhipu:
      type: openai-compatible
      base_url: "https://open.bigmodel.cn/api/paas/v4"
      api_key: "${AIVGEN_ZHIPU_API_KEY}"
    gemini:
      type: gemini
      api_key: "${GEMINI_API_KEY}"
  
  # Optional: defaults for script command
  script:
    provider: zhipu
    model: glm-4.6v
    system-prompt:
      - "@prompts/base.md"
```

### CLI

```bash
# List providers
aiv providers

# Chat with a model
aiv chat --provider <provider> --model <model> --prompt "..."

# List available models
aiv models --provider <provider>

# Generate video script from image
aiv script --image <path> [--output <file>]

# Show config
aiv config show [--json]
```

Global options:
- `--trace` (default: on) - Print request/response trace to stderr
- `--no-trace` - Disable trace output
- `--config <path>` - Use specific config file

### Examples

```bash
# Chat with image
uv run aiv chat \
    --provider zhipu --model glm-4.6v \
    --system-prompt @prompts/base.md \
    --prompt "Audio Voice 用西班牙语，其他内容用中文" \
    --image ~/Downloads/product-image.jpg

# Generate script with config defaults
uv run aiv script \
    --image ~/Downloads/product-image.jpg \
    --prompt "TikTok 德国市场" \
    --prompt "Audio Voice 用德语，其他内容用中文"

# Chat with file prompt
uv run aiv chat \
    --provider zhipu --model glm-4.7 \
    --system-prompt "summarize the following content in English, less than 300 words" \
    --prompt @prompts/script-fr.md
```

### Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format
uv run pytest -q
uv run basedpyright
```

### Supported Providers

- **openai-compatible** - Any OpenAI-compatible API (e.g., Zhipu, Moonshot)
- **gemini** - Native Google Gemini (supports thinking/reasoning)
