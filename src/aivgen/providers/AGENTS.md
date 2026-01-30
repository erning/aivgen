# Providers Agent Notes

Provider implementations for aivgen. New providers are added here.

## Overview

Providers are dynamically dispatched by type in `registry.py`.

Supported types:
- `openai` - OpenAI-compatible API
  - Alias accepted: `openai-compatible`
- `gemini` - Native Google Gemini API with thinking support
- `anthropic` - Native Anthropic Claude API with thinking support
- `ollama` - Native Ollama API for local models

## Adding a New Provider Type

1. Create provider class in this directory (e.g., `my_provider.py`).
2. Add import and dispatch in `registry.py`:
   ```python
   from aivgen.providers.my_provider import MyProvider
   
   if provider_type == "my-provider":
       provider = MyProvider.from_config(name=name, config=data)
       return ProviderRef(name=name, type=provider_type, provider=provider)
   ```

## Provider Interface

Minimal interface expected by CLI:

```python
class MyProvider:
    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> MyProvider:
        # Validate required fields, return instance
        ...
    
    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Return response or iterator for streaming
        ...
    
    def list_models(self) -> Any:
        # Optional: return models list
        ...
```

## OpenAI Compatible Provider

`openai.py` - Uses OpenAI Python SDK.

Required config:
- `base_url`: string
- `api_key`: string

Optional:
- `headers`: mapping of string → string

Header rules:
- Provider-level `headers` apply to every request.
- Per-request headers override provider headers on key conflicts.

## Gemini Provider

`gemini.py` - Native Google Gemini SDK (`google-genai`).

Supports thinking/reasoning for Gemini 2.5+ and 3.0+ models.

Required config:
- `api_key`: string

Example:
```yaml
aivgen:
  providers:
    gemini:
      type: gemini
      api_key: "${GEMINI_API_KEY}"
```

Thinking content is returned via `reasoning_content` field in streaming responses.

## Error Handling

Raise shared errors from `errors.py`:
- `ProviderConfigError` for validation failures (from_config)
- `ProviderRequestError` for runtime request failures

Providers should declare `capabilities` (see `contracts.py`) and must not silently drop unsupported features (e.g., images).
