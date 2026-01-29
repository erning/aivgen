# Providers Agent Notes

Provider implementations for aivgen. New providers are added here.

## Overview

Providers are dynamically dispatched by type in `registry.py`. Currently supports `openai-compatible` type.

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

`openai_compatible.py` - Uses OpenAI Python SDK.

Required config:
- `base_url`: string
- `api_key`: string

Optional:
- `headers`: mapping of string → string

Header rules:
- Provider-level `headers` apply to every request.
- Per-request headers override provider headers on key conflicts.

## Error Handling

Raise `ProviderError` (from `openai_compatible.py`) for validation failures. CLI catches and displays these.
