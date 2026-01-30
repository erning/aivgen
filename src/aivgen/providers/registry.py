from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aivgen.config import AivConfig, ProviderConfig
from aivgen.providers.anthropic import AnthropicProvider
from aivgen.providers.errors import ProviderConfigError
from aivgen.providers.gemini import GeminiProvider
from aivgen.providers.ollama import OllamaProvider
from aivgen.providers.openai import OpenAIProvider


@dataclass(frozen=True)
class ProviderRef:
    name: str
    type: str
    provider: Any


def build_provider(cfg: AivConfig, *, name: str) -> ProviderRef:
    if name not in cfg.aivgen.providers:
        raise ProviderConfigError(f"Unknown provider: {name!r}")

    provider_cfg: ProviderConfig = cfg.aivgen.providers[name]
    data = provider_cfg.data
    provider_type = data.get("type")
    if provider_type is None:
        raise ProviderConfigError(f"Provider {name!r}: missing type")
    if not isinstance(provider_type, str) or not provider_type:
        raise ProviderConfigError(f"Provider {name!r}: invalid type")

    provider_type = {"openai-compatible": "openai"}.get(provider_type, provider_type)

    if provider_type == "openai":
        provider = OpenAIProvider.from_config(name=name, config=data)
        return ProviderRef(name=name, type=provider_type, provider=provider)

    if provider_type == "gemini":
        provider = GeminiProvider.from_config(name=name, config=data)
        return ProviderRef(name=name, type=provider_type, provider=provider)

    if provider_type == "anthropic":
        provider = AnthropicProvider.from_config(name=name, config=data)
        return ProviderRef(name=name, type=provider_type, provider=provider)

    if provider_type == "ollama":
        provider = OllamaProvider.from_config(name=name, config=data)
        return ProviderRef(name=name, type=provider_type, provider=provider)

    raise ProviderConfigError(f"Provider {name!r}: unsupported type {provider_type!r}")
