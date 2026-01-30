from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_images: bool
    supports_tools: bool
    supports_reasoning_stream: bool
    supports_model_list: bool
    supports_stream: bool


class Provider(Protocol):
    capabilities: ProviderCapabilities

    @classmethod
    def from_config(cls, *, name: str, config: Mapping[str, Any]) -> Any: ...

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def list_models(self) -> Any: ...
