from __future__ import annotations

from aivgen.config import AivConfig, AivGenConfig, ProviderConfig
from aivgen.providers.registry import build_provider


def test_build_provider_accepts_openai_compatible_alias() -> None:
    cfg = AivConfig(
        aivgen=AivGenConfig(
            providers={
                "zhipu": ProviderConfig(
                    data={
                        "type": "openai-compatible",
                        "base_url": "https://example.invalid/v1",
                        "api_key": "sk-test",
                    }
                )
            }
        ),
        loaded_from=(),
    )

    ref = build_provider(cfg, name="zhipu")
    assert ref.type == "openai"
