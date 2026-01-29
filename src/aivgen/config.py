from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    data: dict[str, Any]


@dataclass(frozen=True)
class ScriptConfig:
    provider: str | None = None
    model: str | None = None
    system_prompt: list[str] | None = None
    prompt: list[str] | None = None


@dataclass(frozen=True)
class AivGenConfig:
    providers: dict[str, ProviderConfig]
    script: ScriptConfig | None = None


@dataclass(frozen=True)
class AivConfig:
    aivgen: AivGenConfig
    loaded_from: tuple[str, ...]

    def to_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for name, provider in self.aivgen.providers.items():
            providers[name] = (
                _redact_mapping(provider.data)
                if redact_secrets
                else dict(provider.data)
            )

        return {
            "aivgen": {"providers": {**providers}},
            "loaded_from": list(self.loaded_from),
        }

    def to_json(self, *, redact_secrets: bool = True) -> str:
        return json.dumps(
            self.to_dict(redact_secrets=redact_secrets), indent=2, sort_keys=True
        )


DEFAULT_PROJECT_CONFIG_FILENAMES = ("aivgen.yaml",)


def load_config(
    *,
    config_path: str | None = None,
    cwd: Path | None = None,
) -> AivConfig:
    base_dir = cwd or Path.cwd()

    loaded_from: list[str] = []
    merged: dict[str, Any] = _load_default_config()
    loaded_from.append("aivgen:config.default.yaml")

    global_path = Path.home() / ".config" / "aivgen" / "aivgen.yaml"
    if global_path.exists():
        merged = _merge_dicts(merged, _load_yaml_file(global_path))
        loaded_from.append(str(global_path))

    for name in DEFAULT_PROJECT_CONFIG_FILENAMES:
        candidate = base_dir / name
        if candidate.exists():
            merged = _merge_dicts(merged, _load_yaml_file(candidate))
            loaded_from.append(str(candidate))
            break

    if config_path is not None:
        explicit = Path(config_path)
        if not explicit.exists():
            raise ConfigError(f"Config file not found: {explicit}")
        merged = _merge_dicts(merged, _load_yaml_file(explicit))
        loaded_from.append(str(explicit))

    merged = _interpolate_env_vars(merged)

    providers_raw = _get_path(merged, ["aivgen", "providers"])
    if providers_raw is None:
        raise ConfigError("Missing aivgen.providers")
    if not isinstance(providers_raw, dict):
        raise ConfigError("Expected aivgen.providers to be a mapping")

    providers: dict[str, ProviderConfig] = {}
    for provider_name, provider_raw in providers_raw.items():
        if not isinstance(provider_name, str) or not provider_name:
            raise ConfigError("Provider name must be a non-empty string")
        if provider_raw is None:
            provider_raw = {}
        if not isinstance(provider_raw, dict):
            raise ConfigError(
                f"Expected aivgen.providers.{provider_name} to be a mapping"
            )

        providers[provider_name] = ProviderConfig(data=dict(provider_raw))

    # Parse script config if present
    script_raw = _get_path(merged, ["aivgen", "script"])
    script_config: ScriptConfig | None = None
    if script_raw is not None:
        if not isinstance(script_raw, dict):
            raise ConfigError("Expected aivgen.script to be a mapping")
        script_config = ScriptConfig(
            provider=script_raw.get("provider"),
            model=script_raw.get("model"),
            system_prompt=script_raw.get("system-prompt"),
            prompt=script_raw.get("prompt"),
        )

    return AivConfig(
        aivgen=AivGenConfig(providers=providers, script=script_config),
        loaded_from=tuple(loaded_from),
    )


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}") from e

    try:
        data = yaml.safe_load(raw)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping in {path}")
    return data


def _load_default_config() -> dict[str, Any]:
    try:
        default_path = resources.files("aivgen").joinpath("config.default.yaml")
        raw = default_path.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Failed to read built-in config.default.yaml: {e}") from e

    try:
        data = yaml.safe_load(raw)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Invalid built-in config.default.yaml: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("Built-in config.default.yaml top-level must be a mapping")
    return data


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    if isinstance(value, str):
        return _expand_env_placeholders(value)
    return value


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_placeholders(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        return os.getenv(name, "")

    return _ENV_PATTERN.sub(repl, value)


def _get_path(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _get_str(data: dict[str, Any], path: list[str], *, default: str) -> str:
    val = _get_path(data, path)
    if val is None:
        return default
    if isinstance(val, str):
        return val
    raise ConfigError(f"Expected string at {'.'.join(path)}, got {type(val).__name__}")


def _redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return f"{value[:3]}...{value[-2:]}"


_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(^|_)(api[_-]?key|apk[_-]?key|token|secret|password)($|_)"
)


def _redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in value.items():
        if isinstance(v, dict):
            out[k] = _redact_mapping(v)
            continue
        if isinstance(v, list):
            out[k] = [_redact_mapping(x) if isinstance(x, dict) else x for x in v]
            continue
        if (
            isinstance(k, str)
            and isinstance(v, str)
            and _SENSITIVE_KEY_PATTERN.search(k)
        ):
            out[k] = _redact_secret(v)
            continue
        out[k] = v
    return out
