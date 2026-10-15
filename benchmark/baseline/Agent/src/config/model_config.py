"""Load model settings and API keys without storing secrets in YAML."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from benchmark.baseline.Agent.src.providers import LLMProvider, create_provider


MODEL_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ModelConfigError(ValueError):
    """Report invalid model configuration or missing credentials."""


@dataclass(frozen=True)
class ModelConfig:
    """Store one model's non-secret runtime settings."""

    key: str
    provider: str
    model: str
    api_key_env: tuple[str, ...]
    max_tokens: int = 4096
    request_timeout_seconds: float = 300.0
    temperature: float = 0.0
    base_url: Optional[str] = None

    def resolve_api_key(
        self,
        environment: Optional[Mapping[str, str]] = None,
    ) -> str:
        """Read the first configured API key from the environment.

        Arg:
            environment: Optional environment mapping, defaulting to os.environ.

        Returns:
            Non-empty API key value.
        """
        values = environment if environment is not None else os.environ
        for variable_name in self.api_key_env:
            value = values.get(variable_name, "").strip()
            if value:
                return value
        expected = ", ".join(self.api_key_env)
        raise ModelConfigError(
            f"No API key found for '{self.key}'. Set one of: {expected}"
        )

    def create_provider(
        self,
        environment: Optional[Mapping[str, str]] = None,
    ) -> LLMProvider:
        """Create the configured provider using an environment API key.

        Arg:
            environment: Optional environment mapping, defaulting to os.environ.

        Returns:
            Configured model provider.
        """
        return create_provider(
            provider_name=self.provider,
            api_key=self.resolve_api_key(environment),
            model=self.model,
            max_tokens=self.max_tokens,
            timeout_seconds=self.request_timeout_seconds,
            temperature=self.temperature,
            base_url=self.base_url,
        )


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Load simple KEY=VALUE entries without printing secret values.

    Arg:
        path: Optional dotenv-style file.
        override: Whether file values replace existing environment values.

    Returns:
        Names of environment variables loaded from the file.
    """
    if not path.is_file():
        return []

    loaded: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ModelConfigError(
                f"Invalid environment entry at {path}:{line_number}"
            )
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise ModelConfigError(
                f"Invalid environment name at {path}:{line_number}"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def load_model_configs(path: Path) -> dict[str, ModelConfig]:
    """Load and validate all model entries from YAML.

    Arg:
        path: YAML file containing a top-level models mapping.

    Returns:
        Model configurations keyed by experiment name.
    """
    if not path.is_file():
        raise ModelConfigError(f"Model configuration not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelConfigError(f"Invalid model configuration YAML: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ModelConfigError("Model configuration requires a 'models' mapping.")

    configs: dict[str, ModelConfig] = {}
    for raw_key, raw_settings in payload["models"].items():
        key = str(raw_key)
        if not MODEL_KEY_PATTERN.fullmatch(key):
            raise ModelConfigError(f"Invalid model configuration key: {key}")
        if not isinstance(raw_settings, dict):
            raise ModelConfigError(f"Settings for '{key}' must be a mapping.")
        configs[key] = _parse_model_config(key, raw_settings)
    if not configs:
        raise ModelConfigError("At least one model must be configured.")
    return configs


def _parse_model_config(key: str, settings: Mapping[str, Any]) -> ModelConfig:
    """Validate one YAML model entry.

    Arg:
        key: Experiment-facing model key.
        settings: Raw YAML settings for that key.

    Returns:
        Validated model configuration.
    """
    provider = settings.get("provider")
    model = settings.get("model")
    raw_api_key_env = settings.get("api_key_env")
    if not isinstance(provider, str) or not provider.strip():
        raise ModelConfigError(f"Model '{key}' requires provider.")
    if not isinstance(model, str) or not model.strip():
        raise ModelConfigError(f"Model '{key}' requires model.")
    if isinstance(raw_api_key_env, str):
        api_key_env = (raw_api_key_env,)
    elif isinstance(raw_api_key_env, list) and raw_api_key_env:
        api_key_env = tuple(str(item) for item in raw_api_key_env)
    else:
        raise ModelConfigError(f"Model '{key}' requires api_key_env.")
    if any(not ENV_NAME_PATTERN.fullmatch(name) for name in api_key_env):
        raise ModelConfigError(f"Model '{key}' has an invalid api_key_env name.")

    max_tokens = _positive_int(settings, "max_tokens", 4096, key)
    request_timeout = _positive_float(
        settings,
        "request_timeout_seconds",
        300.0,
        key,
    )
    temperature = _nonnegative_float(settings, "temperature", 0.0, key)
    base_url = settings.get("base_url")
    if base_url is not None and (
        not isinstance(base_url, str)
        or not base_url.startswith(("https://", "http://"))
    ):
        raise ModelConfigError(f"Model '{key}' has an invalid base_url.")
    return ModelConfig(
        key=key,
        provider=provider.strip(),
        model=model.strip(),
        api_key_env=api_key_env,
        max_tokens=max_tokens,
        request_timeout_seconds=request_timeout,
        temperature=temperature,
        base_url=base_url,
    )


def _positive_int(
    settings: Mapping[str, Any],
    field_name: str,
    default: int,
    model_key: str,
) -> int:
    """Read one positive integer setting.

    Arg:
        settings: Raw model settings.
        field_name: Field to validate.
        default: Value used when the field is absent.
        model_key: Model key used in error messages.

    Returns:
        Positive integer value.
    """
    value = settings.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigError(
            f"Model '{model_key}' requires positive integer {field_name}."
        )
    return value


def _positive_float(
    settings: Mapping[str, Any],
    field_name: str,
    default: float,
    model_key: str,
) -> float:
    """Read one positive numeric setting.

    Arg:
        settings: Raw model settings.
        field_name: Field to validate.
        default: Value used when the field is absent.
        model_key: Model key used in error messages.

    Returns:
        Positive floating-point value.
    """
    value = settings.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ModelConfigError(
            f"Model '{model_key}' requires positive {field_name}."
        )
    return float(value)


def _nonnegative_float(
    settings: Mapping[str, Any],
    field_name: str,
    default: float,
    model_key: str,
) -> float:
    """Read one non-negative numeric setting.

    Arg:
        settings: Raw model settings.
        field_name: Field to validate.
        default: Value used when the field is absent.
        model_key: Model key used in error messages.

    Returns:
        Non-negative floating-point value.
    """
    value = settings.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ModelConfigError(
            f"Model '{model_key}' requires non-negative {field_name}."
        )
    return float(value)
