#!/usr/bin/env python3
"""Environment-aware configuration helpers.

Configuration precedence:
1. Environment variables
2. config/config.{APP_ENV}.yaml, or CONFIG_FILE when provided
3. Caller defaults
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


APP_ENV = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "local"


@lru_cache(maxsize=1)
def load_app_config() -> Dict[str, Any]:
    config_file = os.getenv("CONFIG_FILE", f"config/config.{APP_ENV}.yaml")
    if not os.path.exists(config_file):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(name: str, default: Any = None) -> Any:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    return load_app_config().get(name, default)


def get_str(name: str, default: str = "") -> str:
    value = get_config(name, default)
    return str(value) if value is not None else default


def get_int(name: str, default: int = 0) -> int:
    return int(get_config(name, default))


def get_float(name: str, default: float = 0.0) -> float:
    return float(get_config(name, default))


def get_bool(name: str, default: bool = False) -> bool:
    value = get_config(name, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def get_list(name: str, default: List[str] | None = None) -> List[str]:
    value = get_config(name, default or [])
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]
