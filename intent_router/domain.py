#!/usr/bin/env python3
"""Domain models for versioned intent routing deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_ENVIRONMENT = "prod"


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class IntentTree:
    intent_tree_id: str
    name: str
    description: str = ""
    intents: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "draft"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentTree":
        return cls(
            intent_tree_id=data["intent_tree_id"],
            name=data.get("name", data["intent_tree_id"]),
            description=data.get("description", ""),
            intents=data.get("intents", []) or [],
            status=data.get("status", "draft"),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class IntentTreeVersion:
    intent_tree_version_id: str
    intent_tree_id: str
    version: int
    name: str
    description: str = ""
    intents: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "manual"
    created_by: str = "system"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentTreeVersion":
        return cls(
            intent_tree_version_id=data["intent_tree_version_id"],
            intent_tree_id=data["intent_tree_id"],
            version=int(data.get("version", 1)),
            name=data.get("name", data["intent_tree_id"]),
            description=data.get("description", ""),
            intents=data.get("intents", []) or [],
            source=data.get("source", "manual"),
            created_by=data.get("created_by", "system"),
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass
class StrategyTemplate:
    strategy_template_id: str
    name: str
    strategy_type: str
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_schema: Dict[str, Any] = field(default_factory=dict)
    required_resources: Dict[str, bool] = field(default_factory=dict)
    built_in: bool = True
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyTemplate":
        return cls(
            strategy_template_id=data["strategy_template_id"],
            name=data.get("name", data["strategy_template_id"]),
            strategy_type=data.get("strategy_type", data["strategy_template_id"]),
            description=data.get("description", ""),
            default_params=data.get("default_params", {}) or {},
            param_schema=data.get("param_schema", {}) or {},
            required_resources=data.get("required_resources", {}) or {},
            built_in=bool(data.get("built_in", True)),
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass
class StrategyConfig:
    strategy_config_id: str
    name: str
    strategy_template_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        return cls(
            strategy_config_id=data["strategy_config_id"],
            name=data.get("name", data["strategy_config_id"]),
            strategy_template_id=data["strategy_template_id"],
            params=data.get("params", {}) or {},
            resources=data.get("resources", {}) or {},
            description=data.get("description", ""),
            version=int(data.get("version", 1)),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class Deployment:
    deployment_id: str
    app_id: str
    environment: str
    intent_tree_version_id: str
    strategy_config_id: str
    tenant_id: str = "default"
    status: str = "active"
    published_by: str = "system"
    published_at: str = field(default_factory=utc_now_iso)
    previous_deployment_id: Optional[str] = None

    @property
    def deployment_key(self) -> str:
        return f"{self.tenant_id}:{self.app_id}:{self.environment}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["deployment_key"] = self.deployment_key
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Deployment":
        return cls(
            deployment_id=data["deployment_id"],
            app_id=data["app_id"],
            environment=data.get("environment", DEFAULT_ENVIRONMENT),
            intent_tree_version_id=data["intent_tree_version_id"],
            strategy_config_id=data["strategy_config_id"],
            tenant_id=data.get("tenant_id", "default"),
            status=data.get("status", "active"),
            published_by=data.get("published_by", "system"),
            published_at=data.get("published_at", utc_now_iso()),
            previous_deployment_id=data.get("previous_deployment_id"),
        )
