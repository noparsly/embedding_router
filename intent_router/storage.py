#!/usr/bin/env python3
"""File-backed repository for the first versioned routing workflow."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import safe_json_path, validate_storage_id
from .domain import Deployment, IntentTree, IntentTreeVersion, StrategyConfig, StrategyTemplate, utc_now_iso
from .templates import built_in_strategy_templates


class FileRepository:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self.intent_tree_dir = os.path.join(data_dir, "intent_trees")
        self.intent_tree_version_dir = os.path.join(data_dir, "intent_tree_versions")
        self.strategy_template_dir = os.path.join(data_dir, "strategy_templates")
        self.strategy_config_dir = os.path.join(data_dir, "strategy_configs")
        self.deployment_dir = os.path.join(data_dir, "deployments")
        self.legacy_intent_dir = os.path.join(data_dir, "intents")
        for directory in [
            self.intent_tree_dir,
            self.intent_tree_version_dir,
            self.strategy_template_dir,
            self.strategy_config_dir,
            self.deployment_dir,
            self.legacy_intent_dir,
        ]:
            os.makedirs(directory, exist_ok=True)
        self.ensure_built_in_strategy_templates()

    @staticmethod
    def _read_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return path

    @staticmethod
    def _list_json(base_dir: str) -> List[Dict[str, Any]]:
        if not os.path.exists(base_dir):
            return []
        result: List[Dict[str, Any]] = []
        for filename in os.listdir(base_dir):
            if not filename.endswith(".json"):
                continue
            item_id = filename[:-5]
            validate_storage_id(item_id, "item_id")
            data = FileRepository._read_json(safe_json_path(base_dir, item_id, "item_id"))
            if data:
                result.append(data)
        return result

    @staticmethod
    def make_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def ensure_built_in_strategy_templates(self) -> None:
        for template in built_in_strategy_templates():
            path = safe_json_path(self.strategy_template_dir, template.strategy_template_id, "strategy_template_id")
            if not os.path.exists(path):
                self._write_json(path, template.to_dict())

    def save_intent_tree(self, tree: IntentTree) -> IntentTree:
        validate_storage_id(tree.intent_tree_id, "intent_tree_id")
        tree.updated_at = utc_now_iso()
        self._write_json(
            safe_json_path(self.intent_tree_dir, tree.intent_tree_id, "intent_tree_id"),
            tree.to_dict(),
        )
        return tree

    def get_intent_tree(self, intent_tree_id: str) -> Optional[IntentTree]:
        data = self._read_json(safe_json_path(self.intent_tree_dir, intent_tree_id, "intent_tree_id"))
        return IntentTree.from_dict(data) if data else None

    def list_intent_trees(self) -> List[Dict[str, Any]]:
        items = self._list_json(self.intent_tree_dir)
        return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_intent_tree(self, intent_tree_id: str) -> bool:
        path = safe_json_path(self.intent_tree_dir, intent_tree_id, "intent_tree_id")
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def create_intent_tree_version(
        self,
        intent_tree_id: str,
        source: str = "manual",
        created_by: str = "system",
    ) -> IntentTreeVersion:
        tree = self.get_intent_tree(intent_tree_id)
        if tree is None:
            legacy = self.get_legacy_intent_config(intent_tree_id)
            if not legacy:
                raise ValueError(f"Intent tree not found: {intent_tree_id}")
            tree = IntentTree(
                intent_tree_id=intent_tree_id,
                name=legacy.get("name", intent_tree_id),
                description=legacy.get("description", ""),
                intents=legacy.get("intents", []) or [],
            )
            self.save_intent_tree(tree)

        existing = [v for v in self.list_intent_tree_versions(intent_tree_id)]
        next_version = max([int(v.get("version", 0)) for v in existing] or [0]) + 1
        version_id = f"{intent_tree_id}_v{next_version}"
        validate_storage_id(version_id, "intent_tree_version_id")
        version = IntentTreeVersion(
            intent_tree_version_id=version_id,
            intent_tree_id=intent_tree_id,
            version=next_version,
            name=tree.name,
            description=tree.description,
            intents=tree.intents,
            source=source,
            created_by=created_by,
        )
        self._write_json(
            safe_json_path(self.intent_tree_version_dir, version_id, "intent_tree_version_id"),
            version.to_dict(),
        )
        return version

    def get_intent_tree_version(self, version_id: str) -> Optional[IntentTreeVersion]:
        data = self._read_json(safe_json_path(self.intent_tree_version_dir, version_id, "intent_tree_version_id"))
        return IntentTreeVersion.from_dict(data) if data else None

    def list_intent_tree_versions(self, intent_tree_id: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self._list_json(self.intent_tree_version_dir)
        if intent_tree_id:
            items = [x for x in items if x.get("intent_tree_id") == intent_tree_id]
        return sorted(items, key=lambda x: (x.get("intent_tree_id", ""), int(x.get("version", 0))), reverse=True)

    def save_strategy_template(self, template: StrategyTemplate) -> StrategyTemplate:
        validate_storage_id(template.strategy_template_id, "strategy_template_id")
        self._write_json(
            safe_json_path(self.strategy_template_dir, template.strategy_template_id, "strategy_template_id"),
            template.to_dict(),
        )
        return template

    def get_strategy_template(self, template_id: str) -> Optional[StrategyTemplate]:
        data = self._read_json(safe_json_path(self.strategy_template_dir, template_id, "strategy_template_id"))
        return StrategyTemplate.from_dict(data) if data else None

    def list_strategy_templates(self) -> List[Dict[str, Any]]:
        return sorted(self._list_json(self.strategy_template_dir), key=lambda x: x.get("strategy_template_id", ""))

    def save_strategy_config(self, config: StrategyConfig) -> StrategyConfig:
        validate_storage_id(config.strategy_config_id, "strategy_config_id")
        if self.get_strategy_template(config.strategy_template_id) is None:
            raise ValueError(f"Strategy template not found: {config.strategy_template_id}")
        config.updated_at = utc_now_iso()
        self._write_json(
            safe_json_path(self.strategy_config_dir, config.strategy_config_id, "strategy_config_id"),
            config.to_dict(),
        )
        return config

    def get_strategy_config(self, config_id: str) -> Optional[StrategyConfig]:
        data = self._read_json(safe_json_path(self.strategy_config_dir, config_id, "strategy_config_id"))
        return StrategyConfig.from_dict(data) if data else None

    def list_strategy_configs(self) -> List[Dict[str, Any]]:
        items = self._list_json(self.strategy_config_dir)
        return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)

    def create_default_strategy_config(self, template_id: str = "hybrid_retrieval_default") -> StrategyConfig:
        template = self.get_strategy_template(template_id)
        if template is None:
            raise ValueError(f"Strategy template not found: {template_id}")
        config_id = f"{template_id}_cfg"
        existing = self.get_strategy_config(config_id)
        if existing:
            return existing
        config = StrategyConfig(
            strategy_config_id=config_id,
            name=f"{template.name} 默认配置",
            strategy_template_id=template.strategy_template_id,
            params={},
            resources={},
            description="由系统基于内置模板创建的默认策略配置。",
        )
        return self.save_strategy_config(config)

    def save_deployment(self, deployment: Deployment) -> Deployment:
        validate_storage_id(deployment.deployment_id, "deployment_id")
        self._write_json(
            safe_json_path(self.deployment_dir, deployment.deployment_id, "deployment_id"),
            deployment.to_dict(),
        )
        return deployment

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        data = self._read_json(safe_json_path(self.deployment_dir, deployment_id, "deployment_id"))
        return Deployment.from_dict(data) if data else None

    def list_deployments(self) -> List[Dict[str, Any]]:
        items = self._list_json(self.deployment_dir)
        return sorted(items, key=lambda x: x.get("published_at", ""), reverse=True)

    def get_active_deployment(self, app_id: str, environment: str = "prod", tenant_id: str = "default") -> Optional[Deployment]:
        for item in self.list_deployments():
            if (
                item.get("tenant_id", "default") == tenant_id
                and item.get("app_id") == app_id
                and item.get("environment", "prod") == environment
                and item.get("status") == "active"
            ):
                return Deployment.from_dict(item)
        return None

    def publish(
        self,
        app_id: str,
        environment: str,
        intent_tree_version_id: str,
        strategy_config_id: str,
        published_by: str = "system",
        tenant_id: str = "default",
    ) -> Deployment:
        if self.get_intent_tree_version(intent_tree_version_id) is None:
            raise ValueError(f"Intent tree version not found: {intent_tree_version_id}")
        if self.get_strategy_config(strategy_config_id) is None:
            raise ValueError(f"Strategy config not found: {strategy_config_id}")

        previous = self.get_active_deployment(app_id, environment, tenant_id)
        if previous:
            previous.status = "inactive"
            self.save_deployment(previous)

        deployment = Deployment(
            deployment_id=f"dep_{uuid.uuid4().hex[:12]}",
            app_id=app_id,
            environment=environment,
            intent_tree_version_id=intent_tree_version_id,
            strategy_config_id=strategy_config_id,
            tenant_id=tenant_id,
            published_by=published_by,
            previous_deployment_id=previous.deployment_id if previous else None,
        )
        return self.save_deployment(deployment)

    def get_legacy_intent_config(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self._read_json(safe_json_path(self.legacy_intent_dir, cache_key, "cache_key"))

    def save_legacy_intent_config(self, cache_key: str, data: Dict[str, Any]) -> str:
        return self._write_json(safe_json_path(self.legacy_intent_dir, cache_key, "cache_key"), data)
