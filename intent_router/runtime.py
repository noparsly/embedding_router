#!/usr/bin/env python3
"""Runtime deployment loader and hot-swap cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .domain import Deployment, IntentTreeVersion, StrategyConfig, StrategyTemplate
from .llm_provider import LLMProvider
from .router import EmbeddingProvider, RouterResult
from .storage import FileRepository
from .strategies import RouteStrategy, StrategyBuildContext, build_strategy
from .templates import merge_template_params


@dataclass
class RuntimeBundle:
    deployment: Deployment
    intent_tree_version: IntentTreeVersion
    strategy_config: StrategyConfig
    strategy_template: StrategyTemplate
    merged_params: Dict[str, Any]
    strategy: RouteStrategy

    @property
    def deployment_key(self) -> str:
        return self.deployment.deployment_key

    def summary(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment.deployment_id,
            "deployment_key": self.deployment.deployment_key,
            "tenant_id": self.deployment.tenant_id,
            "app_id": self.deployment.app_id,
            "environment": self.deployment.environment,
            "intent_tree_version_id": self.intent_tree_version.intent_tree_version_id,
            "strategy_config_id": self.strategy_config.strategy_config_id,
            "strategy_template_id": self.strategy_template.strategy_template_id,
            "strategy_type": self.strategy_template.strategy_type,
            "intent_count": len(self.intent_tree_version.intents),
            "published_at": self.deployment.published_at,
        }


class RuntimeRegistry:
    def __init__(
        self,
        repository: FileRepository,
        provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.llm_provider = llm_provider
        self._bundles: Dict[str, RuntimeBundle] = {}
        self._lock = threading.RLock()

    def build_bundle(self, deployment: Deployment) -> RuntimeBundle:
        tree_version = self.repository.get_intent_tree_version(deployment.intent_tree_version_id)
        if tree_version is None:
            raise ValueError(f"Intent tree version not found: {deployment.intent_tree_version_id}")
        strategy_config = self.repository.get_strategy_config(deployment.strategy_config_id)
        if strategy_config is None:
            raise ValueError(f"Strategy config not found: {deployment.strategy_config_id}")
        template = self.repository.get_strategy_template(strategy_config.strategy_template_id)
        if template is None:
            raise ValueError(f"Strategy template not found: {strategy_config.strategy_template_id}")
        merged_params = merge_template_params(template, strategy_config.params)
        context = StrategyBuildContext(provider=self.provider, llm_provider=self.llm_provider)
        strategy = build_strategy(template.strategy_type, tree_version.intents, merged_params, context)
        return RuntimeBundle(
            deployment=deployment,
            intent_tree_version=tree_version,
            strategy_config=strategy_config,
            strategy_template=template,
            merged_params=merged_params,
            strategy=strategy,
        )

    def reload_deployment(self, deployment: Deployment) -> RuntimeBundle:
        new_bundle = self.build_bundle(deployment)
        with self._lock:
            self._bundles[deployment.deployment_key] = new_bundle
        return new_bundle

    def reload_active(self, app_id: str, environment: str = "prod", tenant_id: str = "default") -> RuntimeBundle:
        deployment = self.repository.get_active_deployment(app_id, environment, tenant_id)
        if deployment is None:
            raise ValueError(f"No active deployment for {tenant_id}:{app_id}:{environment}")
        return self.reload_deployment(deployment)

    def reload_all_active(self) -> int:
        new_bundles: Dict[str, RuntimeBundle] = {}
        for item in self.repository.list_deployments():
            if item.get("status") != "active":
                continue
            deployment = Deployment.from_dict(item)
            bundle = self.build_bundle(deployment)
            new_bundles[deployment.deployment_key] = bundle
        with self._lock:
            self._bundles = new_bundles
        return len(new_bundles)

    def unload_app(self, app_id: str, tenant_id: Optional[str] = None) -> int:
        with self._lock:
            keys = [
                key for key, bundle in self._bundles.items()
                if bundle.deployment.app_id == app_id
                and (tenant_id is None or bundle.deployment.tenant_id == tenant_id)
            ]
            for key in keys:
                self._bundles.pop(key, None)
        return len(keys)

    def get_bundle(self, app_id: str, environment: str = "prod", tenant_id: str = "default") -> Optional[RuntimeBundle]:
        key = f"{tenant_id}:{app_id}:{environment}"
        with self._lock:
            bundle = self._bundles.get(key)
        if bundle is not None:
            return bundle
        deployment = self.repository.get_active_deployment(app_id, environment, tenant_id)
        if deployment is None:
            return None
        return self.reload_deployment(deployment)

    def route(
        self,
        *args,
        app_id: Optional[str] = None,
        environment: str = "prod",
        query: Optional[str] = None,
        tenant_id: str = "default",
        visible_intent_ids: Optional[Sequence[str]] = None,
    ) -> tuple[RouterResult, RuntimeBundle]:
        if args:
            if len(args) == 4:
                tenant_id, app_id, environment, query = args
            elif len(args) == 3:
                app_id, environment, query = args
            else:
                raise TypeError("route expects app_id, environment, query or tenant_id, app_id, environment, query")
        if app_id is None or query is None:
            raise TypeError("route requires app_id and query")
        bundle = self.get_bundle(app_id, environment, tenant_id)
        if bundle is None:
            raise ValueError(f"No runtime bundle for tenant={tenant_id}, app={app_id}, env={environment}")
        return bundle.strategy.route(query=query, visible_intent_ids=visible_intent_ids), bundle

    def list_bundles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [bundle.summary() for bundle in self._bundles.values()]
