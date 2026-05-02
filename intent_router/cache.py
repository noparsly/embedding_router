#!/usr/bin/env python3
"""
意图路由器缓存层

功能：
- 多租户意图缓存，支持 cache_key 寻址
- 自动按 intent config hash 缓存/失效
- 意图配置热更新
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .router import IntentRouter, IntentDef, EmbeddingProvider, RouterConfig, RouterResult


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def validate_storage_id(value: str, field_name: str = "cache_key") -> str:
    """Validate IDs used as cache keys or file names."""
    if not isinstance(value, str) or not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 1-128 chars and contain only letters, numbers, '.', '_' or '-'"
        )
    return value


def safe_json_path(base_dir: str, item_id: str, field_name: str = "cache_key") -> str:
    """Return a JSON path under base_dir after validating the file-backed ID."""
    safe_id = validate_storage_id(item_id, field_name)
    base = Path(base_dir).resolve()
    path = (base / f"{safe_id}.json").resolve()
    if base != path.parent:
        raise ValueError(f"Invalid {field_name}: {item_id}")
    return str(path)


class IntentCache:
    """意图缓存管理器

    管理多个租户的意图路由器缓存，支持：
    - 按 cache_key 缓存意图配置
    - 意图配置变更自动检测和更新
    - 缓存列表和失效管理
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: Optional[RouterConfig] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        """初始化缓存管理器

        Args:
            provider: Embedding 提供者
            config: 路由配置（用于新构建 router）
            persist_dir: 意图配置持久化目录（可选）
        """
        self._cache: Dict[str, IntentRouter] = {}  # full_key -> IntentRouter
        self._provider = provider
        self._config = config or RouterConfig()
        self._persist_dir = persist_dir

    def _hash_intents(self, intents: List[IntentDef]) -> str:
        """计算意图配置的哈希值"""
        # 将 intents 序列化为 JSON 并计算 SHA256
        config_data = []
        for it in intents:
            config_data.append({
                "id": it.id,
                "name": it.name,
                "scope": it.scope,
                "out_of_scope": it.out_of_scope,
                "examples": list(it.examples) if it.examples else [],
                "negative_examples": list(it.negative_examples) if it.negative_examples else [],
            })
        config_str = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _make_full_key(self, cache_key: str, intents: Optional[List[IntentDef]] = None) -> str:
        """生成完整的缓存 key"""
        validate_storage_id(cache_key)
        if intents:
            config_hash = self._hash_intents(intents)
            return f"{cache_key}:{config_hash}"
        return cache_key

    def get_or_build(
        self,
        cache_key: str,
        intents: List[IntentDef],
        config: Optional[RouterConfig] = None,
    ) -> IntentRouter:
        """获取或构建意图路由器

        Args:
            cache_key: 缓存标识（租户 ID 或业务标识）
            intents: 意图定义列表
            config: 可选的路由配置（默认使用初始化时的配置）

        Returns:
            IntentRouter 实例（已缓存或新建）
        """
        if not intents:
            raise ValueError("intents must contain at least one intent")

        full_key = self._make_full_key(cache_key, intents)

        if full_key in self._cache:
            return self._cache[full_key]

        # 构建新的 router
        router_config = config or self._config
        router = IntentRouter(
            intents=intents,
            provider=self._provider,
            config=router_config,
            build_on_init=True,
        )
        self._cache[full_key] = router
        return router

    def get(self, cache_key: str) -> Optional[IntentRouter]:
        """根据 cache_key 获取已缓存的路由器

        Note: 由于同一个 cache_key 可能对应多个不同版本的意图配置，
        优先返回最新创建的版本

        Args:
            cache_key: 缓存标识

        Returns:
            匹配的 IntentRouter 或 None
        """
        validate_storage_id(cache_key)
        # 精确匹配（带 hash 的完整 key）
        if cache_key in self._cache:
            return self._cache[cache_key]
        # 前缀匹配（返回最新的版本）
        prefix_keys = [k for k in self._cache if k.startswith(f"{cache_key}:")]
        if prefix_keys:
            # 返回最后一个（最新创建的）
            return self._cache[prefix_keys[-1]]
        return None

    def invalidate(self, cache_key: str) -> int:
        """使指定 cache_key 的所有缓存失效

        Args:
            cache_key: 缓存标识

        Returns:
            删除的缓存数量
        """
        validate_storage_id(cache_key)
        keys_to_remove = [k for k in self._cache if k.startswith(f"{cache_key}:")]
        for k in keys_to_remove:
            del self._cache[k]
        # 也删除精确匹配的
        if cache_key in self._cache:
            del self._cache[cache_key]
        return len(keys_to_remove)

    def list_caches(self) -> List[Dict[str, any]]:
        """列出所有缓存信息

        Returns:
            缓存信息列表，包含 cache_key、意图数量等
        """
        result = []
        seen_keys: Dict[str, set] = {}
        for full_key in self._cache:
            parts = full_key.split(":")
            ck = parts[0]
            if ck not in seen_keys:
                seen_keys[ck] = set()
            if len(parts) > 1:
                seen_keys[ck].add(parts[1])

        for cache_key, hashes in seen_keys.items():
            router = self._cache.get(f"{cache_key}:{list(hashes)[0]}")
            if router:
                result.append({
                    "cache_key": cache_key,
                    "version_count": len(hashes),
                    "intent_count": len(router.intents),
                    "versions": list(hashes),
                })
        return result

    def route(
        self,
        cache_key: str,
        query: str,
        intents: Optional[List[IntentDef]] = None,
        visible_intent_ids: Optional[Sequence[str]] = None,
        config: Optional[RouterConfig] = None,
        abs_threshold: Optional[float] = None,
        gap_threshold: Optional[float] = None,
    ) -> "RouterResult":
        """自动化的路由方法，会自动处理缓存

        Args:
            cache_key: 缓存标识
            query: 用户查询
            intents: 意图定义（首次必传，后续可省略）
            visible_intent_ids: 可见意图 ID 白名单
            config: 可选的路由配置
            abs_threshold: 绝对相似度阈值
            gap_threshold: 歧义判定阈值

        Returns:
            RouterResult
        """
        if not intents:
            # 尝试从缓存获取
            router = self.get(cache_key)
            if router is None:
                raise ValueError(f"No cached router found for cache_key: {cache_key}")
        else:
            router = self.get_or_build(cache_key, intents, config)

        return router.route(
            query=query,
            visible_intent_ids=visible_intent_ids,
            abs_threshold=abs_threshold,
            gap_threshold=gap_threshold,
        )

    def save_intent_config(self, cache_key: str, intents: List[IntentDef]) -> str:
        """持久化意图配置到文件

        Args:
            cache_key: 缓存标识
            intents: 意图定义列表

        Returns:
            保存的文件路径
        """
        if not self._persist_dir:
            raise ValueError("persist_dir not configured")

        os.makedirs(self._persist_dir, exist_ok=True)

        config_data = {
            "cache_key": cache_key,
            "version": 1,
            "intents": []
        }
        for it in intents:
            config_data["intents"].append({
                "id": it.id,
                "name": it.name,
                "scope": it.scope,
                "out_of_scope": it.out_of_scope,
                "examples": list(it.examples) if it.examples else [],
                "negative_examples": list(it.negative_examples) if it.negative_examples else [],
            })

        file_path = safe_json_path(self._persist_dir, cache_key)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        return file_path

    def load_intent_config(self, cache_key: str) -> Optional[List[IntentDef]]:
        """从文件加载意图配置

        Args:
            cache_key: 缓存标识

        Returns:
            IntentDef 列表或 None
        """
        if not self._persist_dir:
            return None

        file_path = safe_json_path(self._persist_dir, cache_key)
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        intents = []
        for item in config_data.get("intents", []):
            intents.append(IntentDef(
                id=item["id"],
                name=item.get("name", item["id"]),
                scope=item.get("scope", ""),
                out_of_scope=item.get("out_of_scope", ""),
                examples=item.get("examples", []) or [],
                negative_examples=item.get("negative_examples", []) or [],
            ))
        return intents if intents else None

    def preload_from_disk(self) -> int:
        """从持久化目录预加载所有意图配置到缓存

        Returns:
            加载的配置数量
        """
        if not self._persist_dir:
            return 0

        count = 0
        for filename in os.listdir(self._persist_dir):
            if filename.endswith(".json"):
                cache_key = filename[:-5]  # 去掉 .json
                validate_storage_id(cache_key)
                intents = self.load_intent_config(cache_key)
                if intents:
                    self.get_or_build(cache_key, intents)
                    count += 1
        return count


def intent_defs_from_dicts(data: List[dict]) -> List[IntentDef]:
    """将字典列表转换为 IntentDef 列表

    Args:
        data: 意图字典列表

    Returns:
        IntentDef 列表
    """
    if not data:
        raise ValueError("intents must contain at least one intent")

    result = []
    for item in data:
        if "id" not in item or "name" not in item:
            raise ValueError("Intent must have id and name")
        validate_storage_id(str(item["id"]), "intent id")
        result.append(IntentDef(
            id=item["id"],
            name=item["name"],
            scope=item.get("scope", ""),
            out_of_scope=item.get("out_of_scope", ""),
            examples=item.get("examples", []) or [],
            negative_examples=item.get("negative_examples", []) or [],
        ))
    return result
