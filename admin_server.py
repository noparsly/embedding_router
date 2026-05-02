#!/usr/bin/env python3
"""
运营平台 API 服务

提供意图树维护、策略配置、在线测试和发布功能
"""

from __future__ import annotations

import json
import os
import hashlib
import threading
import csv
import io
import time
import math
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from intent_router.router import EmbeddingProvider, IntentRouter, RouterConfig, IntentDef
from intent_router.tencent_provider import TencentEmbeddingProvider
from intent_router.cache import IntentCache, intent_defs_from_dicts, safe_json_path, validate_storage_id
from intent_router.domain import Deployment, IntentTree, StrategyConfig
from intent_router.runtime import RuntimeRegistry
from intent_router.storage import FileRepository
from intent_router.strategies import RouteStrategy, StrategyBuildContext, build_strategy
from intent_router.templates import merge_template_params
from intent_router.settings import APP_ENV, get_bool, get_int, get_list, get_str

# ----------------------------
# 配置参数
# ----------------------------

ADMIN_PORT = get_int("ADMIN_PORT", 8001)
DATA_DIR = get_str("DATA_DIR", "data")
INTENT_DIR = os.path.join(DATA_DIR, "intents")
CONFIG_FILE = os.path.join(DATA_DIR, "admin_config.json")
EVAL_SET_DIR = os.path.join(DATA_DIR, "eval_sets")
EVAL_RUN_DIR = os.path.join(DATA_DIR, "eval_runs")
ROUTER_SERVICE_URL = get_str("ROUTER_SERVICE_URL", "http://localhost:8000").rstrip("/")
ROUTER_ADMIN_API_KEY = get_str("ROUTER_ADMIN_API_KEY", "")
AUTO_PUBLISH_TO_ROUTER = get_bool("AUTO_PUBLISH_TO_ROUTER", True)

# Embedding 配置：生产形态只保留腾讯云在线 Embedding
EMBEDDING_PROVIDER = get_str("EMBEDDING_PROVIDER", "tencent")
publish_lock = threading.RLock()
test_strategy_lock = threading.RLock()
test_strategy_cache: Dict[str, RouteStrategy] = {}
TEST_STRATEGY_CACHE_MAX = 32

# 默认策略配置
DEFAULT_ROUTER_CONFIG = {
    "abs_threshold": 0.35,
    "gap_threshold": 0.05,
    "top_k": 3,
    "weight_intro": 0.5,
    "alpha_short": 0.6,
    "alpha_long": 0.8,
    "enable_bm25": True,
    "short_query_len": 10,
    "question_top_k": 5,
    "example_question_threshold": 0.95,
}

# 默认 LLM 配置
DEFAULT_LLM_CONFIG = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "max_tokens": 256,
    "temperature": 0.1,
    "timeout": 5,
    "min_confidence": 0.65,
}

# ----------------------------
# 创建目录
# ----------------------------

os.makedirs(INTENT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVAL_SET_DIR, exist_ok=True)
os.makedirs(EVAL_RUN_DIR, exist_ok=True)

# ----------------------------
# 配置加载/保存
# ----------------------------

def _load_admin_config() -> Dict[str, Any]:
    """加载管理员配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"router_config": DEFAULT_ROUTER_CONFIG, "llm_config": DEFAULT_LLM_CONFIG}


def _save_admin_config(config: Dict[str, Any]) -> None:
    """保存管理员配置"""
    temp_path = CONFIG_FILE + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, CONFIG_FILE)


# ----------------------------
# FastAPI 应用
# ----------------------------

app = FastAPI(title="Intent Router Admin", version="0.1")

# CORS 配置
ALLOWED_ORIGINS = get_list("ALLOWED_ORIGINS", ["http://localhost", "http://127.0.0.1"])
ADMIN_API_KEY = get_str("ADMIN_API_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_admin_api_key(request: Request, call_next):
    """Require an API key for admin endpoints when ADMIN_API_KEY is configured."""
    if ADMIN_API_KEY and request.url.path.startswith("/admin"):
        supplied = request.headers.get("x-admin-api-key")
        if supplied != ADMIN_API_KEY:
            return JSONResponse({"detail": "Invalid admin API key"}, status_code=401)
    return await call_next(request)

# 加载配置
admin_config = _load_admin_config()
router_config_dict = {**DEFAULT_ROUTER_CONFIG, **admin_config.get("router_config", {})}
llm_config_dict = {**DEFAULT_LLM_CONFIG, **admin_config.get("llm_config", {})}


def _build_provider() -> EmbeddingProvider:
    if EMBEDDING_PROVIDER != "tencent":
        raise RuntimeError("Only EMBEDDING_PROVIDER=tencent is supported in this deployment")
    endpoint = get_str("TENCENT_EMBEDDING_ENDPOINT", "")
    if not endpoint:
        raise RuntimeError("TENCENT_EMBEDDING_ENDPOINT is required")
    return TencentEmbeddingProvider(
        endpoint=endpoint,
        api_key=get_str("TENCENT_API_KEY", ""),
        secret_id=get_str("TENCENT_SECRET_ID", "") or None,
        secret_key=get_str("TENCENT_SECRET_KEY", "") or None,
        model=get_str("TENCENT_MODEL", "sn-large-multi-language-v0.2.5"),
        text_type=get_str("TENCENT_TEXT_TYPE", "") or None,
        instruction=get_str("TENCENT_INSTRUCTION", "") or None,
        service=get_str("TENCENT_SERVICE", "lkeap"),
        version=get_str("TENCENT_VERSION", "2024-05-22"),
        region=get_str("TENCENT_REGION", "ap-guangzhou"),
        token=get_str("TENCENT_TOKEN", ""),
        timeout=get_int("TENCENT_TIMEOUT", 60),
        max_batch_size=get_int("TENCENT_MAX_BATCH_SIZE", 7),
    )


def _build_router_config_from_dict(d: Dict[str, Any]) -> RouterConfig:
    return RouterConfig(
        top_k=d.get("top_k", 3),
        abs_threshold=d.get("abs_threshold", 0.35),
        gap_threshold=d.get("gap_threshold", 0.05),
        weight_intro=d.get("weight_intro", 0.5),
        alpha_short=d.get("alpha_short", 0.6),
        alpha_long=d.get("alpha_long", 0.8),
        enable_bm25=d.get("enable_bm25", True),
        short_query_len=d.get("short_query_len", 10),
        question_top_k=d.get("question_top_k", 5),
        example_question_threshold=d.get("example_question_threshold", 0.95),
    )


def _build_llm_provider(config: Dict[str, Any]):
    """根据配置构建 LLM Provider"""
    if not config.get("api_key"):
        return None
    try:
        from intent_router.llm_provider import OpenAICompatibleProvider
        provider_type = config.get("provider", "deepseek")
        if provider_type == "openai":
            base_url = config.get("base_url", "https://api.openai.com/v1")
            model = config.get("model", "gpt-4")
        elif provider_type == "deepseek":
            base_url = config.get("base_url", "https://api.deepseek.com/v1")
            model = config.get("model", "deepseek-chat")
        else:
            base_url = config.get("base_url", "https://api.deepseek.com/v1")
            model = config.get("model", "deepseek-chat")

        return OpenAICompatibleProvider(
            api_key=config.get("api_key", ""),
            base_url=base_url,
            model=model,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0.1),
            timeout=config.get("timeout", 5),
        )
    except Exception as e:
        print(f"Warning: Failed to initialize LLM provider: {e}")
        return None


# 初始化
repository = FileRepository(DATA_DIR)
router_config = _build_router_config_from_dict(router_config_dict)
llm_provider = _build_llm_provider(llm_config_dict)
embedding_provider = _build_provider()

# 意图缓存管理器（兼容旧 cache_key 接口）
intent_cache = IntentCache(
    provider=embedding_provider,
    config=router_config,
    persist_dir=INTENT_DIR,
)
runtime_registry = RuntimeRegistry(repository=repository, provider=embedding_provider, llm_provider=llm_provider)

# ----------------------------
# 请求/响应模型
# ----------------------------


class RouterConfigModel(BaseModel):
    abs_threshold: float = Field(0.35, description="相似度阈值")
    gap_threshold: float = Field(0.05, description="歧义判定阈值")
    top_k: int = Field(3, description="返回候选数")
    weight_intro: float = Field(0.5, description="意图描述权重")
    alpha_short: float = Field(0.6, description="短查询语义权重")
    alpha_long: float = Field(0.8, description="长查询语义权重")
    enable_bm25: bool = Field(True, description="启用BM25")
    short_query_len: int = Field(10, description="短查询长度阈值")
    question_top_k: int = Field(5, description="问题TopK")
    example_question_threshold: float = Field(0.95, description="示例问题直连阈值")


class LLMConfigModel(BaseModel):
    provider: str = Field("deepseek", description="提供商: deepseek/openai/custom")
    api_key: str = Field("", description="API密钥")
    base_url: str = Field("https://api.deepseek.com/v1", description="API地址")
    model: str = Field("deepseek-chat", description="模型名称")
    max_tokens: int = Field(256, description="最大Token数")
    temperature: float = Field(0.1, description="温度参数")
    timeout: float = Field(5, description="调用超时秒数")
    min_confidence: float = Field(0.65, description="LLM结果最低采信置信度")


class IntentItem(BaseModel):
    id: str
    name: str
    scope: Optional[str] = ""
    out_of_scope: Optional[str] = ""
    examples: List[str] = []
    negative_examples: List[str] = []


class CreateAppRequest(BaseModel):
    app_id: str = Field(..., description="应用 ID")
    name: str = Field(..., description="应用/意图树名称")
    description: str = Field("", description="应用描述")
    intents: List[Dict[str, Any]] = Field(..., description="意图定义列表")


class UpdateAppRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    intents: Optional[List[Dict[str, Any]]] = None


class TestAppRouteRequest(BaseModel):
    app_id: str
    query: str
    abs_threshold: Optional[float] = None
    gap_threshold: Optional[float] = None


class CreateIntentRequest(BaseModel):
    cache_key: str = Field(..., description="缓存标识")
    name: str = Field(..., description="意图名称（用于显示）")
    description: str = Field("", description="意图描述")
    intents: List[Dict[str, Any]] = Field(..., description="意图定义列表")


class UpdateIntentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    intents: Optional[List[Dict[str, Any]]] = None


class TestRouteRequest(BaseModel):
    cache_key: str
    query: str
    abs_threshold: Optional[float] = None
    gap_threshold: Optional[float] = None


class IntentTreeRequest(BaseModel):
    intent_tree_id: str = Field(..., description="意图树 ID")
    name: str = Field(..., description="意图树名称")
    description: str = ""
    intents: List[Dict[str, Any]] = Field(..., description="意图定义列表")


class IntentTreeUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    intents: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None


class StrategyConfigRequest(BaseModel):
    strategy_config_id: str = Field(..., description="策略配置 ID")
    name: str = Field(..., description="策略配置名称")
    strategy_template_id: str = Field(..., description="策略模板 ID")
    description: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)


class StrategyConfigUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None


class TestStrategyRouteRequest(BaseModel):
    query: str
    intent_tree_version_id: Optional[str] = None
    intent_tree_id: Optional[str] = None
    strategy_config_id: Optional[str] = None
    strategy_template_id: str = "hybrid_retrieval_default"
    params_override: Dict[str, Any] = Field(default_factory=dict)
    visible_intent_ids: Optional[List[str]] = None


class PublishRequest(BaseModel):
    app_id: str
    environment: str = "prod"
    intent_tree_id: Optional[str] = None
    intent_tree_version_id: Optional[str] = None
    strategy_config_id: Optional[str] = None
    strategy_template_id: str = "hybrid_retrieval_default"
    published_by: str = "system"
    tenant_id: str = "default"


class PublishedRouteRequest(BaseModel):
    app_id: str
    environment: str = "prod"
    query: str
    visible_intent_ids: Optional[List[str]] = None
    tenant_id: str = "default"


class EvalRunRequest(BaseModel):
    app_id: str
    eval_set_id: str
    strategy_config_id: Optional[str] = None
    strategy_template_id: str = "hybrid_retrieval_default"
    params_override: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# 辅助函数
# ----------------------------


def _load_intent_file(cache_key: str) -> Optional[Dict]:
    """从文件加载意图配置"""
    file_path = safe_json_path(INTENT_DIR, cache_key)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_intent_file(cache_key: str, data: Dict) -> str:
    """保存意图配置到文件"""
    file_path = safe_json_path(INTENT_DIR, cache_key)
    temp_path = file_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, file_path)
    return file_path


def _sync_intent_tree_from_legacy(cache_key: str, data: Dict, source: str) -> Dict[str, Any]:
    """同步旧 cache_key 意图配置到新版意图树和版本快照。"""
    tree = IntentTree(
        intent_tree_id=cache_key,
        name=data.get("name", cache_key),
        description=data.get("description", ""),
        intents=data.get("intents", []) or [],
    )
    repository.save_intent_tree(tree)
    version = repository.create_intent_tree_version(cache_key, source=source)
    return {"intent_tree_id": cache_key, "intent_tree_version_id": version.intent_tree_version_id}


def _router_admin_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if ROUTER_ADMIN_API_KEY:
        headers["x-admin-api-key"] = ROUTER_ADMIN_API_KEY
    return headers


def _call_router_admin(method: str, path: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """Call the recognition service admin API; failures are returned as data."""
    url = f"{ROUTER_SERVICE_URL}{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=_router_admin_headers(),
            json=payload,
            timeout=10,
        )
        try:
            body = resp.json()
        except Exception:
            body = {"text": resp.text}
        return {"ok": resp.ok, "status_code": resp.status_code, "url": url, "response": body}
    except Exception as e:
        return {"ok": False, "status_code": None, "url": url, "error": str(e)}


def _runtime_publish_payload() -> Dict[str, Any]:
    return {
        "router_config": router_config_dict,
        "llm_config": llm_config_dict,
    }


def _publish_runtime_config() -> Dict[str, Any]:
    return _call_router_admin("PUT", "/admin/runtime-config", _runtime_publish_payload())


def _publish_cache_key(cache_key: str) -> Dict[str, Any]:
    return _call_router_admin("POST", f"/admin/reload-cache/{cache_key}")


def _publish_all() -> Dict[str, Any]:
    with publish_lock:
        runtime = _publish_runtime_config()
        reload_result = _call_router_admin("POST", "/admin/reload-all")
        return {"runtime": runtime, "reload": reload_result, "ok": bool(runtime.get("ok") and reload_result.get("ok"))}


def _maybe_publish_cache_key(cache_key: str) -> Optional[Dict[str, Any]]:
    if not AUTO_PUBLISH_TO_ROUTER:
        return None
    with publish_lock:
        runtime = _publish_runtime_config()
        reload_result = _publish_cache_key(cache_key)
        return {"runtime": runtime, "reload": reload_result, "ok": bool(runtime.get("ok") and reload_result.get("ok"))}


def _list_intent_files() -> List[Dict]:
    """列出所有意图配置"""
    result = []
    for filename in os.listdir(INTENT_DIR):
        if filename.endswith(".json"):
            cache_key = filename[:-5]
            validate_storage_id(cache_key)
            data = _load_intent_file(cache_key)
            if data:
                stat = os.stat(os.path.join(INTENT_DIR, filename))
                result.append({
                    "cache_key": cache_key,
                    "name": data.get("name", cache_key),
                    "description": data.get("description", ""),
                    "intent_count": len(data.get("intents", [])),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "file_size": stat.st_size,
                })
    return result


def _read_json_file(base_dir: str, item_id: str, field_name: str) -> Optional[Dict[str, Any]]:
    path = safe_json_path(base_dir, item_id, field_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(base_dir: str, item_id: str, field_name: str, data: Dict[str, Any]) -> str:
    path = safe_json_path(base_dir, item_id, field_name)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)
    return path


def _list_json_files(base_dir: str, field_name: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(base_dir):
        return items
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
        item_id = filename[:-5]
        validate_storage_id(item_id, field_name)
        data = _read_json_file(base_dir, item_id, field_name)
        if data:
            items.append(data)
    return items


def _split_tags(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in ["|", ";", "，"]:
        text = text.replace(sep, ",")
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def _parse_eval_csv(raw: bytes) -> List[Dict[str, Any]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV 需要表头：query, expected_intent_id")
    cases: List[Dict[str, Any]] = []
    for idx, row in enumerate(reader, start=1):
        query = (row.get("query") or row.get("问题") or "").strip()
        expected = (
            row.get("expected_intent_id")
            or row.get("expected_intent")
            or row.get("intent_id")
            or row.get("期望意图ID")
            or row.get("期望意图")
            or ""
        ).strip()
        if not query and not expected:
            continue
        if not query or not expected:
            raise ValueError(f"第 {idx} 行缺少 query 或 expected_intent_id")
        cases.append({
            "case_id": f"case_{idx:04d}",
            "query": query,
            "expected_intent_id": expected,
            "expected_intent_name": (row.get("expected_intent_name") or row.get("意图名称") or "").strip(),
            "tags": _split_tags(row.get("tags") or row.get("标签")),
        })
    if not cases:
        raise ValueError("CSV 没有可用测试案例")
    return cases


def _list_eval_sets() -> List[Dict[str, Any]]:
    items = _list_json_files(EVAL_SET_DIR, "eval_set_id")
    return sorted(items, key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)


def _list_eval_runs() -> List[Dict[str, Any]]:
    items = _list_json_files(EVAL_RUN_DIR, "eval_run_id")
    return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)


def _get_eval_set(eval_set_id: str) -> Optional[Dict[str, Any]]:
    return _read_json_file(EVAL_SET_DIR, eval_set_id, "eval_set_id")


def _save_eval_set(eval_set: Dict[str, Any]) -> Dict[str, Any]:
    _write_json_file(EVAL_SET_DIR, eval_set["eval_set_id"], "eval_set_id", eval_set)
    return eval_set


def _get_eval_run(eval_run_id: str) -> Optional[Dict[str, Any]]:
    return _read_json_file(EVAL_RUN_DIR, eval_run_id, "eval_run_id")


def _save_eval_run(eval_run: Dict[str, Any]) -> Dict[str, Any]:
    _write_json_file(EVAL_RUN_DIR, eval_run["eval_run_id"], "eval_run_id", eval_run)
    return eval_run


def _delete_json_item(base_dir: str, item_id: str, field_name: str) -> bool:
    path = safe_json_path(base_dir, item_id, field_name)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def _delete_app_artifacts(app_id: str) -> Dict[str, Any]:
    """Keep the one-app-one-intent-tree model coherent after app deletion."""
    validate_storage_id(app_id, "app_id")
    artifacts: Dict[str, Any] = {
        "intent_tree_deleted": repository.delete_intent_tree(app_id),
        "intent_tree_versions_deleted": 0,
        "eval_sets_deleted": 0,
        "eval_runs_deleted": 0,
        "deployments_archived": 0,
        "runtime_bundles_unloaded": 0,
    }

    for version in repository.list_intent_tree_versions(app_id):
        if _delete_json_item(
            repository.intent_tree_version_dir,
            version["intent_tree_version_id"],
            "intent_tree_version_id",
        ):
            artifacts["intent_tree_versions_deleted"] += 1

    for eval_set in _list_eval_sets():
        if eval_set.get("app_id") == app_id and _delete_json_item(
            EVAL_SET_DIR,
            eval_set["eval_set_id"],
            "eval_set_id",
        ):
            artifacts["eval_sets_deleted"] += 1

    for eval_run in _list_eval_runs():
        if eval_run.get("app_id") == app_id and _delete_json_item(
            EVAL_RUN_DIR,
            eval_run["eval_run_id"],
            "eval_run_id",
        ):
            artifacts["eval_runs_deleted"] += 1

    for item in repository.list_deployments():
        if item.get("app_id") != app_id:
            continue
        deployment = Deployment.from_dict(item)
        if deployment.status != "inactive":
            deployment.status = "inactive"
            repository.save_deployment(deployment)
            artifacts["deployments_archived"] += 1

    artifacts["runtime_bundles_unloaded"] = runtime_registry.unload_app(app_id)
    return artifacts


def _runtime_loaded(deployment_id: str) -> bool:
    return any(bundle.get("deployment_id") == deployment_id for bundle in runtime_registry.list_bundles())


def _deployment_detail(deployment) -> Dict[str, Any]:
    tree_version = repository.get_intent_tree_version(deployment.intent_tree_version_id)
    strategy_config = repository.get_strategy_config(deployment.strategy_config_id)
    template = repository.get_strategy_template(strategy_config.strategy_template_id) if strategy_config else None
    merged_params = merge_template_params(template, strategy_config.params) if template and strategy_config else {}
    return {
        "deployment": deployment.to_dict(),
        "runtime_loaded": _runtime_loaded(deployment.deployment_id),
        "intent_tree_version": tree_version.to_dict() if tree_version else None,
        "strategy_config": strategy_config.to_dict() if strategy_config else None,
        "strategy_template": template.to_dict() if template else None,
        "merged_params": merged_params,
        "index_status": {
            "strategy_type": template.strategy_type if template else "",
            "hybrid_index_built_on_load": bool(template and template.strategy_type == "hybrid_retrieval"),
        },
    }


def _hydrate_eval_run(eval_run: Dict[str, Any]) -> Dict[str, Any]:
    tree_version = repository.get_intent_tree_version(eval_run.get("intent_tree_version_id", ""))
    strategy_config = repository.get_strategy_config(eval_run.get("strategy_config_id", "")) if eval_run.get("strategy_config_id") else None
    template = repository.get_strategy_template(eval_run.get("strategy_template_id", ""))
    return {
        **eval_run,
        "intent_tree_snapshot": tree_version.to_dict() if tree_version else None,
        "strategy_config_snapshot": strategy_config.to_dict() if strategy_config else None,
        "strategy_template_snapshot": template.to_dict() if template else None,
        "eval_set_snapshot": _get_eval_set(eval_run.get("eval_set_id", "")),
    }


UNKNOWN_INTENT_IDS = {"__unknown__", "unknown", "other", "none", "null", ""}


def _is_unknown_expected(intent_id: str) -> bool:
    return (intent_id or "").strip().lower() in UNKNOWN_INTENT_IDS


def _result_to_dict(res) -> Dict[str, Any]:
    return {
        "selected_intent_id": res.selected_intent_id,
        "selected_intent_name": res.selected_intent_name,
        "method": res.method,
        "confidence": float(res.confidence),
        "top_k": [(i, n, float(s)) for (i, n, s) in res.top_k],
        "is_ood": bool(res.is_ood),
        "is_ambiguous": bool(res.is_ambiguous),
    }


def _resolve_test_strategy(
    strategy_config_id: Optional[str],
    strategy_template_id: str,
    params_override: Dict[str, Any],
):
    strategy_config = None
    if strategy_config_id:
        strategy_config = repository.get_strategy_config(strategy_config_id)
        if strategy_config is None:
            raise ValueError(f"Strategy config not found: {strategy_config_id}")
        template = repository.get_strategy_template(strategy_config.strategy_template_id)
        params = {**strategy_config.params, **params_override}
    else:
        template = repository.get_strategy_template(strategy_template_id)
        if template is None:
            raise ValueError(f"Strategy template not found: {strategy_template_id}")
        params = params_override
    if template is None:
        raise ValueError("Strategy template not found")
    return strategy_config, template, merge_template_params(template, params)


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_or_build_test_strategy(
    intents: List[Dict[str, Any]],
    template,
    strategy_config: Optional[StrategyConfig],
    merged_params: Dict[str, Any],
) -> tuple[RouteStrategy, bool, str]:
    cache_key = ":".join([
        template.strategy_type,
        template.strategy_template_id,
        strategy_config.strategy_config_id if strategy_config else "template",
        _stable_hash(intents),
        _stable_hash(merged_params),
    ])
    with test_strategy_lock:
        cached = test_strategy_cache.get(cache_key)
        if cached is not None:
            return cached, True, cache_key

    strategy = build_strategy(
        template.strategy_type,
        intents,
        merged_params,
        StrategyBuildContext(provider=embedding_provider, llm_provider=llm_provider),
    )
    with test_strategy_lock:
        test_strategy_cache[cache_key] = strategy
        while len(test_strategy_cache) > TEST_STRATEGY_CACHE_MAX:
            test_strategy_cache.pop(next(iter(test_strategy_cache)))
    return strategy, False, cache_key


def _build_eval_strategy(req: EvalRunRequest):
    tree_version = repository.create_intent_tree_version(req.app_id, source="eval")
    strategy_config, template, merged_params = _resolve_test_strategy(
        req.strategy_config_id,
        req.strategy_template_id,
        req.params_override,
    )
    strategy = build_strategy(
        template.strategy_type,
        tree_version.intents,
        merged_params,
        StrategyBuildContext(provider=embedding_provider, llm_provider=llm_provider),
    )
    return tree_version, strategy_config, template, merged_params, strategy


# ----------------------------
# Admin API 端点：第一版最小闭环
# ----------------------------


@app.get("/admin/strategy-templates")
def list_strategy_templates():
    supported = {"hybrid_retrieval", "llm_prompt"}
    templates = [t for t in repository.list_strategy_templates() if t.get("strategy_type") in supported]
    return {"strategy_templates": templates}


@app.get("/admin/strategy-configs")
def list_strategy_configs():
    return {"strategy_configs": repository.list_strategy_configs()}


@app.post("/admin/strategy-configs")
def create_strategy_config(req: StrategyConfigRequest):
    config = StrategyConfig(
        strategy_config_id=req.strategy_config_id,
        name=req.name,
        strategy_template_id=req.strategy_template_id,
        description=req.description,
        params=req.params,
        resources=req.resources,
    )
    try:
        saved = repository.save_strategy_config(config)
        return saved.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/strategy-configs/{strategy_config_id}")
def get_strategy_config(strategy_config_id: str):
    config = repository.get_strategy_config(strategy_config_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Strategy config not found: {strategy_config_id}")
    template = repository.get_strategy_template(config.strategy_template_id)
    return {
        **config.to_dict(),
        "template": template.to_dict() if template else None,
        "merged_params": merge_template_params(template, config.params) if template else config.params,
    }


@app.put("/admin/strategy-configs/{strategy_config_id}")
def update_strategy_config(strategy_config_id: str, req: StrategyConfigUpdateRequest):
    config = repository.get_strategy_config(strategy_config_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Strategy config not found: {strategy_config_id}")
    if req.name is not None:
        config.name = req.name
    if req.description is not None:
        config.description = req.description
    if req.params is not None:
        config.params = req.params
    if req.resources is not None:
        config.resources = req.resources
    config.version += 1
    return repository.save_strategy_config(config).to_dict()


@app.get("/admin/intent-trees")
def list_intent_trees():
    return {"intent_trees": repository.list_intent_trees()}


@app.post("/admin/intent-trees")
def create_intent_tree(req: IntentTreeRequest):
    validate_storage_id(req.intent_tree_id, "intent_tree_id")
    tree = IntentTree(
        intent_tree_id=req.intent_tree_id,
        name=req.name,
        description=req.description,
        intents=req.intents,
    )
    repository.save_intent_tree(tree)
    version = repository.create_intent_tree_version(req.intent_tree_id, source="create")
    return {"intent_tree": tree.to_dict(), "latest_version": version.to_dict()}


@app.get("/admin/intent-trees/{intent_tree_id}")
def get_intent_tree(intent_tree_id: str):
    tree = repository.get_intent_tree(intent_tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"Intent tree not found: {intent_tree_id}")
    return {**tree.to_dict(), "versions": repository.list_intent_tree_versions(intent_tree_id)}


@app.put("/admin/intent-trees/{intent_tree_id}")
def update_intent_tree(intent_tree_id: str, req: IntentTreeUpdateRequest):
    tree = repository.get_intent_tree(intent_tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"Intent tree not found: {intent_tree_id}")
    if req.name is not None:
        tree.name = req.name
    if req.description is not None:
        tree.description = req.description
    if req.intents is not None:
        tree.intents = req.intents
    if req.status is not None:
        tree.status = req.status
    repository.save_intent_tree(tree)
    version = repository.create_intent_tree_version(intent_tree_id, source="update")
    return {"intent_tree": tree.to_dict(), "latest_version": version.to_dict()}


@app.post("/admin/intent-trees/{intent_tree_id}/versions")
def create_intent_tree_version(intent_tree_id: str):
    try:
        return repository.create_intent_tree_version(intent_tree_id, source="manual").to_dict()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/admin/intent-tree-versions/{version_id}")
def get_intent_tree_version(version_id: str):
    version = repository.get_intent_tree_version(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Intent tree version not found: {version_id}")
    return version.to_dict()


@app.post("/admin/test-route")
def test_strategy_route(req: TestStrategyRouteRequest):
    try:
        started_total = time.perf_counter()
        if req.intent_tree_version_id:
            tree_version = repository.get_intent_tree_version(req.intent_tree_version_id)
            if tree_version is None:
                raise ValueError("Intent tree version not found")
            intents = tree_version.intents
            tree_ref = tree_version.intent_tree_version_id
        elif req.intent_tree_id:
            tree = repository.get_intent_tree(req.intent_tree_id)
            if tree is None:
                raise ValueError("Intent tree not found")
            intents = tree.intents
            tree_ref = f"{tree.intent_tree_id}:current:{_stable_hash(intents)}"
        else:
            raise ValueError("intent_tree_version_id or intent_tree_id is required")

        strategy_config, template, merged_params = _resolve_test_strategy(
            req.strategy_config_id,
            req.strategy_template_id,
            req.params_override,
        )
        started_build = time.perf_counter()
        strategy, cache_hit, strategy_cache_key = _get_or_build_test_strategy(
            intents,
            template,
            strategy_config,
            merged_params,
        )
        build_latency_ms = round((time.perf_counter() - started_build) * 1000, 2)
        started_route = time.perf_counter()
        res = strategy.route(req.query, visible_intent_ids=req.visible_intent_ids)
        route_latency_ms = round((time.perf_counter() - started_route) * 1000, 2)
        return {
            "query": req.query,
            "intent_tree_version_id": tree_ref,
            "strategy_template_id": template.strategy_template_id,
            "strategy_config_id": strategy_config.strategy_config_id if strategy_config else None,
            "merged_params": merged_params,
            "perf": {
                "strategy_cache_hit": cache_hit,
                "strategy_cache_key": strategy_cache_key,
                "build_latency_ms": build_latency_ms,
                "route_latency_ms": route_latency_ms,
                "total_latency_ms": round((time.perf_counter() - started_total) * 1000, 2),
            },
            "result": _result_to_dict(res),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/eval-sets")
def list_eval_sets():
    return {"eval_sets": _list_eval_sets()}


@app.get("/admin/eval-sets/{eval_set_id}")
def get_eval_set(eval_set_id: str):
    eval_set = _get_eval_set(eval_set_id)
    if eval_set is None:
        raise HTTPException(status_code=404, detail=f"Eval set not found: {eval_set_id}")
    return eval_set


@app.delete("/admin/eval-sets/{eval_set_id}")
def delete_eval_set(eval_set_id: str):
    path = safe_json_path(EVAL_SET_DIR, eval_set_id, "eval_set_id")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Eval set not found: {eval_set_id}")
    os.remove(path)
    return {"eval_set_id": eval_set_id, "deleted": True}


@app.post("/admin/eval-sets/upload-csv")
async def upload_eval_set_csv(
    app_id: str = Form(...),
    name: str = Form(""),
    file: UploadFile = File(...),
):
    try:
        validate_storage_id(app_id, "app_id")
        if _load_intent_file(app_id) is None and repository.get_intent_tree(app_id) is None:
            raise ValueError(f"App not found: {app_id}")
        raw = await file.read()
        cases = _parse_eval_csv(raw)
        now = datetime.now().isoformat()
        eval_set_id = repository.make_id("eval")
        display_name = name.strip() or os.path.splitext(file.filename or "")[0] or eval_set_id
        eval_set = {
            "eval_set_id": eval_set_id,
            "name": display_name,
            "app_id": app_id,
            "case_count": len(cases),
            "cases": cases,
            "created_at": now,
            "updated_at": now,
        }
        return _save_eval_set(eval_set)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/eval-runs")
def list_eval_runs():
    runs = _list_eval_runs()
    compact = []
    for run in runs:
        compact.append({k: v for k, v in run.items() if k not in {"case_results", "failed_cases", "params_snapshot"}})
    return {"eval_runs": compact}


@app.get("/admin/eval-runs/{eval_run_id}")
def get_eval_run(eval_run_id: str):
    eval_run = _get_eval_run(eval_run_id)
    if eval_run is None:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {eval_run_id}")
    return _hydrate_eval_run(eval_run)


@app.post("/admin/eval-runs")
def create_eval_run(req: EvalRunRequest):
    try:
        validate_storage_id(req.app_id, "app_id")
        eval_set = _get_eval_set(req.eval_set_id)
        if eval_set is None:
            raise ValueError(f"Eval set not found: {req.eval_set_id}")
        if eval_set.get("app_id") and eval_set.get("app_id") != req.app_id:
            raise ValueError("评测集所属应用与当前选择应用不一致")

        tree_version, strategy_config, template, merged_params, strategy = _build_eval_strategy(req)
        case_results: List[Dict[str, Any]] = []
        latencies: List[float] = []
        total = 0
        passed = 0
        top3_passed = 0
        ambiguous_count = 0
        ood_expected_total = 0
        ood_expected_passed = 0
        tag_stats: Dict[str, Dict[str, int]] = {}

        for case in eval_set.get("cases", []):
            total += 1
            expected_id = (case.get("expected_intent_id") or "").strip()
            expected_unknown = _is_unknown_expected(expected_id)
            started = time.perf_counter()
            error = ""
            result: Dict[str, Any] = {}
            try:
                res = strategy.route(case.get("query", ""))
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                result = _result_to_dict(res)
                predicted_id = result.get("selected_intent_id") or "__unknown__"
                top_ids = [item[0] for item in result.get("top_k", [])]
                is_pass = bool(result.get("is_ood")) if expected_unknown else predicted_id == expected_id
                is_top3_pass = is_pass if expected_unknown else expected_id in top_ids[:3]
                ambiguous_count += 1 if result.get("is_ambiguous") else 0
                if expected_unknown:
                    ood_expected_total += 1
                    ood_expected_passed += 1 if is_pass else 0
            except Exception as e:
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                predicted_id = ""
                is_pass = False
                is_top3_pass = False
                error = str(e)
            latencies.append(latency_ms)
            passed += 1 if is_pass else 0
            top3_passed += 1 if is_top3_pass else 0
            for tag in case.get("tags", []) or ["未分组"]:
                bucket = tag_stats.setdefault(tag, {"total": 0, "passed": 0})
                bucket["total"] += 1
                bucket["passed"] += 1 if is_pass else 0
            case_results.append({
                "case_id": case.get("case_id"),
                "query": case.get("query", ""),
                "expected_intent_id": expected_id,
                "expected_intent_name": case.get("expected_intent_name", ""),
                "predicted_intent_id": predicted_id,
                "passed": is_pass,
                "top3_passed": is_top3_pass,
                "latency_ms": latency_ms,
                "tags": case.get("tags", []),
                "error": error,
                "result": result,
            })

        sorted_latencies = sorted(latencies)
        p95_idx = min(len(sorted_latencies) - 1, max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)) if sorted_latencies else 0
        tag_metrics = {
            tag: {
                **stats,
                "accuracy": round(stats["passed"] / stats["total"], 4) if stats["total"] else 0,
            }
            for tag, stats in tag_stats.items()
        }
        metrics = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": round(passed / total, 4) if total else 0,
            "top3_accuracy": round(top3_passed / total, 4) if total else 0,
            "ood_accuracy": round(ood_expected_passed / ood_expected_total, 4) if ood_expected_total else None,
            "ambiguous_rate": round(ambiguous_count / total, 4) if total else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_latency_ms": sorted_latencies[p95_idx] if sorted_latencies else 0,
            "tags": tag_metrics,
        }
        now = datetime.now().isoformat()
        eval_run = {
            "eval_run_id": repository.make_id("run"),
            "app_id": req.app_id,
            "eval_set_id": req.eval_set_id,
            "eval_set_name": eval_set.get("name", req.eval_set_id),
            "intent_tree_version_id": tree_version.intent_tree_version_id,
            "strategy_type": template.strategy_type,
            "strategy_template_id": template.strategy_template_id,
            "strategy_config_id": strategy_config.strategy_config_id if strategy_config else None,
            "strategy_name": strategy_config.name if strategy_config else template.name,
            "params_snapshot": merged_params,
            "metrics": metrics,
            "failed_cases": [item for item in case_results if not item["passed"]],
            "case_results": case_results,
            "created_at": now,
        }
        return _save_eval_run(eval_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/deployments")
def publish_deployment(req: PublishRequest):
    try:
        validate_storage_id(req.app_id, "app_id")
        validate_storage_id(req.environment, "environment")
        if req.intent_tree_id and req.intent_tree_id != req.app_id:
            raise ValueError("app_id must match intent_tree_id because each app owns exactly one intent tree")
        if req.intent_tree_version_id:
            tree_version = repository.get_intent_tree_version(req.intent_tree_version_id)
            if tree_version is None:
                raise ValueError(f"Intent tree version not found: {req.intent_tree_version_id}")
            if tree_version.intent_tree_id != req.app_id:
                raise ValueError("app_id must match the intent_tree_id of intent_tree_version_id")

        strategy_config_id = req.strategy_config_id
        if not strategy_config_id:
            raise ValueError("发布必须选择已保存的策略配置；请先在策略配置页基于模板创建配置")
        if repository.get_strategy_config(strategy_config_id) is None:
            raise ValueError(f"Strategy config not found: {strategy_config_id}")

        version_id = req.intent_tree_version_id
        if not version_id:
            if not req.intent_tree_id:
                raise ValueError("intent_tree_id or intent_tree_version_id is required")
            version_id = repository.create_intent_tree_version(req.intent_tree_id, source="publish", created_by=req.published_by).intent_tree_version_id

        deployment = repository.publish(
            app_id=req.app_id,
            environment=req.environment,
            intent_tree_version_id=version_id,
            strategy_config_id=strategy_config_id,
            published_by=req.published_by,
            tenant_id=req.tenant_id or "default",
        )
        local_bundle = runtime_registry.reload_deployment(deployment)
        router_reload = _call_router_admin("POST", f"/admin/reload-deployment/{deployment.deployment_id}") if AUTO_PUBLISH_TO_ROUTER else None
        return {"deployment": deployment.to_dict(), "local_runtime": local_bundle.summary(), "router_reload": router_reload}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/deployments")
def list_deployments():
    return {"deployments": repository.list_deployments(), "runtime_bundles": runtime_registry.list_bundles()}


@app.get("/admin/deployments/{deployment_id}")
def get_deployment_detail(deployment_id: str):
    deployment = repository.get_deployment(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")
    return _deployment_detail(deployment)


@app.post("/admin/deployments/{deployment_id}/rollback")
def rollback_deployment(deployment_id: str):
    try:
        target = repository.get_deployment(deployment_id)
        if target is None:
            raise ValueError(f"Deployment not found: {deployment_id}")
        deployment = repository.publish(
            app_id=target.app_id,
            environment=target.environment,
            intent_tree_version_id=target.intent_tree_version_id,
            strategy_config_id=target.strategy_config_id,
            tenant_id=target.tenant_id,
            published_by="rollback",
        )
        local_bundle = runtime_registry.reload_deployment(deployment)
        router_reload = _call_router_admin("POST", f"/admin/reload-deployment/{deployment.deployment_id}") if AUTO_PUBLISH_TO_ROUTER else None
        return {
            "rolled_back_from": target.to_dict(),
            "deployment": deployment.to_dict(),
            "local_runtime": local_bundle.summary(),
            "router_reload": router_reload,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/published-route")
def admin_published_route(req: PublishedRouteRequest):
    try:
        res, bundle = runtime_registry.route(
            tenant_id=req.tenant_id or "default",
            app_id=req.app_id,
            environment=req.environment,
            query=req.query,
            visible_intent_ids=req.visible_intent_ids,
        )
        return {
            "selected_intent_id": res.selected_intent_id,
            "selected_intent_name": res.selected_intent_name,
            "method": res.method,
            "confidence": float(res.confidence),
            "top_k": [(i, n, float(s)) for (i, n, s) in res.top_k],
            "is_ood": bool(res.is_ood),
            "is_ambiguous": bool(res.is_ambiguous),
            "deployment": bundle.summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ----------------------------
# Admin API 端点（兼容旧版 cache_key 配置）
# ----------------------------


@app.get("/health")
def health():
    return {"ok": True, "service": "intent-router-admin"}


@app.get("/health/detail")
def health_detail():
    return {
        "status": "healthy",
        "environment": APP_ENV,
        "embedding_provider": EMBEDDING_PROVIDER,
        "router_service_url": ROUTER_SERVICE_URL,
        "auto_publish": AUTO_PUBLISH_TO_ROUTER,
        "intent_count": len(_list_intent_files()),
        "cache_count": len(intent_cache.list_caches()),
        "runtime_bundle_count": len(runtime_registry.list_bundles()),
        "deployment_count": len(repository.list_deployments()),
        "llm_provider_configured": llm_provider is not None,
    }


@app.get("/admin/apps")
def list_apps():
    apps = []
    for item in _list_intent_files():
        app = {k: v for k, v in item.items() if k != "cache_key"}
        app["app_id"] = item.get("cache_key")
        apps.append(app)
    return {"apps": apps, "count": len(apps)}


@app.get("/admin/apps/{app_id}")
def get_app(app_id: str):
    data = _load_intent_file(app_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"App intent tree not found: {app_id}")
    return {**data, "app_id": data.get("cache_key", app_id)}


@app.post("/admin/apps")
def create_app(req: CreateAppRequest):
    legacy_req = CreateIntentRequest(
        cache_key=req.app_id,
        name=req.name,
        description=req.description,
        intents=req.intents,
    )
    result = create_intent(legacy_req)
    return {"app_id": result["cache_key"], "saved": result["saved"], "sync": result.get("sync"), "publish": result.get("publish")}


@app.put("/admin/apps/{app_id}")
def update_app(app_id: str, req: UpdateAppRequest):
    result = update_intent(app_id, UpdateIntentRequest(name=req.name, description=req.description, intents=req.intents))
    return {"app_id": result["cache_key"], "updated": result["updated"], "sync": result.get("sync"), "publish": result.get("publish")}


@app.delete("/admin/apps/{app_id}")
def delete_app(app_id: str):
    result = delete_intent(app_id)
    return {
        "app_id": result["cache_key"],
        "deleted": result["deleted"],
        "artifacts": result.get("artifacts"),
        "publish": result.get("publish"),
    }


@app.post("/admin/apps/test")
def test_app_route(req: TestAppRouteRequest):
    result = test_route(TestRouteRequest(
        cache_key=req.app_id,
        query=req.query,
        abs_threshold=req.abs_threshold,
        gap_threshold=req.gap_threshold,
    ))
    return {**result, "app_id": result.get("cache_key", req.app_id)}


@app.get("/admin/intents")
def list_intents():
    """列出所有意图配置"""
    return {"intents": _list_intent_files(), "count": len(os.listdir(INTENT_DIR))}


@app.get("/admin/intents/{cache_key}")
def get_intent(cache_key: str):
    """获取指定意图配置"""
    data = _load_intent_file(cache_key)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Intent config not found: {cache_key}")
    return data


@app.post("/admin/intents")
def create_intent(req: CreateIntentRequest):
    """创建新意图配置"""
    validate_storage_id(req.cache_key)
    # 检查是否已存在
    if os.path.exists(safe_json_path(INTENT_DIR, req.cache_key)):
        raise HTTPException(status_code=409, detail=f"Intent config already exists: {req.cache_key}")

    now = datetime.now().isoformat()
    data = {
        "name": req.name,
        "description": req.description,
        "cache_key": req.cache_key,
        "created_at": now,
        "updated_at": now,
        "intents": req.intents,
    }

    _save_intent_file(req.cache_key, data)
    synced = _sync_intent_tree_from_legacy(req.cache_key, data, source="legacy_create")

    # 预热缓存
    try:
        intent_defs = intent_defs_from_dicts(req.intents)
        intent_cache.get_or_build(req.cache_key, intent_defs)
    except Exception as e:
        # 保存成功但缓存预热失败，不影响主流程
        pass

    return {"cache_key": req.cache_key, "saved": True, "sync": synced, "publish": _maybe_publish_cache_key(req.cache_key)}


@app.put("/admin/intents/{cache_key}")
def update_intent(cache_key: str, req: UpdateIntentRequest):
    """更新意图配置"""
    data = _load_intent_file(cache_key)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Intent config not found: {cache_key}")

    if req.name is not None:
        data["name"] = req.name
    if req.description is not None:
        data["description"] = req.description
    if req.intents is not None:
        data["intents"] = req.intents

    data["updated_at"] = datetime.now().isoformat()
    _save_intent_file(cache_key, data)
    synced = _sync_intent_tree_from_legacy(cache_key, data, source="legacy_update")

    # 使缓存失效，重新构建
    intent_cache.invalidate(cache_key)
    if req.intents:
        try:
            intent_defs = intent_defs_from_dicts(req.intents)
            intent_cache.get_or_build(cache_key, intent_defs)
        except Exception:
            pass

    return {"cache_key": cache_key, "updated": True, "sync": synced, "publish": _maybe_publish_cache_key(cache_key)}


@app.delete("/admin/intents/{cache_key}")
def delete_intent(cache_key: str):
    """删除意图配置"""
    file_path = safe_json_path(INTENT_DIR, cache_key)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Intent config not found: {cache_key}")

    os.remove(file_path)
    intent_cache.invalidate(cache_key)
    artifacts = _delete_app_artifacts(cache_key)

    publish = _publish_all() if AUTO_PUBLISH_TO_ROUTER else None
    return {"cache_key": cache_key, "deleted": True, "artifacts": artifacts, "publish": publish}


@app.post("/admin/test")
def test_route(req: TestRouteRequest):
    """测试意图路由"""
    # 尝试从缓存或磁盘加载意图
    intent_defs = None

    # 先尝试从缓存获取
    cached_router = intent_cache.get(req.cache_key)
    if cached_router:
        intent_defs = cached_router.intents

    # 如果缓存没有，尝试从磁盘加载
    if intent_defs is None:
        intent_data = _load_intent_file(req.cache_key)
        if intent_data:
            intent_defs = intent_defs_from_dicts(intent_data.get("intents", []))

    if intent_defs is None:
        raise HTTPException(
            status_code=404,
            detail=f"No intent config found for cache_key: {req.cache_key}"
        )

    # 确保缓存中有最新的意图配置
    intent_cache.get_or_build(req.cache_key, intent_defs)

    # 执行路由
    res = intent_cache.route(
        cache_key=req.cache_key,
        query=req.query,
        abs_threshold=req.abs_threshold,
        gap_threshold=req.gap_threshold,
    )

    return {
        "query": req.query,
        "cache_key": req.cache_key,
        "result": {
            "selected_intent_id": res.selected_intent_id,
            "selected_intent_name": res.selected_intent_name,
            "method": res.method,
            "confidence": float(res.confidence),
            "top_k": [(i, n, float(s)) for (i, n, s) in res.top_k],
            "is_ood": bool(res.is_ood),
            "is_ambiguous": bool(res.is_ambiguous),
        }
    }


@app.get("/admin/cache")
def list_cache():
    """列出缓存状态"""
    caches = intent_cache.list_caches()
    return {"caches": caches, "count": len(caches)}


@app.delete("/admin/cache/{cache_key}")
def invalidate_cache(cache_key: str):
    """使指定 cache_key 的缓存失效"""
    count = intent_cache.invalidate(cache_key)
    return {"cache_key": cache_key, "invalidated": count}


@app.get("/admin/cache/clear")
def clear_cache():
    """清空所有缓存"""
    caches = intent_cache.list_caches()
    for c in caches:
        intent_cache.invalidate(c["cache_key"])
    return {"cleared": len(caches)}


@app.post("/admin/publish/{cache_key}")
def publish_intent(cache_key: str):
    """发布单个意图配置和当前策略到意图识别服务。"""
    if _load_intent_file(cache_key) is None:
        raise HTTPException(status_code=404, detail=f"Intent config not found: {cache_key}")
    return _maybe_publish_cache_key(cache_key) or {"ok": False, "detail": "AUTO_PUBLISH_TO_ROUTER is disabled"}


@app.post("/admin/publish-all")
def publish_all():
    """发布所有意图配置和当前策略到意图识别服务。"""
    return _publish_all()


@app.post("/admin/publish-runtime-config")
def publish_runtime_config():
    """只发布当前策略配置到意图识别服务。"""
    return _publish_runtime_config()


# ----------------------------
# 配置管理 API
# ----------------------------


@app.get("/admin/config")
def get_router_config():
    """获取策略配置"""
    return {"config": router_config_dict}


@app.put("/admin/config")
def update_router_config(req: RouterConfigModel):
    """更新策略配置"""
    global router_config, router_config_dict, intent_cache

    router_config_dict = {
        "abs_threshold": req.abs_threshold,
        "gap_threshold": req.gap_threshold,
        "top_k": req.top_k,
        "weight_intro": req.weight_intro,
        "alpha_short": req.alpha_short,
        "alpha_long": req.alpha_long,
        "enable_bm25": req.enable_bm25,
        "short_query_len": req.short_query_len,
        "question_top_k": req.question_top_k,
        "example_question_threshold": req.example_question_threshold,
    }

    # 更新全局配置
    admin_config["router_config"] = router_config_dict
    _save_admin_config(admin_config)

    # 保存配置只失效本地旧索引；首次测试或发布具体应用时再按需构建。
    router_config = _build_router_config_from_dict(router_config_dict)
    intent_cache = IntentCache(
        provider=embedding_provider,
        config=router_config,
        persist_dir=INTENT_DIR,
    )

    return {"updated": True, "config": router_config_dict, "publish": _publish_runtime_config() if AUTO_PUBLISH_TO_ROUTER else None}


@app.get("/admin/llm-config")
def get_llm_config():
    """获取 LLM 配置"""
    # 不返回 api_key 的完整内容
    config = llm_config_dict.copy()
    if config.get("api_key"):
        config["api_key"] = config["api_key"][:8] + "***" + config["api_key"][-4:] if len(config["api_key"]) > 12 else "***"
    return {"config": config, "has_key": bool(llm_config_dict.get("api_key"))}


@app.put("/admin/llm-config")
def update_llm_config(req: LLMConfigModel):
    """更新 LLM 配置"""
    global llm_provider, llm_config_dict

    llm_config_dict = {
        "provider": req.provider,
        "api_key": req.api_key or llm_config_dict.get("api_key", ""),
        "base_url": req.base_url,
        "model": req.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "timeout": req.timeout,
        "min_confidence": req.min_confidence,
    }

    # 保存到文件（api_key 会保存）
    admin_config["llm_config"] = llm_config_dict
    _save_admin_config(admin_config)

    # 重建 LLM Provider
    llm_provider = _build_llm_provider(llm_config_dict)
    runtime_registry.llm_provider = llm_provider

    return {"updated": True, "publish": _publish_runtime_config() if AUTO_PUBLISH_TO_ROUTER else None}


@app.get("/")
def admin_root():
    """重定向到管理页面"""
    return HTMLResponse("""
    <html>
    <head><meta http-equiv="refresh" content="0;url=/admin/"></head>
    <body>Redirecting to <a href="/admin/">Admin Console</a>...</body>
    </html>
    """)


@app.get("/admin/")
def admin_page():
    """运营平台管理页面"""
    return HTMLResponse(_ADMIN_HTML)


@app.get("/admin/app.js")
def admin_js():
    """运营平台 JavaScript"""
    return HTMLResponse(_ADMIN_JS, media_type="application/javascript")


@app.get("/admin/style.css")
def admin_css():
    """运营平台样式"""
    return HTMLResponse(_ADMIN_CSS, media_type="text/css")


# ----------------------------
# 嵌入的前端代码
# ----------------------------

_ADMIN_HTML = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>意图路由器 - 运营平台</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/admin/style.css">
</head>
<body>
    <div class="bg-grid"></div>
    <div class="container">
        <header class="header">
            <div class="header-brand">
                <svg class="header-logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                <div>
                    <h1>意图路由器</h1>
                    <span class="header-subtitle">应用意图路由运营控制台</span>
                </div>
            </div>
        </header>

        <nav class="tabs">
            <button class="tab-btn active" data-tab="intents">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
                应用意图树
            </button>
            <button class="tab-btn" data-tab="config">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
                策略配置
            </button>
            <button class="tab-btn" data-tab="test">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                在线测试
            </button>
            <button class="tab-btn" data-tab="release">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/><path d="M5 5v14"/></svg>
                发布中心
            </button>
        </nav>

        <main class="content">
            <!-- 意图配置 Tab -->
            <div id="tab-intents" class="tab-panel active">
                <div class="panel">
                    <div class="panel-header">
                        <h2>应用意图树列表</h2>
                        <button onclick="showCreateModal()" class="btn btn-primary">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
                            新建应用
                        </button>
                    </div>
                    <div id="intent-list" class="intent-grid"></div>
                </div>

            </div>

            <!-- 策略配置 Tab -->
            <div id="tab-config" class="tab-panel">
                <div class="panel">
                    <div class="panel-header">
                        <h2>可发布策略配置</h2>
                        <button onclick="openStrategyConfigEditor()" class="btn btn-primary">新建策略配置</button>
                    </div>
                    <p class="release-hint">策略模板只作为创建配置的起点；在线测试、批量评测和发布都使用这里保存的策略配置。</p>
                    <div class="release-columns">
                        <div>
                            <h3 class="section-subtitle">策略模板</h3>
                            <div id="strategy-template-list" class="compact-list"></div>
                        </div>
                        <div>
                            <h3 class="section-subtitle">策略配置</h3>
                            <div id="strategy-config-list" class="compact-list"></div>
                        </div>
                    </div>
                    <div id="strategy-config-editor" class="eval-result"></div>
                </div>
                <div class="panel">
                    <h2>系统默认混合检索参数</h2>
                    <p class="release-hint">仅作为内置模板和旧接口的默认值；业务测试和发布请优先使用上方的策略配置。</p>
                    <div class="config-grid" id="config-form">
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">abs_threshold</span>
                                <span class="config-badge">OOD 判定</span>
                            </div>
                            <input type="number" id="cfg-abs-threshold" step="0.01" min="0" max="1">
                            <p class="config-desc">相似度低于此值判定为未知意图</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">gap_threshold</span>
                                <span class="config-badge">歧义检测</span>
                            </div>
                            <input type="number" id="cfg-gap-threshold" step="0.01" min="0" max="1">
                            <p class="config-desc">Top-1 与 Top-2 差值小于此值时判定为歧义</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">top_k</span>
                                <span class="config-badge">输出</span>
                            </div>
                            <input type="number" id="cfg-top-k" step="1" min="1" max="10">
                            <p class="config-desc">返回的候选意图数量</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">weight_intro</span>
                                <span class="config-badge">权重</span>
                            </div>
                            <input type="number" id="cfg-weight-intro" step="0.1" min="0" max="1">
                            <p class="config-desc">意图描述的权重（示例问题权重 = 1 - 此值）</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">alpha_short</span>
                                <span class="config-badge">短查询</span>
                            </div>
                            <input type="number" id="cfg-alpha-short" step="0.1" min="0" max="1">
                            <p class="config-desc">查询字数 &lt; short_query_len 时的语义权重</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">alpha_long</span>
                                <span class="config-badge">长查询</span>
                            </div>
                            <input type="number" id="cfg-alpha-long" step="0.1" min="0" max="1">
                            <p class="config-desc">查询字数 ≥ short_query_len 时的语义权重</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">short_query_len</span>
                                <span class="config-badge">分界线</span>
                            </div>
                            <input type="number" id="cfg-short-query-len" step="1" min="1" max="50">
                            <p class="config-desc">短查询与长查询的字符长度分界线</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">question_top_k</span>
                                <span class="config-badge">Top-K</span>
                            </div>
                            <input type="number" id="cfg-question-top-k" step="1" min="1" max="20">
                            <p class="config-desc">取最高的前 K 个示例问题相似度求均值</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">example_question_threshold</span>
                                <span class="config-badge">直连</span>
                            </div>
                            <input type="number" id="cfg-example-question-threshold" step="0.01" min="0" max="1">
                            <p class="config-desc">示例问题相似度高于此值时直接命中意图</p>
                        </div>
                        <div class="config-card">
                            <div class="config-header">
                                <span class="config-name">enable_bm25</span>
                                <span class="config-badge">开关</span>
                            </div>
                            <select id="cfg-enable-bm25">
                                <option value="true">启用</option>
                                <option value="false">禁用</option>
                            </select>
                            <p class="config-desc">是否启用 BM25 混合检索</p>
                        </div>
                    </div>
                    <div class="form-actions">
                        <button onclick="saveConfig()" class="btn btn-accent">保存系统默认配置</button>
                        <span id="config-status" class="status-msg"></span>
                    </div>
                </div>
                <div class="panel">
                    <h2>大模型连接配置</h2>
                    <div class="llm-form">
                        <div class="form-row">
                            <div class="form-group">
                                <label>Provider</label>
                                <select id="llm-provider" onchange="onLlmProviderChange()">
                                    <option value="deepseek">DeepSeek</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="custom">OpenAI Compatible</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>模型</label>
                                <input type="text" id="llm-model" placeholder="deepseek-chat">
                            </div>
                        </div>
                        <div class="form-group">
                            <label>API Key</label>
                            <input type="password" id="llm-api-key" placeholder="输入 API Key">
                        </div>
                        <div class="form-group">
                            <label>Base URL</label>
                            <input type="text" id="llm-base-url" placeholder="https://api.deepseek.com/v1">
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label>Max Tokens</label>
                                <input type="number" id="llm-max-tokens" step="1" min="64" max="4096" value="256">
                            </div>
                            <div class="form-group">
                                <label>Temperature</label>
                                <input type="number" id="llm-temperature" step="0.1" min="0" max="2" value="0.1">
                            </div>
                            <div class="form-group">
                                <label>Timeout</label>
                                <input type="number" id="llm-timeout" step="1" min="1" max="60" value="5">
                            </div>
                        </div>
                    </div>
                    <div class="form-actions">
                        <button onclick="saveLlmConfig()" class="btn btn-accent">保存大模型配置</button>
                        <span id="llm-status" class="status-msg"></span>
                    </div>
                </div>
            </div>

            <!-- 在线测试 Tab -->
            <div id="tab-test" class="tab-panel">
                <div class="panel">
                    <h2>在线测试</h2>
                    <div class="test-form">
                        <div class="form-row">
                            <div class="form-group">
                                <label>选择应用</label>
                                <select id="test-cache-key"></select>
                            </div>
                            <div class="form-group">
                                <label>测试策略</label>
                                <select id="test-strategy-select"></select>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>测试查询</label>
                            <textarea id="test-query" rows="3" placeholder="输入要测试的查询..."></textarea>
                        </div>
                        <button onclick="testRoute()" class="btn btn-accent">执行测试</button>
                        <div id="test-result" class="result-panel"></div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-header">
                        <h2>批量评测</h2>
                        <button onclick="loadEvalData()" class="btn btn-secondary">刷新评测数据</button>
                    </div>
                    <div class="batch-layout">
                        <div class="batch-card">
                            <h3>上传评测集</h3>
                            <div class="form-group">
                                <label>评测应用</label>
                                <select id="batch-app-id"></select>
                            </div>
                            <div class="form-group">
                                <label>评测集名称</label>
                                <input type="text" id="eval-set-name" placeholder="如 退款场景核心问题">
                            </div>
                            <div class="form-group">
                                <label>CSV 文件</label>
                                <input type="file" id="eval-csv-file" accept=".csv,text/csv">
                                <span class="file-hint">表头支持 query, expected_intent_id, expected_intent_name, tags</span>
                            </div>
                            <button onclick="uploadEvalSet()" class="btn btn-accent">上传并保存</button>
                            <span id="eval-upload-status" class="status-msg"></span>
                        </div>
                        <div class="batch-card">
                            <h3>执行批量测试</h3>
                            <div class="form-group">
                                <label>评测集</label>
                                <select id="batch-eval-set"></select>
                            </div>
                            <div class="form-group">
                                <label>测试策略</label>
                                <select id="batch-strategy-select"></select>
                            </div>
                            <p class="release-hint">批量评测会保存应用、策略、参数快照、响应时间、正确率和失败案例。</p>
                            <button onclick="runBatchEval()" class="btn btn-accent">开始评测</button>
                        </div>
                    </div>
                    <div id="batch-result" class="eval-result"></div>
                </div>
                <div class="panel">
                    <h2>评测记录</h2>
                    <div id="eval-run-list" class="compact-list"></div>
                </div>
            </div>

            <!-- 发布 Tab -->
            <div id="tab-release" class="tab-panel">
                <div class="panel">
                    <div class="panel-header">
                        <h2>发布中心</h2>
                        <button onclick="loadReleaseData()" class="btn btn-secondary">刷新发布数据</button>
                    </div>
                    <div class="release-card">
                        <h3>选择发布对象</h3>
                        <div class="form-row">
                            <div class="form-group">
                                <label>应用 ID（随意图树自动确定）</label>
                                <input type="text" id="release-app-id" placeholder="选择意图树后自动填充" readonly>
                            </div>
                            <div class="form-group">
                                <label>发布环境</label>
                                <select id="release-environment">
                                    <option value="prod">线上环境 (prod)</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label>意图树</label>
                                <select id="release-intent-tree" onchange="onReleaseIntentTreeChange()"></select>
                            </div>
                            <div class="form-group">
                                <label>策略配置</label>
                                <select id="release-strategy-config"></select>
                            </div>
                        </div>
                        <p class="release-hint">发布会固定当前意图树版本和策略配置快照；如需验证效果，请先在“在线测试”完成单条或批量评测。</p>
                        <div class="form-actions">
                            <button onclick="publishDeployment()" class="btn btn-accent">发布并热更新</button>
                            <span id="release-status" class="status-msg"></span>
                        </div>
                    </div>
                </div>
                <div class="panel">
                    <h2>发布历史 / 运行时</h2>
                    <div id="deployment-list" class="compact-list"></div>
                    <div id="deployment-detail" class="eval-result"></div>
                </div>
            </div>

        </main>

        <footer class="footer">
            <span>意图路由器运营平台 v1.0</span>
        </footer>
    </div>

    <!-- Modal -->
    <div id="modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-title">新建意图配置</h2>
                <button onclick="closeModal()" class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>应用 ID (app_id)</label>
                    <input type="text" id="modal-cache-key" placeholder="唯一应用标识，如: refund_router">
                </div>
                <div class="form-group">
                    <label>名称</label>
                    <input type="text" id="modal-name" placeholder="配置名称">
                </div>
                <div class="form-group">
                    <label>描述</label>
                    <input type="text" id="modal-description" placeholder="配置描述">
                </div>
                <div class="form-group">
                    <label>意图定义 (JSON)</label>
                    <textarea id="modal-intents" rows="10" placeholder="[&#10;  {&#10;    &quot;id&quot;: &quot;intent1&quot;,&#10;    &quot;name&quot;: &quot;意图1&quot;,&#10;    &quot;examples&quot;: [&quot;示例1&quot;, &quot;示例2&quot;]&#10;  }&#10;]"></textarea>
                </div>
            </div>
            <div class="modal-footer">
                <button onclick="closeModal()" class="btn btn-secondary">取消</button>
                <button onclick="saveIntent()" class="btn btn-accent">保存应用意图树</button>
            </div>
        </div>
    </div>

    <script src="/admin/app.js"></script>
</body>
</html>
'''

_ADMIN_CSS = ''':root {
    --bg-primary: #0c0c0f;
    --bg-secondary: #141419;
    --bg-tertiary: #1a1a21;
    --bg-elevated: #222230;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-soft: rgba(59, 130, 246, 0.15);
    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.15);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.15);
    --success: #22c55e;
    --danger: #ef4444;
    --radius: 12px;
    --radius-sm: 8px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.bg-grid {
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

.container {
    position: relative;
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px 32px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 24px;
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.header-logo {
    width: 40px;
    height: 40px;
    color: var(--accent);
}

.header h1 {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.header-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    display: block;
    margin-top: -4px;
}

.tabs {
    display: flex;
    gap: 4px;
    background: var(--bg-secondary);
    padding: 6px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-bottom: 24px;
    overflow-x: auto;
}

.tab-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 20px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 500;
    font-family: inherit;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.tab-btn svg {
    width: 18px;
    height: 18px;
}

.tab-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.tab-btn.active {
    background: var(--accent);
    color: #fff;
}

.tab-panel {
    display: none;
    animation: fadeIn 0.3s ease;
}

.tab-panel.active { display: block; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 24px;
}

.panel h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}

.panel-header h2 {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
}

.header-actions {
    display: flex;
    gap: 12px;
}

.modal-lg .modal-content { max-width: 700px; }

.btn-sm { padding: 6px 12px; font-size: 13px; }

.btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    font-family: inherit;
    transition: all 0.2s ease;
}

.btn svg { width: 16px; height: 16px; }

.btn-primary { background: var(--bg-tertiary); color: var(--text-primary); }
.btn-primary:hover { background: var(--bg-elevated); }

.btn-secondary { background: var(--bg-tertiary); color: var(--text-secondary); }
.btn-secondary:hover { background: var(--bg-elevated); color: var(--text-primary); }

.btn-accent { background: var(--accent); color: #fff; }
.btn-accent:hover { background: var(--accent-hover); }

.btn-danger { background: var(--danger); color: #fff; }
.btn-danger:hover { opacity: 0.9; }

.intent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
}

.intent-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 20px;
    transition: all 0.2s ease;
}

.intent-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
}

.intent-card h3 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 8px;
}

.intent-card .cache-key {
    display: inline-block;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 4px 10px;
    border-radius: 4px;
    margin-bottom: 12px;
}

.intent-card .meta {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}

.intent-card .actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}

.test-form, .llm-form, .batch-form {
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    padding: 24px;
}

.release-layout,
.release-columns,
.batch-layout {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 20px;
}

.release-card,
.batch-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 24px;
}

.release-card h3,
.batch-card h3,
.section-subtitle {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 16px;
}

.release-hint {
    color: var(--text-muted);
    font-size: 12px;
    margin: -4px 0 16px;
}

.compact-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.compact-item {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
}

.compact-item-title {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 13px;
    font-weight: 600;
}

.compact-item-meta {
    margin-top: 6px;
    color: var(--text-muted);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}

.pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 11px;
    white-space: nowrap;
}

.form-group { margin-bottom: 20px; }
.form-group:last-child { margin-bottom: 0; }

.form-group label {
    display: block;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.form-group input,
.form-group textarea,
.form-group select {
    width: 100%;
    padding: 12px 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    transition: border-color 0.2s ease;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
    outline: none;
    border-color: var(--accent);
}

.form-group textarea { resize: vertical; min-height: 100px; }

.form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.result-panel {
    margin-top: 20px;
    padding: 20px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    display: none;
    white-space: pre-wrap;
}

.result-panel.show { display: block; }

.eval-result {
    display: none;
    margin-top: 20px;
}

.eval-result.show { display: block; }

.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.metric-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px;
}

.metric-card strong {
    display: block;
    font-size: 24px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    margin-bottom: 4px;
}

.metric-card span {
    color: var(--text-muted);
    font-size: 12px;
}

.failed-list {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    max-height: 360px;
    overflow-y: auto;
}

.failed-item {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
}

.failed-item:last-child { border-bottom: none; }

.failed-title {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 13px;
    margin-bottom: 8px;
}

.failed-meta {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    word-break: break-all;
}

.snapshot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 16px;
}

.snapshot-block {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
}

.snapshot-block summary {
    cursor: pointer;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 600;
}

.json-block {
    margin-top: 12px;
    max-height: 320px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
}

.strategy-editor {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 18px;
}

.strategy-editor-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 16px;
}

.strategy-editor-head h3 {
    margin: 0 0 6px;
    font-size: 15px;
}

.strategy-editor-subtitle {
    color: var(--text-muted);
    font-size: 12px;
}

.strategy-meta-grid {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) minmax(220px, 1.2fr) minmax(180px, 1fr);
    gap: 12px;
    margin-bottom: 12px;
}

.strategy-editor .form-group {
    margin-bottom: 12px;
}

.strategy-param-layout {
    display: grid;
    grid-template-columns: minmax(240px, 0.8fr) minmax(360px, 1.4fr);
    gap: 14px;
    align-items: start;
}

.strategy-param-section {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
}

.strategy-param-section.wide {
    grid-column: 1 / -1;
}

.strategy-param-section h4 {
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 600;
}

.strategy-param-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
}

.strategy-param-card label {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
}

.strategy-param-card small {
    display: block;
    margin-top: 5px;
    min-height: 28px;
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.35;
}

.strategy-param-card input,
.strategy-param-card select {
    width: 100%;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.strategy-prompt-toolbar {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
}

.strategy-token-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.strategy-token {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    background: var(--accent-soft);
    border-radius: 4px;
    padding: 3px 7px;
}

.strategy-prompt {
    width: 100%;
    min-height: 300px;
    resize: vertical;
    padding: 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.65;
}

.strategy-prompt:focus,
.strategy-param-card input:focus,
.strategy-param-card select:focus {
    outline: none;
    border-color: var(--accent);
}

.strategy-editor-note {
    margin-top: 8px;
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.55;
}

@media (max-width: 900px) {
    .strategy-meta-grid,
    .strategy-param-layout {
        grid-template-columns: 1fr;
    }
}

.config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
}

.config-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 20px;
}

.config-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.config-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
}

.config-badge {
    font-size: 10px;
    padding: 3px 8px;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 4px;
    font-weight: 500;
}

.config-card input,
.config-card select {
    width: 100%;
    padding: 10px 14px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 10px;
}

.config-card input:focus,
.config-card select:focus {
    outline: none;
    border-color: var(--accent);
}

.config-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.5;
}

.form-actions {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}

.status-msg {
    font-size: 13px;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.status-msg.show { opacity: 1; }
.status-msg.success { color: var(--success); }
.status-msg.error { color: var(--danger); }

.file-upload {
    position: relative;
}

.file-upload input {
    width: 100%;
    padding: 14px 16px;
    background: var(--bg-primary);
    border: 2px dashed var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
}

.file-hint {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 8px;
}

.checkbox-label {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
}

.checkbox-label input {
    width: 18px;
    height: 18px;
    accent-color: var(--accent);
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.summary-card {
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    padding: 20px;
    text-align: center;
}

.summary-card .value {
    font-size: 32px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    margin-bottom: 4px;
}

.summary-card.pass .value { color: var(--success); }
.summary-card.fail .value { color: var(--danger); }

.summary-card .label {
    font-size: 12px;
    color: var(--text-muted);
}

.details-list {
    max-height: 400px;
    overflow-y: auto;
}

.detail-item {
    padding: 14px 16px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    font-size: 13px;
}

.detail-item.pass { border-left: 3px solid var(--success); }
.detail-item.fail { border-left: 3px solid var(--danger); }

.detail-item .query {
    color: var(--text-muted);
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.detail-item .result {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.detail-item .tag {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-elevated);
    border-radius: 4px;
    font-size: 11px;
}

.modal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    padding: 24px;
}

.modal.show { display: flex; }

.modal-content {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 100%;
    max-width: 560px;
    max-height: 90vh;
    overflow-y: auto;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
}

.modal-header h2 { font-size: 16px; font-weight: 600; }

.modal-close {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border: none;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    font-size: 20px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.modal-close:hover { background: var(--bg-elevated); color: var(--text-primary); }

.modal-body { padding: 24px; }

.modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 20px 24px;
    border-top: 1px solid var(--border);
}

.loading {
    text-align: center;
    padding: 60px;
    color: var(--text-muted);
}

.footer {
    text-align: center;
    padding: 24px;
    font-size: 12px;
    color: var(--text-muted);
}

@media (max-width: 768px) {
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .config-grid { grid-template-columns: 1fr; }
}
'''

_ADMIN_JS = '''let intents = [];
let currentConfig = {};
let currentLlmConfig = {};
let intentTrees = [];
let strategyTemplates = [];
let strategyConfigs = [];
let deployments = [];
let evalSets = [];
let evalRuns = [];

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById('tab-' + tabName).classList.add('active');
    if (tabName === 'release') {
        loadReleaseData();
    } else if (tabName === 'config') {
        loadReleaseData();
    } else if (tabName === 'test') {
        loadReleaseData();
        loadEvalData();
    }
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

async function loadIntents() {
    try {
        const resp = await fetch('/admin/apps');
        const data = await resp.json();
        intents = (data.apps || []).map(app => ({...app, cache_key: app.app_id}));
        renderIntentList();
        updateAppSelects();
    } catch (e) { console.error('Failed to load intents:', e); }
}

function renderIntentList() {
    const container = document.getElementById('intent-list');
    if (intents.length === 0) {
        container.innerHTML = '<div class="loading">暂无配置，点击右上角新建</div>';
        return;
    }
    container.innerHTML = intents.map(it => `
        <div class="intent-card">
            <h3>${escapeHtml(it.name)}</h3>
            <span class="cache-key">app_id: ${escapeHtml(it.cache_key)}</span>
            <p style="color:var(--text-muted);font-size:13px;margin-top:8px;">${escapeHtml(it.description || '')}</p>
            <div class="meta">
                <span>${it.intent_count} 个意图</span>
                <span>${it.updated_at || it.created_at}</span>
            </div>
            <div class="actions">
                <button class="btn btn-primary btn-small" onclick="editIntent('${it.cache_key}')">编辑</button>
                <button class="btn btn-secondary btn-small" onclick="deleteIntent('${it.cache_key}')">删除</button>
            </div>
        </div>
    `).join('');
}

function updateAppSelects() {
    ['test-cache-key', 'batch-app-id'].forEach(id => {
        const select = document.getElementById(id);
        if (!select) return;
        const current = select.value;
        select.innerHTML = '<option value="">-- 请选择 --</option>' +
            intents.map(it => `<option value="${escapeHtml(it.cache_key)}">${escapeHtml(it.name)} (${escapeHtml(it.cache_key)})</option>`).join('');
        if (current) select.value = current;
    });
}

function updateTestStrategySelect() {
    updateStrategySelect('test-strategy-select');
    updateStrategySelect('batch-strategy-select');
}

function updateStrategySelect(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return;
    const current = select.value;
    const configOptions = strategyConfigs.map(c => `
        <option value="config:${escapeHtml(c.strategy_config_id)}">策略配置：${escapeHtml(c.name)} (${escapeHtml(c.strategy_config_id)})</option>
    `).join('');
    select.innerHTML = configOptions || '<option value="">暂无策略配置，请先在“策略配置”页创建</option>';
    if ([...select.options].some(o => o.value === current)) {
        select.value = current;
    } else if (select.options.length) {
        select.value = select.options[0].value;
    }
}

function getStrategyTemplate(templateId) {
    return strategyTemplates.find(t => t.strategy_template_id === templateId);
}

function getStrategyConfig(configId) {
    return strategyConfigs.find(c => c.strategy_config_id === configId);
}

const STRATEGY_PARAM_META = {
    abs_threshold: ['命中阈值', '最高候选低于该值时视为未命中'],
    gap_threshold: ['歧义阈值', 'Top1 与 Top2 分差低于该值时需要澄清'],
    top_k: ['候选数量', '返回给运营查看的候选意图数'],
    ambiguous_action: ['歧义动作', '当前主要用于标记澄清状态'],
    weight_intro: ['描述权重', '意图描述与示例问题的融合比例'],
    question_top_k: ['示例 Top-K', '取最相近示例问题参与聚合'],
    alpha_short: ['短查询语义权重', '短问题中 Embedding 分数占比'],
    alpha_long: ['长查询语义权重', '长问题中 Embedding 分数占比'],
    enable_bm25: ['BM25', '是否启用关键词召回补充'],
    short_query_len: ['短查询长度', '低于该字符数按短查询处理'],
    example_question_threshold: ['示例直连阈值', '命中示例问题足够高时直接采用'],
    llm_min_confidence: ['最低采信置信度', '低于该值时不采信大模型分类结果'],
    prompt_template: ['提示词模板', '支持运行时变量 {intent_list} 和 {query}']
};

function paramTitle(key) {
    return STRATEGY_PARAM_META[key]?.[0] || key;
}

function paramDesc(key) {
    return STRATEGY_PARAM_META[key]?.[1] || '';
}

function renderParamControl(key, spec, value) {
    const safeKey = escapeHtml(key);
    const title = escapeHtml(paramTitle(key));
    const desc = escapeHtml(paramDesc(key));
    if (spec.type === 'bool') {
        return `
            <div class="strategy-param-card">
                <label><span>${title}</span><span>${safeKey}</span></label>
                <select class="strategy-param-input" data-param-key="${safeKey}" data-param-type="bool">
                    <option value="true" ${value ? 'selected' : ''}>开启</option>
                    <option value="false" ${!value ? 'selected' : ''}>关闭</option>
                </select>
                <small>${desc}</small>
            </div>
        `;
    }
    const inputType = spec.type === 'int' || spec.type === 'float' ? 'number' : 'text';
    return `
        <div class="strategy-param-card">
            <label><span>${title}</span><span>${safeKey}</span></label>
            <input class="strategy-param-input" type="${inputType}" data-param-key="${safeKey}" data-param-type="${escapeHtml(spec.type || 'text')}"
                value="${escapeHtml(value ?? '')}" min="${escapeHtml(spec.min ?? '')}" max="${escapeHtml(spec.max ?? '')}" step="${escapeHtml(spec.step ?? 'any')}">
            <small>${desc}</small>
        </div>
    `;
}

function renderStrategyParamEditor(template, params = {}) {
    const schema = template?.param_schema || {};
    const defaults = template?.default_params || {};
    const merged = {...defaults, ...params};
    const keys = Object.keys(schema);
    if (!keys.length) return '<div class="loading">该策略模板暂无可配置参数</div>';
    const textKeys = keys.filter(key => schema[key]?.type === 'text');
    const scalarKeys = keys.filter(key => schema[key]?.type !== 'text');
    const scalarSection = scalarKeys.length ? `
        <div class="strategy-param-section">
            <h4>参数</h4>
            <div class="strategy-param-grid">
                ${scalarKeys.map(key => renderParamControl(key, schema[key] || {}, merged[key])).join('')}
            </div>
        </div>
    ` : '';
    const textSection = textKeys.map(key => `
        <div class="strategy-param-section ${scalarKeys.length ? '' : 'wide'}">
            <div class="strategy-prompt-toolbar">
                <h4>${escapeHtml(paramTitle(key))}</h4>
                <div class="strategy-token-row">
                    <span class="strategy-token">{intent_list}</span>
                    <span class="strategy-token">{query}</span>
                </div>
            </div>
            <textarea class="strategy-param-input strategy-prompt" data-param-key="${escapeHtml(key)}" data-param-type="text" placeholder="写清楚分类边界、输出格式和无法判断时的处理方式">${escapeHtml(merged[key] ?? '')}</textarea>
            <div class="strategy-editor-note">${escapeHtml(paramDesc(key))}。后台会把当前意图树渲染成 {intent_list}，把用户问题渲染成 {query}。</div>
        </div>
    `).join('');
    return `<div class="strategy-param-layout">${scalarSection}${textSection}</div>`;
}

function readStrategyParamsFromEditor() {
    const params = {};
    document.querySelectorAll('#strategy-param-editor .strategy-param-input').forEach(input => {
        const key = input.dataset.paramKey;
        const type = input.dataset.paramType;
        if (type === 'bool') {
            params[key] = input.value === 'true';
        } else if (type === 'int') {
            params[key] = parseInt(input.value || '0', 10);
        } else if (type === 'float') {
            params[key] = parseFloat(input.value || '0');
        } else {
            params[key] = input.value;
        }
    });
    return params;
}

function openStrategyConfigEditor(configId = '') {
    const box = document.getElementById('strategy-config-editor');
    if (!box) return;
    const existing = configId ? getStrategyConfig(configId) : null;
    const templateId = existing?.strategy_template_id || strategyTemplates[0]?.strategy_template_id || '';
    const template = getStrategyTemplate(templateId);
    const suggestedId = configId ? existing.strategy_config_id : `strategy_${Date.now()}`;
    box.className = 'eval-result show';
    box.innerHTML = `
        <div class="strategy-editor">
            <div class="strategy-editor-head">
                <div>
                    <h3>${existing ? '编辑策略配置' : '新建策略配置'}</h3>
                    <div class="strategy-editor-subtitle">${template ? escapeHtml(template.name + ' · ' + template.strategy_type) : '选择一个策略模板后开始配置'}</div>
                </div>
                <button onclick="closeStrategyConfigEditor()" class="btn btn-secondary btn-small">关闭</button>
            </div>
            <div class="strategy-meta-grid">
                <div class="form-group">
                    <label>配置 ID</label>
                    <input type="text" id="strategy-editor-id" value="${escapeHtml(suggestedId)}" ${existing ? 'readonly' : ''}>
                </div>
                <div class="form-group">
                    <label>名称</label>
                    <input type="text" id="strategy-editor-name" value="${escapeHtml(existing?.name || '')}" placeholder="如 债券问答 LLM Prompt">
                </div>
                <div class="form-group">
                    <label>策略模板</label>
                    <select id="strategy-editor-template" onchange="onStrategyEditorTemplateChange()" ${existing ? 'disabled' : ''}>
                        ${strategyTemplates.map(t => `<option value="${escapeHtml(t.strategy_template_id)}" ${t.strategy_template_id === templateId ? 'selected' : ''}>${escapeHtml(t.name)} (${escapeHtml(t.strategy_type)})</option>`).join('')}
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label>描述</label>
                <input type="text" id="strategy-editor-description" value="${escapeHtml(existing?.description || '')}" placeholder="说明这个配置适合哪个应用或场景">
            </div>
            <div id="strategy-param-editor">
                ${renderStrategyParamEditor(template, existing?.params || {})}
            </div>
            <div class="form-actions">
                <button onclick="saveStrategyConfigFromEditor(${existing ? 'true' : 'false'})" class="btn btn-accent">保存策略配置</button>
                <button onclick="closeStrategyConfigEditor()" class="btn btn-secondary">取消</button>
                <span id="strategy-editor-status" class="status-msg"></span>
            </div>
        </div>
    `;
}

function closeStrategyConfigEditor() {
    const box = document.getElementById('strategy-config-editor');
    if (box) {
        box.className = 'eval-result';
        box.innerHTML = '';
    }
}

function onStrategyEditorTemplateChange() {
    const templateId = document.getElementById('strategy-editor-template').value;
    const template = getStrategyTemplate(templateId);
    document.getElementById('strategy-param-editor').innerHTML = renderStrategyParamEditor(template, {});
}

function duplicateStrategyConfig(configId) {
    const source = getStrategyConfig(configId);
    if (!source) return;
    openStrategyConfigEditor();
    document.getElementById('strategy-editor-id').value = `${source.strategy_config_id}_copy`;
    document.getElementById('strategy-editor-name').value = `${source.name} 副本`;
    document.getElementById('strategy-editor-template').value = source.strategy_template_id;
    document.getElementById('strategy-editor-description').value = source.description || '';
    const template = getStrategyTemplate(source.strategy_template_id);
    document.getElementById('strategy-param-editor').innerHTML = renderStrategyParamEditor(template, source.params || {});
}

async function saveStrategyConfigFromEditor(isUpdate) {
    const status = document.getElementById('strategy-editor-status');
    const id = document.getElementById('strategy-editor-id').value.trim();
    const name = document.getElementById('strategy-editor-name').value.trim();
    const templateId = document.getElementById('strategy-editor-template').value;
    const description = document.getElementById('strategy-editor-description').value.trim();
    const params = readStrategyParamsFromEditor();
    if (!id || !name || !templateId) {
        alert('请填写配置 ID、名称并选择策略模板');
        return;
    }
    const payload = isUpdate
        ? {name, description, params, resources: {}}
        : {strategy_config_id: id, name, strategy_template_id: templateId, description, params, resources: {}};
    try {
        status.textContent = '保存中...';
        status.className = 'status-msg show';
        const resp = await fetch('/admin/strategy-configs' + (isUpdate ? '/' + id : ''), {
            method: isUpdate ? 'PUT' : 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || '保存失败');
        status.textContent = '已保存';
        status.className = 'status-msg success show';
        await loadReleaseData();
        openStrategyConfigEditor(data.strategy_config_id);
    } catch (e) {
        status.textContent = e.message || '保存失败';
        status.className = 'status-msg error show';
    }
}

async function loadConfig() {
    try {
        const resp = await fetch('/admin/config');
        const data = await resp.json();
        currentConfig = data.config || {};
        fillConfigForm(currentConfig);
    } catch (e) { console.error('Failed to load config:', e); }
}

function fillConfigForm(cfg) {
    document.getElementById('cfg-abs-threshold').value = cfg.abs_threshold ?? 0.35;
    document.getElementById('cfg-gap-threshold').value = cfg.gap_threshold ?? 0.05;
    document.getElementById('cfg-top-k').value = cfg.top_k ?? 3;
    document.getElementById('cfg-weight-intro').value = cfg.weight_intro ?? 0.5;
    document.getElementById('cfg-alpha-short').value = cfg.alpha_short ?? 0.6;
    document.getElementById('cfg-alpha-long').value = cfg.alpha_long ?? 0.8;
    document.getElementById('cfg-enable-bm25').value = cfg.enable_bm25 ? 'true' : 'false';
    document.getElementById('cfg-short-query-len').value = cfg.short_query_len ?? 10;
    document.getElementById('cfg-question-top-k').value = cfg.question_top_k ?? 5;
    document.getElementById('cfg-example-question-threshold').value = cfg.example_question_threshold ?? 0.95;
}

async function saveConfig() {
    const cfg = {
        abs_threshold: parseFloat(document.getElementById('cfg-abs-threshold').value),
        gap_threshold: parseFloat(document.getElementById('cfg-gap-threshold').value),
        top_k: parseInt(document.getElementById('cfg-top-k').value),
        weight_intro: parseFloat(document.getElementById('cfg-weight-intro').value),
        alpha_short: parseFloat(document.getElementById('cfg-alpha-short').value),
        alpha_long: parseFloat(document.getElementById('cfg-alpha-long').value),
        enable_bm25: document.getElementById('cfg-enable-bm25').value === 'true',
        short_query_len: parseInt(document.getElementById('cfg-short-query-len').value),
        question_top_k: parseInt(document.getElementById('cfg-question-top-k').value),
        example_question_threshold: parseFloat(document.getElementById('cfg-example-question-threshold').value),
    };
    try {
        const resp = await fetch('/admin/config', {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(cfg)
        });
        const status = document.getElementById('config-status');
        if (resp.ok) {
            status.textContent = '保存成功'; status.className = 'status-msg success show';
        } else {
            status.textContent = '保存失败'; status.className = 'status-msg error show';
        }
        setTimeout(() => { status.classList.remove('show'); }, 2000);
    } catch (e) { console.error('Failed to save config:', e); }
}

async function loadLlmConfig() {
    try {
        const resp = await fetch('/admin/llm-config');
        const data = await resp.json();
        currentLlmConfig = data.config || {};
        fillLlmConfigForm(currentLlmConfig);
    } catch (e) { console.error('Failed to load LLM config:', e); }
}

function fillLlmConfigForm(cfg) {
    document.getElementById('llm-provider').value = cfg.provider || 'deepseek';
    document.getElementById('llm-api-key').value = '';
    document.getElementById('llm-api-key').placeholder = cfg.api_key ? '已配置（输入新值可修改）' : '输入 API Key';
    document.getElementById('llm-base-url').value = cfg.base_url || 'https://api.deepseek.com/v1';
    document.getElementById('llm-model').value = cfg.model || 'deepseek-chat';
    document.getElementById('llm-max-tokens').value = cfg.max_tokens || 256;
    document.getElementById('llm-temperature').value = cfg.temperature || 0.1;
    document.getElementById('llm-timeout').value = cfg.timeout || 5;
}

function onLlmProviderChange() {
    const provider = document.getElementById('llm-provider').value;
    const baseUrlInput = document.getElementById('llm-base-url');
    const modelInput = document.getElementById('llm-model');
    if (provider === 'deepseek') {
        baseUrlInput.value = 'https://api.deepseek.com/v1';
        modelInput.value = 'deepseek-chat';
    } else if (provider === 'openai') {
        baseUrlInput.value = 'https://api.openai.com/v1';
        modelInput.value = 'gpt-4';
    }
}

async function saveLlmConfig() {
    const cfg = {
        provider: document.getElementById('llm-provider').value,
        api_key: document.getElementById('llm-api-key').value,
        base_url: document.getElementById('llm-base-url').value,
        model: document.getElementById('llm-model').value,
        max_tokens: parseInt(document.getElementById('llm-max-tokens').value),
        temperature: parseFloat(document.getElementById('llm-temperature').value),
        timeout: parseFloat(document.getElementById('llm-timeout').value),
        min_confidence: currentLlmConfig.min_confidence ?? 0.65,
    };
    try {
        const resp = await fetch('/admin/llm-config', {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(cfg)
        });
        const status = document.getElementById('llm-status');
        if (resp.ok) {
            status.textContent = '保存成功'; status.className = 'status-msg success show';
        } else {
            status.textContent = '保存失败'; status.className = 'status-msg error show';
        }
        setTimeout(() => { status.classList.remove('show'); }, 2000);
    } catch (e) { console.error('Failed to save LLM config:', e); }
}

function buildStrategyPayload(strategyValue) {
    const payload = { params_override: {} };
    if (strategyValue.startsWith('config:')) {
        payload.strategy_config_id = strategyValue.slice('config:'.length);
    } else if (strategyValue.startsWith('template:')) {
        payload.strategy_template_id = strategyValue.slice('template:'.length) || 'hybrid_retrieval_default';
    }
    return payload;
}

async function testRoute() {
    const cacheKey = document.getElementById('test-cache-key').value;
    const query = document.getElementById('test-query').value.trim();
    const strategyValue = document.getElementById('test-strategy-select').value;
    if (!cacheKey || !query) { alert('请选择应用并输入查询'); return; }
    if (!strategyValue) { alert('请先在“策略配置”页创建并选择策略配置'); return; }
    const resultBox = document.getElementById('test-result');
    resultBox.classList.add('show');
    resultBox.textContent = '测试中...';
    try {
        const payload = {
            ...buildStrategyPayload(strategyValue),
            intent_tree_id: cacheKey,
            query
        };
        const resp = await fetch('/admin/test-route', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        resultBox.innerHTML = resp.ok
            ? formatResult(data.result, data) + '\\n\\n<span style="color:var(--text-secondary)">测试策略:</span> ' + escapeHtml(data.strategy_config_id || data.strategy_template_id || '')
            : '<span style="color:var(--danger)">错误: ' + escapeHtml(data.detail || 'Unknown') + '</span>';
    } catch (e) { resultBox.innerHTML = '<span style="color:var(--danger)">请求失败</span>'; }
}

function formatResult(result, meta = {}) {
    if (!result) return '';
    const topK = result.top_k || [];
    const best = topK.length ? topK[0] : null;
    const second = topK.length > 1 ? topK[1] : null;
    const params = meta.merged_params || meta.params_snapshot || meta.strategy_params || {};
    const strategyType = meta.strategy_type || meta.strategy_template?.strategy_type || '';
    const isLlm = strategyType === 'llm_prompt' || result.method === 'llm_prompt';
    const strategyLabel = isLlm ? '大模型 Prompt' : '混合检索';
    const statusLabel = result.is_ambiguous
        ? '需要澄清'
        : (result.is_ood ? '未达到命中阈值' : '已命中');
    const threshold = isLlm ? params.llm_min_confidence : params.abs_threshold;
    const gapThreshold = params.gap_threshold;
    const items = topK.map(t => '<span style="color:var(--accent)">  →</span> ' + escapeHtml(t[0]) + ' (' + escapeHtml(t[1]) + '): ' + (t[2]*100).toFixed(1) + '%').join('\\n');
    const displayName = result.is_ambiguous || result.is_ood ? statusLabel : result.selected_intent_name;
    let hint = '';
    if (result.is_ambiguous && best) {
        hint = `\\n<span style="color:var(--text-secondary)">推荐候选:</span> ${escapeHtml(best[1])} (${escapeHtml(best[0])}) ${(best[2]*100).toFixed(1)}%`;
        if (second) {
            const gapText = (Math.abs(best[2] - second[2]) * 100).toFixed(1) + '%';
            const thresholdText = Number.isFinite(gapThreshold) ? `，低于当前歧义阈值 ${(gapThreshold * 100).toFixed(1)}%` : '';
            hint += `\\n<span style="color:var(--text-secondary)">原因:</span> Top1 与 Top2 差距 ${gapText}${thresholdText}`;
        }
    } else if (result.is_ood && best) {
        const thresholdText = Number.isFinite(threshold) ? `，低于当前命中阈值 ${(threshold * 100).toFixed(1)}%` : '';
        hint = `\\n<span style="color:var(--text-secondary)">推荐候选:</span> ${escapeHtml(best[1])} (${escapeHtml(best[0])}) ${(best[2]*100).toFixed(1)}%`
            + `\\n<span style="color:var(--text-secondary)">原因:</span> 最高候选分数 ${(best[2]*100).toFixed(1)}%${thresholdText}`;
    }
    const thresholdLine = Number.isFinite(threshold)
        ? `\\n<span style="color:var(--text-secondary)">命中阈值:</span> ${(threshold * 100).toFixed(1)}%`
        : '';
    return `<span style="color:var(--text-secondary)">识别结果:</span> ${escapeHtml(displayName)}${hint}
<span style="color:var(--text-secondary)">置信度:</span> ${(result.confidence*100).toFixed(1)}%
<span style="color:var(--text-secondary)">识别策略:</span> ${strategyLabel}
<span style="color:var(--text-secondary)">判定状态:</span> ${statusLabel}${thresholdLine}

<span style="color:var(--text-secondary)">候选列表:</span>
${items}`;
}

async function loadEvalData() {
    try {
        const [setsResp, runsResp] = await Promise.all([
            fetch('/admin/eval-sets'),
            fetch('/admin/eval-runs')
        ]);
        evalSets = (await setsResp.json()).eval_sets || [];
        evalRuns = (await runsResp.json()).eval_runs || [];
        renderEvalSets();
        renderEvalRuns();
    } catch (e) {
        console.error('Failed to load eval data:', e);
    }
}

function renderEvalSets() {
    const select = document.getElementById('batch-eval-set');
    if (!select) return;
    const current = select.value;
    select.innerHTML = '<option value="">-- 选择评测集 --</option>' + evalSets.map(s => `
        <option value="${escapeHtml(s.eval_set_id)}">${escapeHtml(s.name)} · ${s.case_count || 0} 条 (${escapeHtml(s.app_id || '')})</option>
    `).join('');
    if (current) select.value = current;
}

function renderEvalRuns() {
    const box = document.getElementById('eval-run-list');
    if (!box) return;
    if (!evalRuns.length) {
        box.innerHTML = '<div class="loading">暂无评测记录</div>';
        return;
    }
    box.innerHTML = evalRuns.slice(0, 20).map(run => {
        const m = run.metrics || {};
        return `
            <div class="compact-item">
                <div class="compact-item-title">
                    <span>${escapeHtml(run.eval_set_name || run.eval_set_id)} / ${escapeHtml(run.app_id)}</span>
                    <span class="pill">正确率 ${((m.accuracy || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="compact-item-meta">
                    ${escapeHtml(run.strategy_name || run.strategy_template_id)} · ${m.passed || 0}/${m.total || 0} · avg ${m.avg_latency_ms || 0}ms · ${escapeHtml(run.created_at || '')}
                </div>
                <div class="actions">
                    <button class="btn btn-secondary btn-small" onclick="viewEvalRun('${escapeHtml(run.eval_run_id)}')">查看详情</button>
                </div>
            </div>
        `;
    }).join('');
}

async function uploadEvalSet() {
    const appId = document.getElementById('batch-app-id').value || document.getElementById('test-cache-key').value;
    const name = document.getElementById('eval-set-name').value.trim();
    const fileInput = document.getElementById('eval-csv-file');
    const status = document.getElementById('eval-upload-status');
    if (!appId) { alert('请选择评测应用'); return; }
    if (!fileInput.files.length) { alert('请选择 CSV 文件'); return; }
    const form = new FormData();
    form.append('app_id', appId);
    form.append('name', name);
    form.append('file', fileInput.files[0]);
    status.textContent = '上传中...';
    status.className = 'status-msg show';
    try {
        const resp = await fetch('/admin/eval-sets/upload-csv', { method: 'POST', body: form });
        const data = await resp.json();
        if (resp.ok) {
            status.textContent = '已保存：' + data.case_count + ' 条';
            status.className = 'status-msg success show';
            document.getElementById('eval-set-name').value = '';
            fileInput.value = '';
            await loadEvalData();
            document.getElementById('batch-eval-set').value = data.eval_set_id;
        } else {
            status.textContent = '上传失败：' + (data.detail || 'Unknown');
            status.className = 'status-msg error show';
        }
    } catch (e) {
        status.textContent = '上传请求失败';
        status.className = 'status-msg error show';
    }
}

async function runBatchEval() {
    const appId = document.getElementById('batch-app-id').value || document.getElementById('test-cache-key').value;
    const evalSetId = document.getElementById('batch-eval-set').value;
    const strategyValue = document.getElementById('batch-strategy-select').value || document.getElementById('test-strategy-select').value;
    const box = document.getElementById('batch-result');
    if (!appId || !evalSetId) { alert('请选择应用和评测集'); return; }
    if (!strategyValue) { alert('请先在“策略配置”页创建并选择策略配置'); return; }
    box.className = 'eval-result show';
    box.innerHTML = '<div class="result-panel show">批量评测中...</div>';
    try {
        const payload = {
            ...buildStrategyPayload(strategyValue),
            app_id: appId,
            eval_set_id: evalSetId
        };
        const resp = await fetch('/admin/eval-runs', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (resp.ok) {
            renderEvalRunSummary(data);
            await loadEvalData();
        } else {
            box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">评测失败: ' + escapeHtml(data.detail || 'Unknown') + '</span></div>';
        }
    } catch (e) {
        box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">评测请求失败</span></div>';
    }
}

function renderEvalRunSummary(run) {
    const box = document.getElementById('batch-result');
    const m = run.metrics || {};
    const failed = run.failed_cases || [];
    const paramKeys = Object.keys(run.params_snapshot || {});
    const intentSnapshot = run.intent_tree_snapshot || {};
    const evalSetSnapshot = run.eval_set_snapshot || {};
    box.className = 'eval-result show';
    box.innerHTML = `
        <div class="metric-row">
            <div class="metric-card"><strong>${((m.accuracy || 0) * 100).toFixed(1)}%</strong><span>Top-1 正确率</span></div>
            <div class="metric-card"><strong>${m.passed || 0}/${m.total || 0}</strong><span>通过案例</span></div>
            <div class="metric-card"><strong>${m.avg_latency_ms || 0}ms</strong><span>平均响应</span></div>
            <div class="metric-card"><strong>${m.p95_latency_ms || 0}ms</strong><span>P95 响应</span></div>
        </div>
        <div class="compact-item">
            <div class="compact-item-title"><span>${escapeHtml(run.strategy_name || run.strategy_template_id)}</span><span class="pill">${escapeHtml(run.eval_run_id)}</span></div>
            <div class="compact-item-meta">评测集=${escapeHtml(run.eval_set_name || run.eval_set_id)} · 意图版本=${escapeHtml(run.intent_tree_version_id)} · 参数快照已保存${paramKeys.length ? ' · ' + escapeHtml(paramKeys.join(', ')) : ''}</div>
        </div>
        <div class="snapshot-grid">
            <details class="snapshot-block">
                <summary>意图树快照 · ${escapeHtml(intentSnapshot.name || run.app_id || '')} · ${(intentSnapshot.intents || []).length} 个意图</summary>
                <pre class="json-block">${escapeHtml(JSON.stringify(intentSnapshot, null, 2))}</pre>
            </details>
            <details class="snapshot-block">
                <summary>策略参数快照 · ${escapeHtml(run.strategy_type || run.strategy_template_id || '')}</summary>
                <pre class="json-block">${escapeHtml(JSON.stringify({
                    strategy_template_id: run.strategy_template_id,
                    strategy_config_id: run.strategy_config_id,
                    params_snapshot: run.params_snapshot || {}
                }, null, 2))}</pre>
            </details>
            <details class="snapshot-block">
                <summary>评测集快照 · ${(evalSetSnapshot.cases || []).length || (m.total || 0)} 条案例</summary>
                <pre class="json-block">${escapeHtml(JSON.stringify(evalSetSnapshot, null, 2))}</pre>
            </details>
        </div>
        <h3 class="section-subtitle" style="margin-top:18px;">失败案例 ${failed.length}</h3>
        ${renderFailedCases(failed)}
    `;
}

function renderFailedCases(failed) {
    if (!failed.length) return '<div class="compact-item"><div class="compact-item-title">全部通过</div></div>';
    return `<div class="failed-list">${failed.slice(0, 50).map(item => `
        <div class="failed-item">
            <div class="failed-title">
                <span>${escapeHtml(item.query)}</span>
                <span class="pill">${escapeHtml(item.result?.method || item.error || 'failed')}</span>
            </div>
            <div class="failed-meta">
                expected=${escapeHtml(item.expected_intent_id)} · predicted=${escapeHtml(item.predicted_intent_id || '')} · confidence=${(((item.result || {}).confidence || 0) * 100).toFixed(1)}% · ${item.latency_ms || 0}ms
            </div>
        </div>
    `).join('')}</div>`;
}

async function viewEvalRun(runId) {
    const box = document.getElementById('batch-result');
    box.className = 'eval-result show';
    box.innerHTML = '<div class="result-panel show">加载评测记录...</div>';
    try {
        const resp = await fetch('/admin/eval-runs/' + runId);
        const data = await resp.json();
        if (resp.ok) renderEvalRunSummary(data);
        else box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">加载失败: ' + escapeHtml(data.detail || 'Unknown') + '</span></div>';
    } catch (e) {
        box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">加载请求失败</span></div>';
    }
}

function showCreateModal() {
    document.getElementById('modal-title').textContent = '新建应用意图树';
    document.getElementById('modal-cache-key').value = '';
    document.getElementById('modal-cache-key').disabled = false;
    document.getElementById('modal-name').value = '';
    document.getElementById('modal-description').value = '';
    document.getElementById('modal-intents').value = JSON.stringify([
        { id: 'intent1', name: '意图1', examples: ['示例1', '示例2'] }
    ], null, 2);
    document.getElementById('modal').classList.add('show');
}

async function editIntent(cacheKey) {
    const resp = await fetch('/admin/apps/' + cacheKey);
    const data = await resp.json();
    document.getElementById('modal-title').textContent = '编辑应用意图树';
    document.getElementById('modal-cache-key').value = cacheKey;
    document.getElementById('modal-cache-key').disabled = true;
    document.getElementById('modal-name').value = data.name || '';
    document.getElementById('modal-description').value = data.description || '';
    document.getElementById('modal-intents').value = JSON.stringify(data.intents || [], null, 2);
    document.getElementById('modal').classList.add('show');
}

async function deleteIntent(cacheKey) {
    if (!confirm('确定要删除配置 ' + cacheKey + ' 吗？')) return;
    const resp = await fetch('/admin/apps/' + cacheKey, {method: 'DELETE'});
    if (resp.ok) {
        await loadIntents();
        loadReleaseData();
    } else { alert('删除失败'); }
}

async function saveIntent() {
    const cacheKey = document.getElementById('modal-cache-key').value.trim();
    const name = document.getElementById('modal-name').value.trim();
    const description = document.getElementById('modal-description').value.trim();
    let intentsJson = document.getElementById('modal-intents').value.trim();
    if (!cacheKey || !name) { alert('请填写必填项'); return; }
    let intentsData;
    try { intentsData = JSON.parse(intentsJson); } catch (e) { alert('意图定义 JSON 格式错误'); return; }
    const isEdit = document.getElementById('modal-cache-key').disabled;
    const url = isEdit ? '/admin/apps/' + cacheKey : '/admin/apps';
    const method = isEdit ? 'PUT' : 'POST';
    const resp = await fetch(url, {
        method: method,
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ app_id: cacheKey, name: name, description: description, intents: intentsData })
    });
    if (resp.ok) {
        closeModal();
        await loadIntents();
        loadReleaseData();
    }
    else { const data = await resp.json(); alert('保存失败: ' + (data.detail || 'Unknown')); }
}

function closeModal() { document.getElementById('modal').classList.remove('show'); }
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadReleaseData() {
    try {
        const [treesResp, templatesResp, configsResp, deploymentsResp] = await Promise.all([
            fetch('/admin/intent-trees'),
            fetch('/admin/strategy-templates'),
            fetch('/admin/strategy-configs'),
            fetch('/admin/deployments')
        ]);
        intentTrees = (await treesResp.json()).intent_trees || [];
        strategyTemplates = (await templatesResp.json()).strategy_templates || [];
        strategyConfigs = (await configsResp.json()).strategy_configs || [];
        const deploymentData = await deploymentsResp.json();
        deployments = deploymentData.deployments || [];
        updateTestStrategySelect();
        renderReleaseData(deploymentData.runtime_bundles || []);
    } catch (e) {
        console.error('Failed to load release data:', e);
    }
}

function renderReleaseData(runtimeBundles) {
    const treeSelect = document.getElementById('release-intent-tree');
    const configSelect = document.getElementById('release-strategy-config');
    if (treeSelect) {
        const currentTree = treeSelect.value;
        treeSelect.innerHTML = '<option value="">-- 选择意图树 --</option>' +
            intentTrees.map(t => `<option value="${escapeHtml(t.intent_tree_id)}">${escapeHtml(t.name)} (${escapeHtml(t.intent_tree_id)})</option>`).join('');
        if (currentTree) treeSelect.value = currentTree;
    }
    if (configSelect) {
        updateStrategySelect('release-strategy-config');
    }

    const templateList = document.getElementById('strategy-template-list');
    if (templateList) {
        templateList.innerHTML = strategyTemplates.length ? strategyTemplates.map(t => `
            <div class="compact-item">
                <div class="compact-item-title"><span>${escapeHtml(t.name)}</span><span class="pill">${escapeHtml(t.strategy_type)}</span></div>
                <div class="compact-item-meta">${escapeHtml(t.strategy_template_id)} · ${escapeHtml(t.description || '')}</div>
            </div>
        `).join('') : '<div class="loading">暂无策略模板</div>';
    }

    const configList = document.getElementById('strategy-config-list');
    if (configList) {
        configList.innerHTML = strategyConfigs.length ? strategyConfigs.map(c => `
            <div class="compact-item">
                <div class="compact-item-title"><span>${escapeHtml(c.name)}</span><span class="pill">v${c.version || 1}</span></div>
                <div class="compact-item-meta">${escapeHtml(c.strategy_config_id)} · template=${escapeHtml(c.strategy_template_id)}</div>
                <div class="actions">
                    <button class="btn btn-primary btn-small" onclick="openStrategyConfigEditor('${escapeHtml(c.strategy_config_id)}')">编辑</button>
                    <button class="btn btn-secondary btn-small" onclick="duplicateStrategyConfig('${escapeHtml(c.strategy_config_id)}')">复制</button>
                </div>
            </div>
        `).join('') : '<div class="loading">暂无策略配置；可以从模板新建一份用于测试和发布</div>';
    }

    const deploymentList = document.getElementById('deployment-list');
    if (deploymentList) {
        const activeRuntime = (runtimeBundles || []).map(b => b.deployment_id);
        deploymentList.innerHTML = deployments.length ? deployments.map(d => `
            <div class="compact-item">
                <div class="compact-item-title">
                    <span>${escapeHtml(d.tenant_id)} / ${escapeHtml(d.app_id)} / ${escapeHtml(d.environment)}</span>
                    <span class="pill">${escapeHtml(d.status)}${activeRuntime.includes(d.deployment_id) ? ' · loaded' : ''}</span>
                </div>
                <div class="compact-item-meta">deployment=${escapeHtml(d.deployment_id)} · tree=${escapeHtml(d.intent_tree_version_id)} · strategy=${escapeHtml(d.strategy_config_id)}</div>
                <div class="actions">
                    <button class="btn btn-secondary btn-small" onclick="viewDeployment('${escapeHtml(d.deployment_id)}')">详情</button>
                    <button class="btn btn-primary btn-small" onclick="rollbackDeployment('${escapeHtml(d.deployment_id)}')">回滚到此版本</button>
                </div>
            </div>
        `).join('') : '<div class="loading">暂无发布记录</div>';
    }
}

async function viewDeployment(deploymentId) {
    const box = document.getElementById('deployment-detail');
    box.className = 'eval-result show';
    box.innerHTML = '<div class="result-panel show">加载发布详情...</div>';
    try {
        const resp = await fetch('/admin/deployments/' + deploymentId);
        const data = await resp.json();
        if (resp.ok) renderDeploymentDetail(data);
        else box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">加载失败: ' + escapeHtml(data.detail || 'Unknown') + '</span></div>';
    } catch (e) {
        box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">加载请求失败</span></div>';
    }
}

function renderDeploymentDetail(data) {
    const box = document.getElementById('deployment-detail');
    const d = data.deployment || {};
    const tree = data.intent_tree_version || {};
    const template = data.strategy_template || {};
    const indexStatus = data.index_status || {};
    box.className = 'eval-result show';
    box.innerHTML = `
        <div class="metric-row">
            <div class="metric-card"><strong>${escapeHtml(d.status || '')}</strong><span>发布状态</span></div>
            <div class="metric-card"><strong>${data.runtime_loaded ? '是' : '否'}</strong><span>Runtime Loaded</span></div>
            <div class="metric-card"><strong>${escapeHtml(template.strategy_type || '')}</strong><span>策略类型</span></div>
            <div class="metric-card"><strong>${indexStatus.hybrid_index_built_on_load ? '已构建' : '不需要'}</strong><span>混合检索索引</span></div>
        </div>
        <div class="compact-item">
            <div class="compact-item-title"><span>${escapeHtml(d.app_id)} / ${escapeHtml(d.environment)}</span><span class="pill">${escapeHtml(d.deployment_id)}</span></div>
            <div class="compact-item-meta">published=${escapeHtml(d.published_at || '')} · tree=${escapeHtml(d.intent_tree_version_id || '')} · strategy=${escapeHtml(d.strategy_config_id || '')}</div>
        </div>
        <div class="snapshot-grid">
            <details class="snapshot-block">
                <summary>发布绑定的意图树版本 · ${escapeHtml(tree.name || '')} · ${(tree.intents || []).length} 个意图</summary>
                <pre class="json-block">${escapeHtml(JSON.stringify(tree, null, 2))}</pre>
            </details>
            <details class="snapshot-block">
                <summary>发布绑定的策略快照 · ${escapeHtml(template.name || '')}</summary>
                <pre class="json-block">${escapeHtml(JSON.stringify({
                    strategy_config: data.strategy_config || {},
                    strategy_template: data.strategy_template || {},
                    merged_params: data.merged_params || {}
                }, null, 2))}</pre>
            </details>
        </div>
    `;
}

async function rollbackDeployment(deploymentId) {
    if (!confirm('确定要回滚到这个发布版本吗？这会创建一条新的 active 发布记录并热更新运行时。')) return;
    const box = document.getElementById('deployment-detail');
    box.className = 'eval-result show';
    box.innerHTML = '<div class="result-panel show">回滚中...</div>';
    try {
        const resp = await fetch('/admin/deployments/' + deploymentId + '/rollback', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
            const routerReload = data.router_reload;
            if (routerReload && routerReload.ok === false) {
                const reason = routerReload.error || routerReload.response?.detail || routerReload.status_code || 'Unknown';
                box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">本地回滚成功，但识别服务热更新失败：</span>' + escapeHtml(reason) + '<br>新发布：' + escapeHtml(data.deployment.deployment_id) + '</div>';
            } else {
                box.innerHTML = '<div class="result-panel show"><span style="color:var(--success)">已回滚并创建新发布：</span>' + escapeHtml(data.deployment.deployment_id) + '</div>';
            }
            await loadReleaseData();
        } else {
            box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">回滚失败: ' + escapeHtml(data.detail || 'Unknown') + '</span></div>';
        }
    } catch (e) {
        box.innerHTML = '<div class="result-panel show"><span style="color:var(--danger)">回滚请求失败</span></div>';
    }
}

function onReleaseIntentTreeChange() {
    const treeId = document.getElementById('release-intent-tree').value;
    const appInput = document.getElementById('release-app-id');
    appInput.value = treeId || '';
}

async function publishDeployment() {
    const environment = document.getElementById('release-environment').value;
    const intentTreeId = document.getElementById('release-intent-tree').value;
    const appId = intentTreeId;
    document.getElementById('release-app-id').value = appId;
    const strategyValue = document.getElementById('release-strategy-config').value;
    const status = document.getElementById('release-status');
    if (!appId || !intentTreeId) {
        alert('请填写应用并选择意图树');
        return;
    }
    if (!strategyValue || !strategyValue.startsWith('config:')) {
        alert('请选择已保存的策略配置；模板需要先在“策略配置”页保存为配置后再发布');
        return;
    }
    try {
        status.textContent = '发布中...';
        status.className = 'status-msg show';
        const payload = {
            app_id: appId,
            environment,
            intent_tree_id: intentTreeId
        };
        payload.strategy_config_id = strategyValue.replace('config:', '');
        const resp = await fetch('/admin/deployments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (resp.ok) {
            const routerReload = data.router_reload;
            if (routerReload && routerReload.ok === false) {
                status.textContent = '本地发布成功，但识别服务热更新失败：' + (routerReload.error || routerReload.response?.detail || routerReload.status_code || 'Unknown');
                status.className = 'status-msg error show';
            } else {
                status.textContent = '发布成功：' + data.deployment.deployment_id;
                status.className = 'status-msg success show';
            }
            loadReleaseData();
        } else {
            status.textContent = '发布失败：' + (data.detail || 'Unknown');
            status.className = 'status-msg error show';
        }
    } catch (e) {
        status.textContent = '发布请求失败';
        status.className = 'status-msg error show';
    }
    setTimeout(() => status.classList.remove('show'), 5000);
}

// -------- Init --------

loadIntents();
loadConfig();
loadLlmConfig();
loadReleaseData();
loadEvalData();
'''
