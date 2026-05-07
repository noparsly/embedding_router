#!/usr/bin/env python3
"""
- 提供HTTP接口，接收用户查询并返回意图识别结果
- 使用腾讯云在线 embedding provider
- 支持两种协议：简单的HTTP POST接口和标准的MCP协议
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from intent_router.router import IntentRouter, RouterConfig, IntentDef
from intent_router.tencent_provider import TencentEmbeddingProvider
from intent_router.cache import IntentCache, intent_defs_from_dicts
from intent_router.runtime import RuntimeRegistry
from intent_router.storage import FileRepository
from intent_router.domain import DEFAULT_ENVIRONMENT
from intent_router.settings import APP_ENV, get_bool, get_float, get_int, get_list, get_str

# ----------------------------
# 通用路由参数
# ----------------------------

ABS_THRESHOLD = get_float("ABS_THRESHOLD", 0.6)
GAP_THRESHOLD = get_float("GAP_THRESHOLD", 0.03)
TOP_K = get_int("TOP_K", 3)
AMBIGUOUS_ACTION = get_str("AMBIGUOUS_ACTION", "clarify")  # clarify | pick_top1
WEIGHT_INTRO = get_float("WEIGHT_INTRO", 0.5)
QUESTION_TOP_K = get_int("QUESTION_TOP_K", 5)
ALPHA_SHORT = get_float("ALPHA_SHORT", 0.6)
ALPHA_LONG = get_float("ALPHA_LONG", 0.8)
ENABLE_BM25 = get_bool("ENABLE_BM25", True)
SHORT_QUERY_LEN = get_int("SHORT_QUERY_LEN", 10)
EXAMPLE_QUESTION_THRESHOLD = get_float("EXAMPLE_QUESTION_THRESHOLD", 0.95)

# ----------------------------
# 腾讯 embedding 参数
# ----------------------------

TENCENT_EMBEDDING_ENDPOINT = get_str("TENCENT_EMBEDDING_ENDPOINT", "")
TENCENT_SECRET_ID = get_str("TENCENT_SECRET_ID", "")
TENCENT_SECRET_KEY = get_str("TENCENT_SECRET_KEY", "")
TENCENT_API_KEY = get_str("TENCENT_API_KEY", "")
TENCENT_MODEL = get_str("TENCENT_MODEL", "sn-large-multi-language-v0.2.5")
TENCENT_TEXT_TYPE = get_str("TENCENT_TEXT_TYPE", "")
TENCENT_INSTRUCTION = get_str("TENCENT_INSTRUCTION", "")
TENCENT_SERVICE = get_str("TENCENT_SERVICE", "lkeap")
TENCENT_VERSION = get_str("TENCENT_VERSION", "2024-05-22")
TENCENT_REGION = get_str("TENCENT_REGION", "ap-guangzhou")
TENCENT_TOKEN = get_str("TENCENT_TOKEN", "")
TENCENT_TIMEOUT = get_int("TENCENT_TIMEOUT", 60)
TENCENT_MAX_BATCH_SIZE = get_int("TENCENT_MAX_BATCH_SIZE", 7)

# LLM Provider 配置（用于 llm_prompt 策略）
LLM_API_KEY = get_str("LLM_API_KEY", "")
LLM_BASE_URL = get_str("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = get_str("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = get_float("LLM_TIMEOUT", 5)
ROUTER_ADMIN_API_KEY = get_str("ROUTER_ADMIN_API_KEY", "")
DATA_DIR = get_str("DATA_DIR", "data")
INTENT_CACHE_DIR = get_str("INTENT_CACHE_DIR", "data/intents")
reload_lock = threading.RLock()


class RouteRequest(BaseModel):
    """意图识别请求模型"""
    query: str = Field(..., description="用户查询文本")
    visible_intent_ids: Optional[List[str]] = Field(None, description="可见的意图ID列表（权限过滤）")


class RouteResponse(BaseModel):
    """意图识别响应模型"""
    selected_intent_id: str
    selected_intent_name: str
    method: str
    confidence: float
    top_k: List[tuple]
    is_ood: bool
    is_ambiguous: bool


app = FastAPI(title="Embedding Intent Router (Tencent)", version="0.1")


def _build_provider() -> TencentEmbeddingProvider:
    """构建腾讯云 Embedding 提供者"""
    if not TENCENT_EMBEDDING_ENDPOINT:
        raise RuntimeError("TENCENT_EMBEDDING_ENDPOINT is required for server_tencent.py")
    return TencentEmbeddingProvider(
        endpoint=TENCENT_EMBEDDING_ENDPOINT,
        api_key=TENCENT_API_KEY,
        secret_id=TENCENT_SECRET_ID or None,
        secret_key=TENCENT_SECRET_KEY or None,
        model=TENCENT_MODEL,
        text_type=TENCENT_TEXT_TYPE or None,
        instruction=TENCENT_INSTRUCTION or None,
        service=TENCENT_SERVICE,
        version=TENCENT_VERSION,
        region=TENCENT_REGION,
        token=TENCENT_TOKEN,
        timeout=TENCENT_TIMEOUT,
        max_batch_size=TENCENT_MAX_BATCH_SIZE,
    )


def _build_router_config() -> RouterConfig:
    """构建路由器配置"""
    return RouterConfig(
        top_k=TOP_K,
        abs_threshold=ABS_THRESHOLD,
        gap_threshold=GAP_THRESHOLD,
        ambiguous_action=AMBIGUOUS_ACTION,
        weight_intro=WEIGHT_INTRO,
        question_top_k=QUESTION_TOP_K,
        alpha_short=ALPHA_SHORT,
        alpha_long=ALPHA_LONG,
        enable_bm25=ENABLE_BM25,
        short_query_len=SHORT_QUERY_LEN,
        example_question_threshold=EXAMPLE_QUESTION_THRESHOLD,
    )


def _build_llm_provider_from_config(config: Optional[Dict[str, Any]] = None):
    """构建 LLM 提供者"""
    cfg = config or {}
    api_key = cfg.get("api_key", LLM_API_KEY)
    if not api_key:
        return None
    try:
        from intent_router.llm_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=cfg.get("base_url", LLM_BASE_URL),
            model=cfg.get("model", LLM_MODEL),
            max_tokens=int(cfg.get("max_tokens", 256)),
            temperature=float(cfg.get("temperature", 0.1)),
            timeout=float(cfg.get("timeout", LLM_TIMEOUT)),
        )
    except Exception as e:
        print(f"Warning: Failed to initialize LLM provider: {e}")
        return None

llm_provider = _build_llm_provider_from_config()
embedding_provider = _build_provider()
repository = FileRepository(DATA_DIR)

# 意图缓存管理器（兼容 cache_key 老接口）
intent_cache = IntentCache(
    provider=embedding_provider,
    config=_build_router_config(),
    persist_dir=INTENT_CACHE_DIR,
)
runtime_registry = RuntimeRegistry(repository=repository, provider=embedding_provider, llm_provider=llm_provider)


def _require_router_admin(req: Request) -> Optional[JSONResponse]:
    if ROUTER_ADMIN_API_KEY and req.headers.get("x-admin-api-key") != ROUTER_ADMIN_API_KEY:
        return JSONResponse({"detail": "Invalid router admin API key"}, status_code=401)
    return None


def _router_config_from_dict(data: Optional[Dict[str, Any]]) -> RouterConfig:
    data = data or {}
    return RouterConfig(
        top_k=int(data.get("top_k", TOP_K)),
        abs_threshold=float(data.get("abs_threshold", ABS_THRESHOLD)),
        gap_threshold=float(data.get("gap_threshold", GAP_THRESHOLD)),
        ambiguous_action=data.get("ambiguous_action", AMBIGUOUS_ACTION),
        weight_intro=float(data.get("weight_intro", 0.5)),
        alpha_short=float(data.get("alpha_short", 0.6)),
        alpha_long=float(data.get("alpha_long", 0.8)),
        enable_bm25=bool(data.get("enable_bm25", True)),
        short_query_len=int(data.get("short_query_len", 10)),
        question_top_k=int(data.get("question_top_k", 5)),
        example_question_threshold=float(data.get("example_question_threshold", 0.95)),
    )


def _reload_cache_key(cache_key: str):
    with reload_lock:
        deployment = repository.get_deployment(cache_key)
        if deployment is not None:
            bundle = runtime_registry.reload_deployment(deployment)
            return {"deployment_id": cache_key, "deployment_key": bundle.deployment_key, "intent_count": len(bundle.intent_tree_version.intents)}
        intents = intent_cache.load_intent_config(cache_key)
        if not intents:
            raise ValueError(f"No intent config or deployment found for key: {cache_key}")
        intent_cache.invalidate(cache_key)
        router_obj = intent_cache.get_or_build(cache_key, intents)
        return {"cache_key": cache_key, "intent_count": len(router_obj.intents)}


def _rebuild_intent_cache(router_config: RouterConfig, llm_config: Optional[Dict[str, Any]] = None) -> int:
    global intent_cache, llm_provider
    llm_config = llm_config or {}
    llm_provider = _build_llm_provider_from_config(llm_config)
    with reload_lock:
        intent_cache = IntentCache(
            provider=embedding_provider,
            config=router_config,
            persist_dir=INTENT_CACHE_DIR,
        )
        runtime_registry.llm_provider = llm_provider
        runtime_registry.reload_all_active()
        return intent_cache.preload_from_disk()


@app.get("/health")
def health():
    """健康检查接口"""
    return {"ok": True}


@app.get("/health/detail")
def health_detail():
    """详细健康检查，便于区分环境和运行时配置。"""
    return {
        "status": "healthy",
        "environment": APP_ENV,
        "embedding_endpoint_configured": bool(TENCENT_EMBEDDING_ENDPOINT),
        "cache_count": len(intent_cache.list_caches()),
        "runtime_bundle_count": len(runtime_registry.list_bundles()),
        "llm_provider_configured": llm_provider is not None,
    }


@app.get("/admin/cache")
def admin_cache(req: Request):
    rejected = _require_router_admin(req)
    if rejected:
        return rejected
    return {"caches": intent_cache.list_caches(), "runtime_bundles": runtime_registry.list_bundles()}


@app.post("/admin/reload-cache/{cache_key}")
def admin_reload_cache(cache_key: str, req: Request):
    rejected = _require_router_admin(req)
    if rejected:
        return rejected
    try:
        return {"ok": True, **_reload_cache_key(cache_key)}
    except Exception as e:
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


@app.post("/admin/reload-all")
def admin_reload_all(req: Request):
    rejected = _require_router_admin(req)
    if rejected:
        return rejected
    try:
        with reload_lock:
            count = intent_cache.preload_from_disk()
            runtime_count = runtime_registry.reload_all_active()
        return {"ok": True, "loaded": count, "runtime_loaded": runtime_count}
    except Exception as e:
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


@app.post("/admin/reload-deployment/{deployment_id}")
def admin_reload_deployment(deployment_id: str, req: Request):
    rejected = _require_router_admin(req)
    if rejected:
        return rejected
    try:
        deployment = repository.get_deployment(deployment_id)
        if deployment is None:
            return JSONResponse({"ok": False, "detail": f"Deployment not found: {deployment_id}"}, status_code=404)
        bundle = runtime_registry.reload_deployment(deployment)
        return {"ok": True, "bundle": bundle.summary()}
    except Exception as e:
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


@app.put("/admin/runtime-config")
async def admin_runtime_config(req: Request):
    rejected = _require_router_admin(req)
    if rejected:
        return rejected
    try:
        body = await req.json()
        router_config = _router_config_from_dict((body or {}).get("router_config"))
        loaded = _rebuild_intent_cache(router_config, (body or {}).get("llm_config"))
        return {"ok": True, "loaded": loaded, "runtime_bundles": runtime_registry.list_bundles()}
    except Exception as e:
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=400)


class RouteRequestWithCustomIntents(RouteRequest):
    """带有自定义意图的路由请求"""
    intents: Optional[List[Dict[str, Any]]] = None
    abs_threshold: Optional[float] = None
    gap_threshold: Optional[float] = None


class RouteRequestV1(BaseModel):
    """V1 版本意图识别请求（发布态接口：应用 + 环境）。"""
    app_id: str = Field(..., description="应用 ID。当前一个应用对应一个意图树")
    query: str = Field(..., description="用户查询文本")
    environment: str = Field("prod", description="环境：prod/test/dev")
    visible_intent_ids: Optional[List[str]] = Field(None, description="可见的意图ID列表")


class RouteResponseV1(BaseModel):
    """V1 版本意图识别响应"""
    selected_intent_id: str
    selected_intent_name: str
    method: str
    confidence: float
    top_k: List[tuple]
    is_ood: bool
    is_ambiguous: bool
    cached: bool = Field(None, description="是否命中缓存")


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequestWithCustomIntents):
    """HTTP 意图识别接口"""
    if not req.intents:
        raise HTTPException(
            status_code=400,
            detail="The legacy /route endpoint requires intents. Use /v1/route with cache_key for managed intents."
        )
    try:
        intent_defs = intent_defs_from_dicts(req.intents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid intents format: {str(e)}")

    temp_router = IntentRouter(
        intents=intent_defs,
        provider=intent_cache._provider,
        config=_build_router_config(),
    )
    res = temp_router.route(
        req.query,
        visible_intent_ids=req.visible_intent_ids,
        abs_threshold=req.abs_threshold,
        gap_threshold=req.gap_threshold,
    )
    return RouteResponse(
        selected_intent_id=res.selected_intent_id,
        selected_intent_name=res.selected_intent_name,
        method=res.method,
        confidence=res.confidence,
        top_k=[(i, n, s) for (i, n, s) in res.top_k],
        is_ood=res.is_ood,
        is_ambiguous=res.is_ambiguous,
    )


@app.post("/v1/route")
def route_v1(req: RouteRequestV1):
    """V1 版本 HTTP 意图识别接口：应用 + 环境发布态路由。"""
    try:
        res, bundle = runtime_registry.route(
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


# 尝试从磁盘预加载已保存的意图配置和发布态配置
try:
    intent_cache.preload_from_disk()
    runtime_registry.reload_all_active()
except Exception:
    pass


MCP_PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26"]
MCP_SERVER_NAME = get_str("MCP_SERVER_NAME", "embedding-intent-router-tencent")
MCP_SERVER_VERSION = get_str("MCP_SERVER_VERSION", "0.1")
MCP_PLUGIN_NAME = get_str("MCP_PLUGIN_NAME", "intent_router")
_origin_env = get_list("MCP_ALLOWED_ORIGINS", [])
MCP_ALLOWED_ORIGINS = _origin_env
_mcp_sessions: Dict[str, Dict[str, str]] = {}


def _rpc_error(req_id, code: int, message: str, data=None):
    """构建 MCP 错误响应"""
    err = {"code": int(code), "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _rpc_result(req_id, result: dict):
    """构建 MCP 成功响应"""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _tool_definitions():
    """返回 MCP 工具定义列表"""
    return [
        {
            "name": "route_intent",
            "description": "Route a user query to the best intent using the published app runtime.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "description": "Published application id", "default": "bond_qa"},
                    "environment": {"type": "string", "description": "Environment, default prod", "default": "prod"},
                    "query": {"type": "string", "description": "User query"},
                    "visible_intent_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional allowlist of intent ids",
                    },
                },
                "required": ["query"],
            },
        }
    ]


def _check_origin(req: Request) -> Optional[Response]:
    """校验跨域来源，未通过时返回 403 响应。"""
    origin = req.headers.get("origin")
    if origin and MCP_ALLOWED_ORIGINS and "*" not in MCP_ALLOWED_ORIGINS and origin not in MCP_ALLOWED_ORIGINS:
        return Response(status_code=403)
    return None


def _json_response(payload: dict, status_code: int = 200) -> JSONResponse:
    """统一 MCP JSON 响应头。"""
    response = JSONResponse(payload, status_code=status_code)
    response.headers["ngrok-skip-browser-warning"] = "1"
    return response


def _sse_headers() -> Dict[str, str]:
    """统一 SSE 响应头。"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "ngrok-skip-browser-warning": "1",
    }


async def _dispatch_mcp(body: dict, req: Request) -> Response:
    """处理 MCP JSON-RPC 请求。"""
    if isinstance(body, list):
        return _json_response(_rpc_error(None, -32600, "Batch requests not supported"), status_code=400)
    if not isinstance(body, dict):
        return _json_response(_rpc_error(None, -32600, "Invalid Request"), status_code=400)

    jsonrpc = body.get("jsonrpc")
    method = body.get("method")
    req_id = body.get("id", None)
    params = body.get("params") or {}

    if jsonrpc != "2.0":
        return _json_response(_rpc_error(req_id, -32600, "Invalid Request"), status_code=400)

    if method == "initialize":
        client_version = (params or {}).get("protocolVersion")
        if client_version not in MCP_PROTOCOL_VERSIONS:
            return _json_response(
                _rpc_error(
                    req_id,
                    -32602,
                    "Unsupported protocol version",
                    {"supported": MCP_PROTOCOL_VERSIONS, "requested": client_version},
                ),
                status_code=400,
            )
        result = {
            "protocolVersion": client_version,
            "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
            "capabilities": {"tools": {"listChanged": False}},
        }
        return _json_response(_rpc_result(req_id, result))

    protocol_header = req.headers.get("mcp-protocol-version")
    if protocol_header and protocol_header not in MCP_PROTOCOL_VERSIONS:
        return _json_response(
            _rpc_error(
                req_id,
                -32602,
                "Unsupported MCP-Protocol-Version header",
                {"supported": MCP_PROTOCOL_VERSIONS, "received": protocol_header},
            ),
            status_code=400,
        )

    if req_id is None:
        return Response(status_code=202)

    if method == "tools/list":
        return _json_response(_rpc_result(req_id, {"tools": _tool_definitions()}))

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name != "route_intent":
            return _json_response(_rpc_error(req_id, -32601, f"Unknown tool: {name}"), status_code=404)

        app_id = (arguments or {}).get("app_id") or "bond_qa"
        environment = (arguments or {}).get("environment") or "prod"
        query = (arguments or {}).get("query")
        visible = (arguments or {}).get("visible_intent_ids")

        if not isinstance(app_id, str) or not app_id.strip():
            return _json_response(_rpc_error(req_id, -32602, "Invalid params: app_id is required"), status_code=400)
        if not isinstance(query, str) or not query.strip():
            return _json_response(_rpc_error(req_id, -32602, "Invalid params: query is required"), status_code=400)
        if visible is not None and not (
            isinstance(visible, list) and all(isinstance(x, str) for x in visible)
        ):
            return _json_response(_rpc_error(req_id, -32602, "Invalid params: visible_intent_ids"), status_code=400)

        try:
            res, bundle = runtime_registry.route(
                app_id=app_id,
                environment=environment,
                query=query,
                visible_intent_ids=visible,
            )

            payload = {
                "selected_intent_id": res.selected_intent_id,
                "selected_intent_name": res.selected_intent_name,
                "method": res.method,
                "confidence": float(res.confidence),
                "top_k": [(i, n, float(s)) for (i, n, s) in res.top_k],
                "is_ood": bool(res.is_ood),
                "is_ambiguous": bool(res.is_ambiguous),
                "deployment": bundle.summary(),
            }
            return _json_response(
                _rpc_result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                        "structuredContent": payload,
                        "isError": False,
                    },
                )
            )
        except HTTPException:
            raise
        except Exception as e:
            return _json_response(_rpc_error(req_id, -32603, f"Internal error: {str(e)}"), status_code=500)

    return _json_response(_rpc_error(req_id, -32601, "Method not found"), status_code=404)


@app.get("/mcp")
def mcp_get():
    """MCP 协议 GET 探测入口，便于平台进行 streamableHttp 探测。"""
    return JSONResponse(
        {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "protocolVersions": MCP_PROTOCOL_VERSIONS,
            "capabilities": {"tools": {"listChanged": False}},
            "transports": {
                "streamableHttp": {"endpoint": "/mcp", "method": "POST"},
                "sse": {"endpoint": f"/mcp/{MCP_PLUGIN_NAME}/sse"},
            },
        }
    )


async def _sse_event_stream(endpoint: str):
    """提供平台探测所需的 SSE 事件流。"""
    initial = {
        "type": "endpoint",
        "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
        "endpoint": endpoint,
    }
    yield f"event: endpoint\ndata: {json.dumps(initial, ensure_ascii=False)}\n\n"
    while True:
        yield 'event: message\ndata: {"jsonrpc":"2.0","method":"ping"}\n\n'
        await asyncio.sleep(15)


@app.get("/sse")
@app.get("/mcp/sse")
async def mcp_sse():
    """SSE 探测入口，兼容平台的 sse 类型接入。"""
    endpoint = "/mcp"
    return StreamingResponse(_sse_event_stream(endpoint), media_type="text/event-stream", headers=_sse_headers())


@app.get("/mcp/{plugin_name}/sse")
async def mcp_plugin_sse(plugin_name: str):
    """兼容腾讯平台的插件式 SSE 握手入口。"""
    session_id = str(uuid.uuid4())
    endpoint = f"/mcp/{plugin_name}/message?sessionId={session_id}"
    _mcp_sessions[session_id] = {"plugin_name": plugin_name}
    return StreamingResponse(_sse_event_stream(endpoint), media_type="text/event-stream", headers=_sse_headers())


@app.post("/mcp/sse")
async def mcp_sse_post(req: Request):
    """兼容将 SSE 地址当作 Streamable HTTP 地址提交的 MCP 客户端。"""
    return await mcp_post(req)


@app.post("/mcp/{plugin_name}/sse")
async def mcp_plugin_sse_post(plugin_name: str, req: Request):
    """兼容将插件式 SSE 地址当作消息提交地址的 MCP 客户端。"""
    return await mcp_plugin_message(plugin_name, req)


@app.post("/mcp")
async def mcp_post(req: Request):
    """MCP 协议 POST 请求入口，处理所有 MCP 方法调用"""
    rejected = _check_origin(req)
    if rejected is not None:
        return rejected

    try:
        body = await req.json()
    except Exception:
        return _json_response(_rpc_error(None, -32700, "Parse error"), status_code=400)
    return await _dispatch_mcp(body, req)


@app.post("/")
async def mcp_root_post(req: Request):
    """兼容直接将根路径当作 MCP 入口的调用方。"""
    return await mcp_post(req)


@app.post("/mcp/{plugin_name}/message")
async def mcp_plugin_message(plugin_name: str, req: Request):
    """兼容腾讯平台的插件式 message 入口。"""
    rejected = _check_origin(req)
    if rejected is not None:
        return rejected

    session_id = req.query_params.get("sessionId")
    if session_id:
        session = _mcp_sessions.get(session_id)
        if session is None:
            _mcp_sessions[session_id] = {"plugin_name": plugin_name}
        elif session.get("plugin_name") != plugin_name:
            return _json_response(_rpc_error(None, -32600, "Invalid session"), status_code=400)

    try:
        body = await req.json()
    except Exception:
        return _json_response(_rpc_error(None, -32700, "Parse error"), status_code=400)
    return await _dispatch_mcp(body, req)
