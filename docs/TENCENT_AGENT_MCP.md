# 腾讯智能体开发平台 MCP 接入说明

本文记录将 Lighthouse 上的在线意图识别 MCP 服务接入腾讯云智能体开发平台的配置方式和踩坑点。

## 当前可用配置

腾讯智能体开发平台新建 MCP 插件时，优先使用：

```text
接入类型：可流式传输的 HTTP（streamableHttp）
URL：http://49.235.149.139/mcp
Header：不填
请求超时时长：60 秒
执行超时时长：300 秒
```

不要优先使用 `:8000` 端口地址。当前服务器已经通过 `/opt/adp-support/deploy/nginx/default.conf` 将公网 80 端口的 `/mcp` 反代到意图识别 router 服务。

如果必须使用 SSE，可尝试：

```text
接入类型：sse
URL：http://49.235.149.139/mcp/intent_router/sse
```

但腾讯平台当前使用 `streamableHttp + /mcp` 更稳定。

## MCP 工具

当前 MCP Server 暴露一个工具：

```text
route_intent
```

工具参数：

```json
{
  "query": "用户问题",
  "app_id": "bond_qa",
  "environment": "prod",
  "visible_intent_ids": ["可选意图ID白名单"]
}
```

为了兼容腾讯平台的插件校验，`app_id` 已改为可选，默认值为 `bond_qa`；`environment` 默认 `prod`。平台只要能传入 `query` 就可以完成调用。

## 验证命令

从本地或任意公网机器验证：

```bash
curl http://49.235.149.139/mcp
```

预期返回 MCP 服务信息，包含：

```json
{
  "name": "embedding-intent-router-tencent",
  "capabilities": {
    "tools": {
      "listChanged": false
    }
  }
}
```

验证工具列表：

```bash
curl -X POST http://49.235.149.139/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-protocol-version: 2025-11-25" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'
```

验证工具调用：

```bash
curl -X POST http://49.235.149.139/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-protocol-version: 2025-11-25" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "route_intent",
      "arguments": {
        "query": "某公司债券即将回售，需要准备什么材料？"
      }
    }
  }'
```

## 本次踩坑记录

### URL 格式必须准确

错误写法：

```text
http://49.235.149.139/:8000/mcp
```

正确写法：

```text
http://49.235.149.139:8000/mcp
```

更推荐经过 Nginx 反代后的标准端口：

```text
http://49.235.149.139/mcp
```

### 腾讯平台会带 Origin

腾讯智能体平台发起 `POST /mcp` 时可能带 `Origin`。旧逻辑默认只允许 localhost，导致返回 `403`。

当前修复点在 `server_tencent.py`：

```python
MCP_ALLOWED_ORIGINS = _origin_env

if origin and MCP_ALLOWED_ORIGINS and "*" not in MCP_ALLOWED_ORIGINS and origin not in MCP_ALLOWED_ORIGINS:
    return Response(status_code=403)
```

生产环境 `MCP_ALLOWED_ORIGINS` 为空表示不限制 Origin；如需显式放行，也可设置：

```bash
MCP_ALLOWED_ORIGINS=*
```

### 选择 streamableHttp，不要混用 SSE URL

腾讯平台 `streamableHttp` 应配置：

```text
http://49.235.149.139/mcp
```

不要把 `streamableHttp` 配到：

```text
http://49.235.149.139/mcp/intent_router/sse
```

旧日志里出现过腾讯平台对 `/mcp/intent_router/sse` 发 `POST`，会触发 `405 Method Not Allowed`。当前服务已加兼容入口，但配置上仍应避免混用。

### tools/list 的 schema 要保守

腾讯平台会在保存插件时读取 `tools/list` 并校验工具定义。过复杂或不完全兼容的 JSON Schema 会导致平台弹出：

```text
460009-MCP server连接失败
```

本次确认过的风险点：

- `outputSchema` 中使用 tuple 风格数组 schema，平台可能校验失败。
- `title` 字段不一定被平台接受。
- `app_id` 作为必填业务参数，可能增加平台保存阶段校验失败概率。

当前工具定义保守保留：

```json
{
  "name": "route_intent",
  "description": "Route a user query to the best intent using the published app runtime.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "app_id": {"type": "string", "default": "bond_qa"},
      "environment": {"type": "string", "default": "prod"}
    },
    "required": ["query"]
  }
}
```

### `202 Accepted` 是正常现象

腾讯平台初始化 MCP 时会发送没有 `id` 的 notification。服务端按 MCP 语义返回 `202 Accepted` 是正常的，不是失败。

日志中看到类似序列是正常的：

```text
POST /mcp -> 200
POST /mcp -> 202
POST /mcp -> 200
```

## 服务器改动位置

意图识别服务：

```bash
cd ~/intent-router
docker compose up -d --build router
docker logs --tail 120 intent-router-online
```

公网 Nginx 反代：

```bash
cd /opt/adp-support
docker compose up -d --build nginx
docker logs --tail 80 adp-support-nginx-1
```

Nginx 反代配置位于：

```text
/opt/adp-support/deploy/nginx/default.conf
```

关键反代逻辑：

```nginx
location /mcp {
    proxy_pass http://intent_router;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

## 排障步骤

1. 看腾讯平台请求有没有打到 Nginx：

```bash
ssh adp-lighthouse
docker logs --tail 120 adp-support-nginx-1
```

2. 看 router 是否返回了 403、405 或 500：

```bash
docker logs --tail 160 intent-router-online
```

3. 本地模拟腾讯平台的 Origin：

```bash
curl -i -X POST http://49.235.149.139/mcp \
  -H "Origin: https://lke.cloud.tencent.com" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-11-25",
      "capabilities": {}
    }
  }'
```

4. 如果腾讯平台仍报 `460009`，但日志里三步请求都是 `200/202/200`，优先怀疑 `tools/list` 返回的工具 schema 不被平台接受。
