# Intent Router Ops

轻量意图识别运营平台和在线 MCP 路由服务。

当前部署形态只保留：

- 腾讯云在线 Embedding
- 混合检索策略：Embedding + BM25
- 大模型 Prompt 策略
- 意图树运营、策略配置、在线测试、批量评测、发布和回滚
- HTTP `/v1/route` 与 MCP `/mcp` 对外服务

## 架构

```text
admin_server.py        运营平台，端口 8001
server_tencent.py      在线意图识别服务，端口 8000
intent_router/         核心路由、策略、存储、运行时
data/                  运行态数据目录，生产环境挂载持久化卷，不提交 Git
```

运营链路：

```text
配置意图树 -> 创建策略配置 -> 在线/批量测试 -> 发布 -> 在线服务按 app_id 路由
```

## 本地 Docker 启动

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env`，至少配置：

```bash
TENCENT_SECRET_ID=your-secret-id
TENCENT_SECRET_KEY=your-secret-key
ADMIN_API_KEY=change-me
ROUTER_ADMIN_API_KEY=change-me-router-admin-key
```

如使用大模型 Prompt 策略，再配置：

```bash
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

3. 启动：

```bash
docker compose up -d --build
```

4. 访问：

- 运营平台：http://localhost:8001/admin/
- 在线服务：http://localhost:8000/health

## 腾讯云 Lighthouse 部署

服务器安装 Docker 和 Compose Plugin 后：

```bash
git clone <your-repo-url> intent-router
cd intent-router
cp .env.example .env
vim .env
docker compose up -d --build
```

生产建议：

- 只对可信 IP 暴露 `8001`，或放到 Nginx + HTTPS + Basic/Auth 后面
- `ADMIN_API_KEY` 和 `ROUTER_ADMIN_API_KEY` 必须设置强随机值
- `data/` 使用服务器持久化目录，升级镜像时不要删除
- 不要把 `.env`、`data/admin_config.json`、云厂商密钥提交到 GitHub

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `APP_ENV` | 运行环境，建议生产为 `prod` |
| `DATA_DIR` | 运行态数据目录，Docker 中默认 `/app/data` |
| `EMBEDDING_PROVIDER` | 固定使用 `tencent` |
| `TENCENT_EMBEDDING_ENDPOINT` | 腾讯云 Embedding endpoint，默认 `https://lkeap.tencentcloudapi.com` |
| `TENCENT_SECRET_ID` | 腾讯云 Secret ID |
| `TENCENT_SECRET_KEY` | 腾讯云 Secret Key |
| `TENCENT_MODEL` | Embedding 模型，默认 `sn-large-multi-language-v0.2.5` |
| `ADMIN_API_KEY` | 运营平台 `/admin*` 接口密钥 |
| `ROUTER_ADMIN_API_KEY` | 运营平台热更新在线服务时使用的密钥 |
| `ROUTER_SERVICE_URL` | admin 访问 router 的地址，compose 内为 `http://router:8000` |
| `LLM_API_KEY` | OpenAI-compatible LLM Key |
| `LLM_BASE_URL` | OpenAI-compatible Base URL |
| `LLM_MODEL` | LLM 模型名 |

## API

### HTTP 路由

```bash
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "bond_qa",
    "query": "某公司债券即将回售，需要准备什么材料？",
    "environment": "prod"
  }'
```

### MCP 调用

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-protocol-version: 2025-11-25" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "route_intent",
      "arguments": {
        "app_id": "bond_qa",
        "query": "某公司债券即将回售，需要准备什么材料？"
      }
    }
  }'
```

## 开发验证

```bash
python3 -m py_compile admin_server.py server_tencent.py mcp_client.py intent_router/*.py
python3 test_router_core.py
docker compose config
```

## Git 管理约定

提交代码：

- `admin_server.py`
- `server_tencent.py`
- `intent_router/`
- `Dockerfile`
- `docker-compose.yml`
- `requirements*.txt`
- `.env.example`
- `README.md`

不要提交：

- `.env`
- `data/`
- `config/*.local.yaml`
- Python/Node 缓存
- 镜像 tar 包
- 临时测试数据和个人文档
