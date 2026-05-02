# 操作手册

这份手册记录意图识别运营平台从本地修改、推送 GitHub、更新 Lighthouse、验证上线到排障回滚的完整流程。

## 角色边界

```text
Mac 本地仓库        写代码、改 UI、改部署文件、跑测试
GitHub             保存代码主线，不保存密钥和运行态数据
Lighthouse 服务器  运行 Docker 服务，保存生产 data/ 和 .env
运营平台           配置意图树、策略配置、测试、评测、发布和回滚
在线服务           对外提供 /v1/route 和 /mcp
```

## 目录约定

Mac 本地开发目录：

```bash
cd /Users/wanglei/Desktop/智能体建设/plugins/intent-router-clean
```

Lighthouse 服务器目录：

```bash
cd ~/intent-router
```

不要再用旧的 `simplified_embedding_router` 做日常开发，那个目录保留为历史参考即可。

## 日常开发流程

1. 在 Mac 本地拉取最新代码：

```bash
cd /Users/wanglei/Desktop/智能体建设/plugins/intent-router-clean
git pull
```

2. 修改代码。

3. 本地基础验证：

```bash
python3 -m py_compile admin_server.py server_tencent.py mcp_client.py intent_router/*.py
python3 test_router_core.py
docker compose config
```

4. 如果涉及 Docker 或前端页面，建议本地启动验证：

```bash
cp .env.example .env
# 按需填写 .env
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health/detail
```

5. 提交代码：

```bash
git status
git add .
git commit -m "描述这次修改"
git push
```

## Lighthouse 更新流程

1. 登录服务器：

```bash
ssh ubuntu@你的服务器IP
```

2. 进入项目目录并拉取代码：

```bash
cd ~/intent-router
git pull
```

3. 重建并重启服务：

```bash
docker compose up -d --build
```

如果只改了 `.env`：

```bash
docker compose up -d --force-recreate
```

4. 验证容器：

```bash
docker compose ps
docker compose logs --tail=80 admin
docker compose logs --tail=80 router
```

5. 验证接口：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health/detail
```

6. 打开运营平台：

```text
http://服务器公网IP:8001/admin/
```

## 运行态数据原则

`data/` 是生产运行态数据，包含：

- 意图树
- 意图树版本
- 策略配置
- 发布记录
- 评测集和评测记录
- admin 本地保存的连接配置

原则：

- `data/` 不提交 GitHub。
- 不要用 Mac 本地 `data/` 覆盖服务器 `data/`。
- 服务器升级代码时只 `git pull` 和 `docker compose up -d --build`。
- 如需备份，备份服务器上的 `~/intent-router/data`。

备份命令示例：

```bash
cd ~/intent-router
tar -czf intent-router-data-$(date +%Y%m%d-%H%M%S).tar.gz data
```

## .env 配置

服务器 `.env` 不提交 GitHub。首次部署：

```bash
cp .env.example .env
vim .env
```

关键配置：

```bash
APP_ENV=prod
DATA_DIR=data
INTENT_CACHE_DIR=data/intents
EMBEDDING_PROVIDER=tencent

TENCENT_EMBEDDING_ENDPOINT=https://lkeap.tencentcloudapi.com
TENCENT_SECRET_ID=你的SecretId
TENCENT_SECRET_KEY=你的SecretKey

ROUTER_SERVICE_URL=http://router:8000
AUTO_PUBLISH_TO_ROUTER=true
ROUTER_ADMIN_API_KEY=强随机字符串

# 当前版本如果直接浏览器访问 admin，ADMIN_API_KEY 建议留空，
# 用 Lighthouse 安全组限制 8001 访问来源。
ADMIN_API_KEY=
```

生成随机 key：

```bash
openssl rand -hex 32
```

## 运营平台使用流程

标准运营链路：

```text
创建/维护意图树
-> 创建策略配置
-> 在线单条测试
-> 上传评测集并批量评测
-> 选择意图树和策略配置发布
-> 调 /v1/route 或 /mcp 验证
```

注意：

- 意图 `id` 必须是英文/数字/点/下划线/中划线，例如 `laws_regulations_query`。
- 意图 `name` 可以写中文。
- 发布中心只选择已经保存的策略配置，不直接发布模板。
- 混合检索首次测试或发布会构建向量索引，第一次慢一点是正常的。

## 在线服务验证

HTTP：

```bash
curl -X POST http://127.0.0.1:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "bond_qa",
    "query": "某公司债券即将回售，需要准备什么材料？",
    "environment": "prod"
  }'
```

MCP：

```bash
python3 mcp_client.py \
  --url http://127.0.0.1:8000/mcp \
  --app-id bond_qa \
  --query "某公司债券即将回售，需要准备什么材料？"
```

## 常见问题

### 访问 admin 显示 Invalid admin API key

原因：配置了 `ADMIN_API_KEY`，但浏览器页面不会自动带 `x-admin-api-key`。

当前建议：

```bash
ADMIN_API_KEY=
```

然后用 Lighthouse 安全组限制 `8001` 只允许自己的 IP 访问。

修改后重启：

```bash
docker compose up -d --force-recreate admin
```

### pip 下载超时

Dockerfile 默认使用腾讯云 PyPI 镜像。如果仍然慢，可以改用阿里云：

```bash
docker compose build --no-cache \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  --build-arg PIP_TRUSTED_HOST=mirrors.aliyun.com
docker compose up -d
```

### requirements-docker.txt not found

确认服务器代码完整：

```bash
ls -lah
ls -lah requirements-docker.txt
sed -n '1,80p' Dockerfile
```

Dockerfile 里必须有：

```dockerfile
COPY requirements-docker.txt .
```

### 修改代码后页面没变化

服务器上重新构建并重启：

```bash
docker compose up -d --build
```

浏览器强制刷新页面。

### 发布后在线服务没生效

检查 admin 能否访问 router：

```bash
docker compose logs --tail=120 admin
docker compose logs --tail=120 router
```

确认 `.env`：

```bash
ROUTER_SERVICE_URL=http://router:8000
AUTO_PUBLISH_TO_ROUTER=true
ROUTER_ADMIN_API_KEY=两边一致
```

## 回滚

运营平台发布中心支持回滚到历史发布版本。回滚会创建一条新的 active 发布记录，并热更新在线服务。

如果容器级别要回退代码：

```bash
cd ~/intent-router
git log --oneline -5
git checkout <commit>
docker compose up -d --build
```

回到最新代码：

```bash
git checkout main
git pull
docker compose up -d --build
```

## 安全清单

上线前确认：

- `.env` 不在 Git。
- `data/` 不在 Git。
- `ADMIN_API_KEY` 如留空，必须用安全组限制 8001。
- `ROUTER_ADMIN_API_KEY` 使用强随机值。
- 腾讯云 Secret 和 LLM Key 不出现在 README、代码或 Git 历史里。
- 8000 如对外开放，应由调用方网络边界或上层网关保护。
