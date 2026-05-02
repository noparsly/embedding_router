FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.cloud.tencent.com


WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir \
    -i ${PIP_INDEX_URL} \
    --trusted-host ${PIP_TRUSTED_HOST} \
    -r requirements-docker.txt

COPY server_tencent.py .
COPY admin_server.py .
COPY mcp_client.py .
COPY intent_router/ ./intent_router/

EXPOSE 8000 8001

CMD ["python3", "-m", "uvicorn", "admin_server:app", "--host", "0.0.0.0", "--port", "8001"]
