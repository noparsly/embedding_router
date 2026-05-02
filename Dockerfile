FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY server_tencent.py .
COPY admin_server.py .
COPY mcp_client.py .
COPY intent_router/ ./intent_router/

EXPOSE 8000 8001

CMD ["python3", "-m", "uvicorn", "admin_server:app", "--host", "0.0.0.0", "--port", "8001"]
