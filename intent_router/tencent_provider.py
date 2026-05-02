#!/usr/bin/env python3
"""
腾讯embedding服务提供商

功能：
- 提供使用腾讯embedding API的实现
- 作为当前部署形态的在线 Embedding Provider
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np

from .router import EmbeddingProvider

try:
    import requests
except Exception:
    requests = None

class TencentEmbeddingProvider(EmbeddingProvider):
    """基于腾讯云 Embedding 服务的向量生成器实现"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: str = "lke-text-embedding-v1",
        text_type: Optional[str] = None,
        instruction: Optional[str] = None,
        service: str = "lkeap",
        version: str = "2024-05-22",
        region: str = "ap-guangzhou",
        token: str = "",
        timeout: int = 60,
        max_batch_size: int = 7,
        cache_size: int = 2000,
    ) -> None:
        """初始化腾讯云 Embedding 提供者，配置连接参数"""
        if requests is None:
            raise RuntimeError(
                "requests 未安装。请运行 pip install -r requirements.txt"
            )

        normalized = endpoint.rstrip("/").rstrip("\\")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
        if not parsed.netloc:
            raise ValueError(f"Invalid endpoint: {endpoint}")

        self.endpoint = normalized
        self.host = parsed.netloc
        self.api_key = api_key
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.model = model
        self.text_type = text_type
        self.instruction = instruction
        self.service = service
        self.version = version
        self.region = region
        self.token = token
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size

    def _build_headers(self, body: str) -> dict:
        """构建请求头（对齐 examples/Embedding_Client.py 的 TC3 协议）"""
        if not self.secret_id or not self.secret_key:
            raise RuntimeError("TencentEmbeddingProvider 缺少 secret_id/secret_key")

        timestamp = str(int(time.time()))
        content_type = "application/json; charset=utf-8"
        auth = _get_auth(
            secret_id=self.secret_id,
            secret_key=self.secret_key,
            host=self.host,
            content_type=content_type,
            timestamp=timestamp,
            body=body,
            service=self.service,
        )

        headers = {
            "Host": self.host,
            "X-TC-Timestamp": timestamp,
            "X-TC-Version": self.version,
            "X-TC-Action": "GetEmbedding",
            "X-TC-Region": self.region,
            "Authorization": auth,
            "Content-Type": content_type,
        }
        if self.token:
            headers["X-TC-Token"] = self.token
        return headers

    def _build_payload(self, texts: Sequence[str]) -> dict:
        """构建请求体，字段对齐 examples/Embedding_Client.py"""
        payload = {
            "Model": self.model,
            "Inputs": list(texts),
        }
        if self.text_type:
            payload["TextType"] = self.text_type
        if self.instruction:
            payload["Instruction"] = self.instruction
        return payload

    def _call_api(self, texts: Sequence[str]) -> list:
        """调用腾讯云 Embedding API"""
        url = f"{self.endpoint}/"
        payload = self._build_payload(texts)
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        headers = self._build_headers(payload_json)

        response = requests.post(
            url,
            headers=headers,
            data=payload_json.encode("utf-8"),
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()
        data = _extract_data(result)
        if not data:
            raise RuntimeError(f"Invalid response: {result}")

        embeddings = []
        for item in data:
            if "Embedding" not in item:
                raise RuntimeError(f"Missing embedding in response: {item}")
            embeddings.append(item["Embedding"])

        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}")

        return embeddings

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """将文本列表转换为向量（调用腾讯云 API）"""
        if not texts:
            return np.array([], dtype=np.float32)

        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_positions: list[int] = []

        for idx, text in enumerate(texts):
            key = self._text_cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue
            uncached_texts.append(text)
            uncached_positions.append(idx)

        for i in range(0, len(uncached_texts), self.max_batch_size):
            batch = uncached_texts[i:i + self.max_batch_size]
            positions = uncached_positions[i:i + self.max_batch_size]
            embeddings = self._call_api(batch)
            for text, embedding, pos in zip(batch, embeddings, positions):
                vec = np.asarray(embedding, dtype=np.float32)
                self._remember(text, vec)
                results[pos] = vec
            if i + self.max_batch_size < len(uncached_texts):
                time.sleep(0.1)  # 避免请求过于频繁

        if any(item is None for item in results):
            raise RuntimeError("Embedding cache assembly failed")
        embeddings_np = np.vstack([item for item in results if item is not None]).astype(np.float32)
        # 归一化
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("Embedding API returned zero-vector embeddings")
        embeddings_np = embeddings_np / norms

        return embeddings_np

    @staticmethod
    def _text_cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _remember(self, text: str, vec: np.ndarray) -> None:
        key = self._text_cache_key(text)
        if key in self._cache:
            self._cache[key] = vec
            return
        self._cache[key] = vec
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)


def _extract_data(result: dict) -> list:
    """兼容内部网关和腾讯云标准响应结构。"""
    if isinstance(result.get("Data"), list):
        return result["Data"]
    response = result.get("Response")
    if isinstance(response, dict) and isinstance(response.get("Data"), list):
        return response["Data"]
    return []


def _get_auth(
    secret_id: str,
    secret_key: str,
    host: str,
    content_type: str,
    timestamp: str,
    body: str,
    service: str,
) -> str:
    canonical_headers = f"content-type:{content_type}\nhost:{host}\n"
    signed_headers = "content-type;host"
    hashed_request_payload = _sha256_hex(body.encode("utf-8"))
    canonical_request = (
        "POST\n"
        "/\n"
        "\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{hashed_request_payload}"
    )
    date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = _sha256_hex(canonical_request.encode("utf-8"))
    string_to_sign = (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        f"{hashed_canonical_request}"
    )
    signature = _calculate_signature(secret_key, date, service, string_to_sign)
    return (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )


def _calculate_signature(secret_key: str, date: str, service: str, string_to_sign: str) -> str:
    secret_date = _hmac256(f"TC3{secret_key}".encode("utf-8"), date)
    secret_service = _hmac256(secret_date, service)
    secret_signing = _hmac256(secret_service, "tc3_request")
    return _hmac256(secret_signing, string_to_sign).hex()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
