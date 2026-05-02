#!/usr/bin/env python3
"""
LLM Provider - 支持 OpenAI 兼容接口的 LLM 调用

用途：
- 为 llm_prompt 策略提供意图分类能力
- 支持 OpenAI 兼容接口，如 OpenAI、DeepSeek 和企业网关
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    import openai
except Exception:
    openai = None


@dataclass
class LLMResponse:
    """LLM 调用响应"""
    content: str
    intent_id: Optional[str] = None
    intent_name: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class LLMProvider(ABC):
    """LLM 提供者抽象类"""

    @abstractmethod
    def classify_intent(
        self,
        query: str,
        intents: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> LLMResponse:
        """使用 LLM 进行意图分类

        Args:
            query: 用户查询
            intents: 意图定义列表

        Returns:
            LLMResponse，包含分类结果
        """
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容接口的 LLM 提供者

    支持：
    - OpenAI API
    - 其他 OpenAI 兼容服务

    生产环境建议优先使用统一的 OpenAI 兼容接口，避免为不同模型维护多套调用逻辑。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 256,
        temperature: float = 0.1,
        timeout: float = 5.0,
    ) -> None:
        """
        Args:
            api_key: API 密钥
            base_url: API 基础地址
            model: 模型名称
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            timeout: 请求超时时间（秒）
        """
        if openai is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.client = openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def classify_intent(
        self,
        query: str,
        intents: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> LLMResponse:
        """使用 LLM 进行意图分类"""
        # 构建意图列表
        intent_list = "\n".join([
            f"- {it['id']}: {it['name']}"
            + (f" (示例: {', '.join(it.get('examples', [])[:2])})" if it.get('examples') else "")
            for it in intents
        ])

        default_prompt = """你是一个意图分类器。请根据用户查询，从以下意图中选择最匹配的一个。

可用意图：
{intent_list}

要求：
1. 只输出一个意图 ID
2. 同时输出置信度（0-1之间）
3. 简短解释为什么选择这个意图

输出格式（JSON）：
{{"intent_id": "意图ID", "confidence": 0.95, "reasoning": "简短解释"}}
"""
        system_prompt = (prompt_template or default_prompt).format(intent_list=intent_list, query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            content = response.choices[0].message.content.strip()

            # 尝试解析 JSON
            import json
            try:
                # 提取 JSON（可能在反引号内）
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                result = json.loads(content.strip())
                return LLMResponse(
                    content=content,
                    intent_id=result.get("intent_id"),
                    confidence=result.get("confidence"),
                    reasoning=result.get("reasoning"),
                )
            except json.JSONDecodeError:
                # 如果解析失败，尝试简单提取
                return LLMResponse(content=content)

        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}")


class MockLLMProvider(LLMProvider):
    """测试用的 Mock LLM 提供者"""

    def __init__(self, mock_response: Optional[Dict[str, Any]] = None) -> None:
        self.mock_response = mock_response or {
            "intent_id": "unknown",
            "confidence": 0.5,
            "reasoning": "mock response"
        }

    def classify_intent(
        self,
        query: str,
        intents: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=str(self.mock_response),
            **self.mock_response
        )
