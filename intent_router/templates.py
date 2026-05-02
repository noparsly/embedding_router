#!/usr/bin/env python3
"""Built-in strategy templates and parameter helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from .domain import StrategyTemplate
from .router import RouterConfig


HYBRID_DEFAULT_PARAMS: Dict[str, Any] = {
    "abs_threshold": 0.65,
    "gap_threshold": 0.03,
    "top_k": 3,
    "ambiguous_action": "clarify",
    "weight_intro": 0.55,
    "question_top_k": 7,
    "alpha_short": 0.45,
    "alpha_long": 0.65,
    "enable_bm25": True,
    "short_query_len": 10,
    "example_question_threshold": 0.95,
}

EMBEDDING_ONLY_DEFAULT_PARAMS: Dict[str, Any] = {
    **HYBRID_DEFAULT_PARAMS,
    "enable_bm25": False,
    "abs_threshold": 0.62,
    "gap_threshold": 0.04,
    "alpha_short": 1.0,
    "alpha_long": 1.0,
}

LLM_PROMPT_DEFAULT_PARAMS: Dict[str, Any] = {
    "llm_min_confidence": 0.65,
    "prompt_template": """你是一个意图分类器。请根据用户查询，从以下意图中选择最匹配的一个。

可用意图：
{intent_list}

用户查询：
{query}

要求：
1. 只能选择可用意图中的 intent_id；无法判断时输出 unknown。
2. 不要回答用户问题，只做意图分类。
3. 输出 JSON，不要输出额外文本。
4. confidence 是 0 到 1 之间的小数。

输出格式：
{{"intent_id":"意图ID或unknown","confidence":0.95,"reasoning":"简短解释"}}
""",
}

PARAM_SCHEMA: Dict[str, Any] = {
    "abs_threshold": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "gap_threshold": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "top_k": {"type": "int", "min": 1, "max": 20, "step": 1},
    "weight_intro": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "alpha_short": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "alpha_long": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "enable_bm25": {"type": "bool"},
    "short_query_len": {"type": "int", "min": 1, "max": 100, "step": 1},
    "question_top_k": {"type": "int", "min": 1, "max": 50, "step": 1},
    "example_question_threshold": {"type": "float", "min": 0, "max": 1, "step": 0.01},
}

LLM_PARAM_SCHEMA: Dict[str, Any] = {
    "llm_min_confidence": {"type": "float", "min": 0, "max": 1, "step": 0.01},
    "prompt_template": {"type": "text"},
}


def built_in_strategy_templates() -> List[StrategyTemplate]:
    return [
        StrategyTemplate(
            strategy_template_id="hybrid_retrieval_default",
            name="默认混合检索",
            strategy_type="hybrid_retrieval",
            description="Embedding + BM25 混合检索，适合大多数意图树，作为默认推荐模板。",
            default_params=HYBRID_DEFAULT_PARAMS.copy(),
            param_schema=PARAM_SCHEMA.copy(),
            required_resources={"embedding": True, "llm": False},
        ),
        StrategyTemplate(
            strategy_template_id="llm_prompt_default",
            name="默认大模型 Prompt",
            strategy_type="llm_prompt",
            description="直接用大模型和提示词识别意图，适合意图定义复杂、样本较少或需要解释的场景。",
            default_params=LLM_PROMPT_DEFAULT_PARAMS.copy(),
            param_schema=LLM_PARAM_SCHEMA.copy(),
            required_resources={"embedding": False, "llm": True},
        ),
    ]


def merge_template_params(template: StrategyTemplate, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    merged = template.default_params.copy()
    merged.update(params or {})
    return merged


def router_config_from_params(params: Dict[str, Any]) -> RouterConfig:
    return RouterConfig(
        top_k=int(params.get("top_k", 3)),
        abs_threshold=float(params.get("abs_threshold", 0.35)),
        gap_threshold=float(params.get("gap_threshold", 0.05)),
        ambiguous_action=str(params.get("ambiguous_action", "clarify")),
        weight_intro=float(params.get("weight_intro", 0.5)),
        question_top_k=int(params.get("question_top_k", 5)),
        alpha_short=float(params.get("alpha_short", 0.6)),
        alpha_long=float(params.get("alpha_long", 0.8)),
        enable_bm25=bool(params.get("enable_bm25", True)),
        short_query_len=int(params.get("short_query_len", 10)),
        example_question_threshold=float(params.get("example_question_threshold", 0.95)),
    )
