#!/usr/bin/env python3
"""Strategy wrappers for routing templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .cache import intent_defs_from_dicts
from .llm_provider import LLMProvider
from .router import EmbeddingProvider, IntentDef, IntentRouter, RouterResult
from .templates import router_config_from_params


@dataclass
class StrategyBuildContext:
    provider: EmbeddingProvider
    llm_provider: Optional[LLMProvider] = None


class RouteStrategy:
    strategy_type = "base"

    def __init__(self, intents: List[IntentDef], params: Dict[str, Any], context: StrategyBuildContext) -> None:
        self.intents = intents
        self.params = params
        self.context = context

    def route(self, query: str, visible_intent_ids: Optional[Sequence[str]] = None) -> RouterResult:
        raise NotImplementedError


class HybridRetrievalStrategy(RouteStrategy):
    strategy_type = "hybrid_retrieval"

    def __init__(self, intents: List[IntentDef], params: Dict[str, Any], context: StrategyBuildContext) -> None:
        super().__init__(intents, params, context)
        config = router_config_from_params({**params, "enable_bm25": params.get("enable_bm25", True)})
        self.router = IntentRouter(intents=intents, provider=context.provider, config=config)

    def route(self, query: str, visible_intent_ids: Optional[Sequence[str]] = None) -> RouterResult:
        return self.router.route(query=query, visible_intent_ids=visible_intent_ids)


class LLMPromptStrategy(RouteStrategy):
    strategy_type = "llm_prompt"

    def route(self, query: str, visible_intent_ids: Optional[Sequence[str]] = None) -> RouterResult:
        llm_provider = self.context.llm_provider
        if llm_provider is None:
            return RouterResult(
                selected_intent_id="unknown",
                selected_intent_name="其他/不确定",
                method="llm_prompt_unavailable",
                confidence=0.0,
                top_k=[],
                is_ood=True,
                is_ambiguous=False,
            )

        candidate_intents = self.intents
        if visible_intent_ids is not None:
            visible_set = set(visible_intent_ids)
            candidate_intents = [it for it in self.intents if it.id in visible_set]
        if not candidate_intents:
            candidate_intents = self.intents

        try:
            min_confidence = float(self.params.get("llm_min_confidence", 0.0))
            prompt_template = self.params.get("prompt_template")
            llm_resp = llm_provider.classify_intent(
                query,
                [it.__dict__ for it in candidate_intents],
                prompt_template=prompt_template,
            )
            llm_confidence = float(llm_resp.confidence) if llm_resp.confidence is not None else 0.0
            intent_by_id = {it.id: it for it in candidate_intents}
            selected = intent_by_id.get(llm_resp.intent_id or "")
            if selected is None or llm_confidence < min_confidence:
                return RouterResult(
                    selected_intent_id="unknown",
                    selected_intent_name="其他/不确定",
                    method="llm_prompt",
                    confidence=llm_confidence,
                    top_k=[],
                    is_ood=True,
                    is_ambiguous=False,
                )
            return RouterResult(
                selected_intent_id=selected.id,
                selected_intent_name=selected.name,
                method="llm_prompt",
                confidence=llm_confidence,
                top_k=[(selected.id, selected.name, llm_confidence)],
                is_ood=False,
                is_ambiguous=False,
            )
        except Exception:
            return RouterResult(
                selected_intent_id="unknown",
                selected_intent_name="其他/不确定",
                method="llm_prompt_error",
                confidence=0.0,
                top_k=[],
                is_ood=True,
                is_ambiguous=False,
            )


STRATEGY_TYPES = {
    HybridRetrievalStrategy.strategy_type: HybridRetrievalStrategy,
    LLMPromptStrategy.strategy_type: LLMPromptStrategy,
}


def build_strategy(
    strategy_type: str,
    intents_data: List[Dict[str, Any]],
    params: Dict[str, Any],
    context: StrategyBuildContext,
) -> RouteStrategy:
    if strategy_type not in STRATEGY_TYPES:
        raise ValueError(f"Unsupported strategy_type: {strategy_type}")
    intents = intent_defs_from_dicts(intents_data)
    return STRATEGY_TYPES[strategy_type](intents=intents, params=params, context=context)
