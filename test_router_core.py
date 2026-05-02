#!/usr/bin/env python3
"""Lightweight router regression tests that do not load external models."""

from __future__ import annotations

import unittest
import tempfile

import numpy as np

from intent_router.domain import IntentTree, StrategyConfig
from intent_router.runtime import RuntimeRegistry
from intent_router.storage import FileRepository
from intent_router.cache import intent_defs_from_dicts, validate_storage_id
from intent_router.llm_provider import LLMResponse
from intent_router.router import EmbeddingProvider, IntentDef, IntentRouter, RouterConfig


class FakeProvider(EmbeddingProvider):
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "alpha" in text:
                vec = [1.0, 0.0, 0.0]
            elif "beta" in text:
                vec = [0.0, 1.0, 0.0]
            else:
                vec = [0.0, 0.0, 1.0]
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


class FakeLLMProvider:
    def __init__(self, intent_id="a", confidence=0.8):
        self.intent_id = intent_id
        self.confidence = confidence

    def classify_intent(self, query, intents, prompt_template=None):
        return LLMResponse(
            content="{}",
            intent_id=self.intent_id,
            confidence=self.confidence,
        )


def make_router(enable_bm25=True, abs_threshold=0.35):
    intents = [
        IntentDef(id="a", name="alpha intent", examples=["alpha sample"]),
        IntentDef(id="b", name="beta intent", examples=["beta sample"]),
    ]
    return IntentRouter(
        intents=intents,
        provider=FakeProvider(),
        config=RouterConfig(
            abs_threshold=abs_threshold,
            gap_threshold=0.0,
            enable_bm25=enable_bm25,
            ambiguous_action="pick_top1",
        ),
    )


class RouterCoreTest(unittest.TestCase):
    def test_bm25_disabled_uses_raw_semantic_score(self):
        router = make_router(enable_bm25=False)
        result = router.route("alpha")
        self.assertEqual(result.selected_intent_id, "a")
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_zero_bm25_match_does_not_boost_ood(self):
        router = make_router(enable_bm25=True, abs_threshold=0.1)
        result = router.route("unmatched")
        self.assertTrue(result.is_ood)
        self.assertEqual(result.selected_intent_id, "unknown")
        self.assertAlmostEqual(result.confidence, 0.0)

    def test_zero_threshold_override_is_respected(self):
        router = make_router(enable_bm25=False, abs_threshold=0.9)
        result = router.route("unmatched", abs_threshold=0.0)
        self.assertFalse(result.is_ood)

    def test_high_similarity_example_question_direct_match(self):
        router = make_router(enable_bm25=True, abs_threshold=0.9)
        result = router.route("alpha")
        self.assertEqual(result.selected_intent_id, "a")
        self.assertEqual(result.method, "direct_match")
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_empty_intents_are_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one intent"):
            IntentRouter(intents=[], provider=FakeProvider(), config=RouterConfig())
        with self.assertRaisesRegex(ValueError, "at least one intent"):
            intent_defs_from_dicts([])

    def test_storage_ids_are_strict(self):
        self.assertEqual(validate_storage_id("tenant.v1-foo_bar"), "tenant.v1-foo_bar")
        with self.assertRaises(ValueError):
            validate_storage_id("../tenant")

    def test_runtime_registry_routes_active_deployment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = FileRepository(tmpdir)
            tree = IntentTree(
                intent_tree_id="tree_a",
                name="Tree A",
                intents=[
                    {"id": "a", "name": "alpha intent", "examples": ["alpha sample"]},
                    {"id": "b", "name": "beta intent", "examples": ["beta sample"]},
                ],
            )
            repo.save_intent_tree(tree)
            version = repo.create_intent_tree_version("tree_a")
            strategy_config = StrategyConfig(
                strategy_config_id="cfg_a",
                name="Config A",
                strategy_template_id="hybrid_retrieval_default",
                params={"abs_threshold": 0.1, "gap_threshold": 0.0, "enable_bm25": False},
            )
            repo.save_strategy_config(strategy_config)
            deployment = repo.publish(
                tenant_id="tenant_a",
                app_id="app_a",
                environment="prod",
                intent_tree_version_id=version.intent_tree_version_id,
                strategy_config_id=strategy_config.strategy_config_id,
            )
            registry = RuntimeRegistry(repository=repo, provider=FakeProvider())
            bundle = registry.reload_deployment(deployment)
            self.assertEqual(bundle.deployment_key, "tenant_a:app_a:prod")
            result, used_bundle = registry.route("tenant_a", "app_a", "prod", "alpha")
            self.assertEqual(result.selected_intent_id, "a")
            self.assertEqual(used_bundle.strategy_template.strategy_type, "hybrid_retrieval")

    def test_runtime_registry_routes_llm_prompt_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = FileRepository(tmpdir)
            tree = IntentTree(
                intent_tree_id="tree_llm",
                name="Tree LLM",
                intents=[
                    {"id": "a", "name": "alpha intent", "examples": ["alpha sample"]},
                    {"id": "b", "name": "beta intent", "examples": ["beta sample"]},
                ],
            )
            repo.save_intent_tree(tree)
            version = repo.create_intent_tree_version("tree_llm")
            strategy_config = StrategyConfig(
                strategy_config_id="cfg_llm",
                name="LLM Config",
                strategy_template_id="llm_prompt_default",
                params={"llm_min_confidence": 0.6},
            )
            repo.save_strategy_config(strategy_config)
            deployment = repo.publish(
                tenant_id="tenant_a",
                app_id="app_llm",
                environment="prod",
                intent_tree_version_id=version.intent_tree_version_id,
                strategy_config_id=strategy_config.strategy_config_id,
            )
            registry = RuntimeRegistry(
                repository=repo,
                provider=FakeProvider(),
                llm_provider=FakeLLMProvider(intent_id="b", confidence=0.8),
            )
            registry.reload_deployment(deployment)
            result, used_bundle = registry.route("tenant_a", "app_llm", "prod", "anything")
            self.assertEqual(result.selected_intent_id, "b")
            self.assertEqual(result.method, "llm_prompt")
            self.assertEqual(used_bundle.strategy_template.strategy_type, "llm_prompt")


if __name__ == "__main__":
    unittest.main()
