#!/usr/bin/env python3
"""
Embedding Router 核心逻辑

功能：
- 实现基于embedding的意图识别核心算法
- 支持TopK排序、阈值判断、歧义检测、OOD（未知意图）识别
- 可通过配置调整识别策略
- 支持预计算和加载向量索引，提高性能
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None  # type: ignore


@dataclass
class IntentDef:
    """意图/路由定义，包含意图ID、名称、范围和示例"""

    id: str
    name: str
    scope: str = ""
    out_of_scope: str = ""
    examples: List[str] | None = None
    negative_examples: List[str] | None = None

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []
        if self.negative_examples is None:
            self.negative_examples = []


@dataclass
class RouterConfig:
    """路由配置，包含模型、阈值、Top-K 等参数"""

    top_k: int = 3
    abs_threshold: float = 0.35
    gap_threshold: float = 0.05
    unknown_intent_id: str = "unknown"
    unknown_intent_name: str = "其他/不确定"
    ambiguous_action: str = "clarify"
    weight_intro: float = 0.5
    question_top_k: int = 5
    example_question_threshold: float = 0.95
    short_query_len: int = 10
    alpha_short: float = 0.6
    alpha_long: float = 0.8
    enable_bm25: bool = True
    stopwords_path: Optional[str] = "config/stopwords.txt"


@dataclass
class RouterResult:
    """意图识别结果，包含识别出的意图、置信度等信息"""

    selected_intent_id: str
    selected_intent_name: str
    method: str
    confidence: float
    top_k: List[Tuple[str, str, float]]
    is_ood: bool
    is_ambiguous: bool


class EmbeddingProvider:
    """Embedding 生成抽象类，所有 Embedding 提供者需实现此接口"""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class BM25OkapiLite:
    """简化版 BM25Okapi，实现无额外依赖的全文检索打分。"""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lens = np.asarray([len(toks) for toks in corpus_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) > 0 else 0.0

        self.tfs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        n_docs = len(corpus_tokens)
        if n_docs == 0:
            return

        df: Dict[str, int] = {}
        for toks in corpus_tokens:
            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self.tfs.append(tf)
            for tok in tf.keys():
                df[tok] = df.get(tok, 0) + 1

        for tok, dfi in df.items():
            self.idf[tok] = math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        n_docs = len(self.tfs)
        if n_docs == 0:
            return np.zeros((0,), dtype=np.float32)
        if not query_tokens:
            return np.zeros((n_docs,), dtype=np.float32)

        q_unique = set(query_tokens)
        scores = np.zeros((n_docs,), dtype=np.float32)
        for i, tf in enumerate(self.tfs):
            dl = float(self.doc_lens[i])
            norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-12)))
            s = 0.0
            for tok in q_unique:
                f = tf.get(tok, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(tok, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + norm)
            scores[i] = float(s)
        return scores


class IntentRouter:
    """基于 Embedding 的意图路由器，负责意图识别核心逻辑"""

    def __init__(
        self,
        intents: List[IntentDef],
        provider: EmbeddingProvider,
        config: RouterConfig,
        build_on_init: bool = True,
    ) -> None:
        """初始化路由器，加载意图配置和 Embedding 提供者"""
        self.intents = intents
        self.provider = provider
        self.config = config

        self._intent_ids: List[str] = []
        self._intent_names: List[str] = []
        self._question_offsets: Optional[np.ndarray] = None
        self._intro_indices: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._bm25: Optional[BM25OkapiLite] = None
        self._stopwords: Set[str] = self._load_stopwords(config.stopwords_path)
        self._doc_tokens: List[List[str]] = []

        if build_on_init:
            self.build_index()

    @staticmethod
    def _build_intro_text(intent: IntentDef) -> str:
        meta_parts = [intent.name]
        if intent.scope:
            meta_parts.append(f"范围：{intent.scope}")
        if intent.out_of_scope:
            meta_parts.append(f"不包括：{intent.out_of_scope}")
        return "。".join(meta_parts)

    @staticmethod
    def load_intents_from_json(path: str) -> List[IntentDef]:
        """从 JSON 文件加载意图定义"""
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        intents: List[IntentDef] = []
        for item in cfg.get("intents", []):
            intents.append(
                IntentDef(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    scope=item.get("scope", ""),
                    out_of_scope=item.get("out_of_scope", ""),
                    examples=item.get("examples", []) or [],
                    negative_examples=item.get("negative_examples", []) or [],
                )
            )
        return intents

    def build_index(self) -> None:
        """计算并缓存语义向量与 BM25 索引。"""
        if not self.intents:
            raise ValueError("IntentRouter requires at least one intent")

        intent_ids: List[str] = []
        intent_names: List[str] = []
        question_offsets: List[int] = [0]
        intro_indices: List[int] = []
        all_vecs: List[np.ndarray] = []
        all_tokens: List[List[str]] = []

        for it in self.intents:
            questions: List[str] = list(it.examples)
            if not questions:
                questions = [it.name]

            intro_text = self._build_intro_text(it)

            texts: List[str] = questions + [intro_text]

            vecs = self.provider.encode(texts)
            intent_ids.append(it.id)
            intent_names.append(it.name)
            all_vecs.append(vecs)
            question_offsets.append(question_offsets[-1] + len(questions))
            intro_indices.append(question_offsets[-1] + len(intent_ids) - 1)

            for text in texts:
                all_tokens.append(self._tokenize(text))

        self._intent_ids = intent_ids
        self._intent_names = intent_names
        self._question_offsets = np.asarray(question_offsets, dtype=np.int32)
        self._intro_indices = np.asarray(intro_indices, dtype=np.int32)
        self._vectors = np.vstack(all_vecs).astype(np.float32)
        self._doc_tokens = all_tokens
        self._bm25 = BM25OkapiLite(all_tokens) if self.config.enable_bm25 else None

    def save_index(self, path: str) -> None:
        """将计算出的索引保存到 .npz 文件"""
        if self._question_offsets is None or self._intro_indices is None or self._vectors is None:
            raise RuntimeError("索引未构建")
        np.savez_compressed(
            path,
            intent_ids=np.asarray(self._intent_ids),
            intent_names=np.asarray(self._intent_names),
            question_offsets=self._question_offsets,
            intro_indices=self._intro_indices,
            vectors=self._vectors,
        )

    @classmethod
    def load_with_index(
        cls,
        intents: List[IntentDef],
        provider: EmbeddingProvider,
        config: RouterConfig,
        index_path: str,
    ) -> "IntentRouter":
        """从 .npz 文件加载预计算的索引，快速初始化路由器"""
        router = cls(intents=intents, provider=provider, config=config, build_on_init=False)
        data = np.load(index_path, allow_pickle=False)
        router._intent_ids = list(data["intent_ids"])
        router._intent_names = list(data["intent_names"])
        router._question_offsets = data["question_offsets"].astype(np.int32)
        router._intro_indices = data["intro_indices"].astype(np.int32)
        router._vectors = data["vectors"].astype(np.float32)
        rebuilt_tokens: List[List[str]] = []
        for it in router.intents:
            questions = list(it.examples or [])
            if not questions:
                questions = [it.name]
            intro_text = router._build_intro_text(it)
            for text in questions + [intro_text]:
                rebuilt_tokens.append(router._tokenize(text))
        router._doc_tokens = rebuilt_tokens
        router._bm25 = BM25OkapiLite(rebuilt_tokens) if router.config.enable_bm25 else None
        return router

    @staticmethod
    def _load_stopwords(path: Optional[str]) -> Set[str]:
        defaults = {"的", "了", "和", "是", "在", "吗", "呢", "啊", "哦", "请问", "一下"}
        if not path:
            return defaults
        if not os.path.exists(path):
            return defaults
        out = set(defaults)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    out.add(word)
        return out

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if jieba is not None:
            toks = [t.strip() for t in jieba.lcut(text) if t.strip()]
        else:
            toks = [ch for ch in text if not ch.isspace()]
        return [t for t in toks if t not in self._stopwords]

    @staticmethod
    def _topk_mean(values: np.ndarray, k: int) -> float:
        if values.size == 0:
            return 0.0
        kk = max(1, min(int(k), int(values.size)))
        idx = np.argpartition(values, -kk)[-kk:]
        return float(np.mean(values[idx]))

    def _candidate_indices(self, visible_intent_ids: Optional[Sequence[str]]) -> List[int]:
        if not visible_intent_ids:
            return list(range(len(self._intent_ids)))
        vis = set(visible_intent_ids)
        return [i for i, iid in enumerate(self._intent_ids) if iid in vis]

    def route(
        self,
        query: str,
        visible_intent_ids: Optional[Sequence[str]] = None,
        custom_intents: Optional[List[IntentDef]] = None,
        abs_threshold: Optional[float] = None,
        gap_threshold: Optional[float] = None,
    ) -> RouterResult:
        """意图识别核心方法，将用户查询路由到最匹配的意图"""
        # 如果提供了自定义意图，使用自定义意图创建临时路由器
        if custom_intents:
            # 创建临时配置，使用自定义阈值
            temp_config = RouterConfig(
                abs_threshold=self.config.abs_threshold if abs_threshold is None else float(abs_threshold),
                gap_threshold=self.config.gap_threshold if gap_threshold is None else float(gap_threshold),
                top_k=self.config.top_k,
                question_top_k=self.config.question_top_k,
                example_question_threshold=self.config.example_question_threshold,
                weight_intro=self.config.weight_intro,
                enable_bm25=self.config.enable_bm25,
                short_query_len=self.config.short_query_len,
                alpha_short=self.config.alpha_short,
                alpha_long=self.config.alpha_long,
                ambiguous_action=self.config.ambiguous_action,
                unknown_intent_id=self.config.unknown_intent_id,
                unknown_intent_name=self.config.unknown_intent_name,
                stopwords_path=self.config.stopwords_path,
            )
            temp_router = IntentRouter(
                intents=custom_intents,
                provider=self.provider,
                config=temp_config,
                build_on_init=True
            )
            return temp_router.route(query, visible_intent_ids)
        
        # 使用自定义阈值（如果提供）
        current_abs_threshold = self.config.abs_threshold if abs_threshold is None else float(abs_threshold)
        current_gap_threshold = self.config.gap_threshold if gap_threshold is None else float(gap_threshold)

        if self._question_offsets is None or self._intro_indices is None or self._vectors is None:
            raise RuntimeError("索引未构建")

        cand_idx = self._candidate_indices(visible_intent_ids)
        if not cand_idx:
            return RouterResult(
                selected_intent_id=self.config.unknown_intent_id,
                selected_intent_name=self.config.unknown_intent_name,
                method="ood",
                confidence=0.0,
                top_k=[],
                is_ood=True,
                is_ambiguous=False,
            )

        q_vec = self.provider.encode([query])
        doc_sims = cosine_similarity(q_vec, self._vectors)[0]

        direct_scored: List[Tuple[int, float]] = []
        if self.config.example_question_threshold > 0:
            for i in cand_idx:
                q_start = int(self._question_offsets[i])
                q_end = int(self._question_offsets[i + 1])
                q_sims = doc_sims[q_start:q_end]
                if q_sims.size == 0:
                    continue
                best_question_sim = float(np.max(q_sims))
                if best_question_sim >= self.config.example_question_threshold:
                    direct_scored.append((i, best_question_sim))

        if direct_scored:
            direct_scored.sort(key=lambda x: x[1], reverse=True)
            topk = [
                (self._intent_ids[i], self._intent_names[i], float(s))
                for i, s in direct_scored[: max(self.config.top_k, 1)]
            ]
            best_i, best_s = direct_scored[0]
            second_s = direct_scored[1][1] if len(direct_scored) > 1 else -1.0
            is_ambiguous = (
                len(direct_scored) > 1
                and (best_s - second_s) < current_gap_threshold
            )
            if is_ambiguous and self.config.ambiguous_action == "clarify":
                return RouterResult(
                    selected_intent_id=self.config.unknown_intent_id,
                    selected_intent_name=self.config.unknown_intent_name,
                    method="direct_match_ambiguous",
                    confidence=float(best_s),
                    top_k=topk,
                    is_ood=False,
                    is_ambiguous=True,
                )
            return RouterResult(
                selected_intent_id=self._intent_ids[best_i],
                selected_intent_name=self._intent_names[best_i],
                method="direct_match",
                confidence=float(best_s),
                top_k=topk,
                is_ood=False,
                is_ambiguous=is_ambiguous,
            )

        bm25_scores = np.zeros_like(doc_sims)
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(self._tokenize(query))

        semantic_scored: List[Tuple[int, float]] = []
        bm25_scored: List[Tuple[int, float]] = []
        for i in cand_idx:
            q_start = int(self._question_offsets[i])
            q_end = int(self._question_offsets[i + 1])
            intro_idx = int(self._intro_indices[i])

            q_sims = doc_sims[q_start:q_end]
            intro_sim = float(doc_sims[intro_idx])
            q_sem = self._topk_mean(q_sims, self.config.question_top_k)
            sem = float(self.config.weight_intro * intro_sim + (1.0 - self.config.weight_intro) * q_sem)
            semantic_scored.append((i, sem))

            q_bm25_arr = bm25_scores[q_start:q_end]
            intro_bm25 = float(bm25_scores[intro_idx])
            q_bm25 = self._topk_mean(q_bm25_arr, self.config.question_top_k)
            bm25 = float(self.config.weight_intro * intro_bm25 + (1.0 - self.config.weight_intro) * q_bm25)
            bm25_scored.append((i, bm25))

        # BM25 归一化：仅在存在有效关键词命中且候选间有区分度时引入词法分数。
        # 全 0 或全相等时保持为 0，避免给所有意图统一加分、影响 OOD 校准。
        bm25_norm: Dict[int, float] = {}
        if self.config.enable_bm25 and bm25_scored:
            bm25_scores_list = [s for _, s in bm25_scored]
            bm25_max = max(bm25_scores_list)
            bm25_min = min(bm25_scores_list)
            bm25_range = bm25_max - bm25_min
            if bm25_max > 0.0 and bm25_range > 1e-12:
                for i, s in bm25_scored:
                    bm25_norm[i] = float((s - bm25_min) / bm25_range)

        alpha = self.config.alpha_short if len(query) < self.config.short_query_len else self.config.alpha_long
        alpha = max(0.0, min(1.0, float(alpha)))

        scored: List[Tuple[int, float]] = []
        if not self.config.enable_bm25:
            scored = [(i, float(sem)) for i, sem in semantic_scored]
        else:
            for i, sem in semantic_scored:
                final_score = float(alpha * sem + (1.0 - alpha) * bm25_norm.get(i, 0.0))
                scored.append((i, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        topk: List[Tuple[str, str, float]] = []
        for i, s in scored[: max(self.config.top_k, 1)]:
            topk.append((self._intent_ids[i], self._intent_names[i], float(s)))

        best_i, best_s = scored[0]
        second_s = scored[1][1] if len(scored) > 1 else -1.0

        is_ood = best_s < current_abs_threshold
        is_ambiguous = (len(scored) > 1) and ((best_s - second_s) < current_gap_threshold)

        if is_ood:
            return RouterResult(
                selected_intent_id=self.config.unknown_intent_id,
                selected_intent_name=self.config.unknown_intent_name,
                method="ood",
                confidence=float(best_s),
                top_k=topk,
                is_ood=True,
                is_ambiguous=False,
            )

        if is_ambiguous and self.config.ambiguous_action == "clarify":
            return RouterResult(
                selected_intent_id=self.config.unknown_intent_id,
                selected_intent_name=self.config.unknown_intent_name,
                method="ambiguous",
                confidence=float(best_s),
                top_k=topk,
                is_ood=False,
                is_ambiguous=True,
            )

        return RouterResult(
            selected_intent_id=self._intent_ids[best_i],
            selected_intent_name=self._intent_names[best_i],
            method="hybrid",
            confidence=float(best_s),
            top_k=topk,
            is_ood=False,
            is_ambiguous=is_ambiguous,
        )


def load_router(
    intents_json_path: str,
    provider: EmbeddingProvider,
    abs_threshold: Optional[float] = None,
    gap_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> IntentRouter:
    """加载意图配置并初始化 IntentRouter 实例"""
    intents = IntentRouter.load_intents_from_json(intents_json_path)
    cfg = RouterConfig()
    if abs_threshold is not None:
        cfg.abs_threshold = float(abs_threshold)
    if gap_threshold is not None:
        cfg.gap_threshold = float(gap_threshold)
    if top_k is not None:
        cfg.top_k = int(top_k)

    return IntentRouter(intents=intents, provider=provider, config=cfg)
