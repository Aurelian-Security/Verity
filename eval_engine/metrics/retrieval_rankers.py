from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Metric(BaseMetric):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        try:
            self.validate_inputs(question, contexts, answer)

            if not contexts:
                return MetricResult(self.name, 0.0, raw={"ranked_contexts": []})

            query_terms = _tokenize(question)
            tokenized_docs = [_tokenize(doc) for doc in contexts]
            avgdl = sum(len(doc) for doc in tokenized_docs) / max(len(tokenized_docs), 1)

            doc_freq: Counter[str] = Counter()
            for doc in tokenized_docs:
                for term in set(doc):
                    doc_freq[term] += 1

            n_docs = len(contexts)
            scores: list[tuple[int, float]] = []

            for idx, doc in enumerate(tokenized_docs):
                tf = Counter(doc)
                doc_len = len(doc)
                score = 0.0

                for term in query_terms:
                    if term not in tf:
                        continue

                    idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                    numerator = tf[term] * (self.k1 + 1)
                    denominator = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1))
                    score += idf * numerator / denominator

                scores.append((idx, score))

            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            top_score = ranked[0][1] if ranked else 0.0

            return MetricResult(
                metric_name=self.name,
                score=float(top_score),
                raw={
                    "ranked_contexts": [
                        {"index": idx, "score": score, "context": contexts[idx]}
                        for idx, score in ranked
                    ]
                },
            )
        except Exception as exc:
            return self.error_result(str(exc))


class ReciprocalRankFusionMetric(BaseMetric):
    name = "rrf"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        try:
            rankings: list[list[int]] = kwargs.get("rankings", [])
            k: int = kwargs.get("k", 60)

            if not rankings:
                return MetricResult(self.name, 0.0, raw={"fused_ranking": []})

            fused_scores: defaultdict[int, float] = defaultdict(float)

            for ranking in rankings:
                for rank, doc_idx in enumerate(ranking, start=1):
                    fused_scores[doc_idx] += 1.0 / (k + rank)

            fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            top_score = fused[0][1] if fused else 0.0

            return MetricResult(
                metric_name=self.name,
                score=float(top_score),
                raw={
                    "fused_ranking": [
                        {"index": idx, "score": score}
                        for idx, score in fused
                    ]
                },
            )
        except Exception as exc:
            return self.error_result(str(exc))


class MMRMetric(BaseMetric):
    name = "mmr"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        try:
            self.validate_inputs(question, contexts, answer)

            lambda_mult: float = kwargs.get("lambda_mult", 0.7)
            top_k: int = kwargs.get("top_k", min(5, len(contexts)))

            query_terms = set(_tokenize(question))
            doc_terms = [set(_tokenize(ctx)) for ctx in contexts]

            def jaccard(a: set[str], b: set[str]) -> float:
                if not a and not b:
                    return 0.0
                return len(a & b) / max(len(a | b), 1)

            relevance = [jaccard(query_terms, terms) for terms in doc_terms]

            selected: list[int] = []
            candidates = set(range(len(contexts)))

            while candidates and len(selected) < top_k:
                best_idx = None
                best_score = float("-inf")

                for idx in candidates:
                    diversity_penalty = max(
                        [jaccard(doc_terms[idx], doc_terms[s]) for s in selected],
                        default=0.0,
                    )
                    mmr_score = lambda_mult * relevance[idx] - (1 - lambda_mult) * diversity_penalty

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                selected.append(best_idx)
                candidates.remove(best_idx)

            return MetricResult(
                metric_name=self.name,
                score=float(sum(relevance[i] for i in selected) / max(len(selected), 1)),
                raw={
                    "selected_contexts": [
                        {
                            "index": idx,
                            "relevance": relevance[idx],
                            "context": contexts[idx],
                        }
                        for idx in selected
                    ]
                },
            )
        except Exception as exc:
            return self.error_result(str(exc))


class LearningToRankHeuristicMetric(BaseMetric):
    name = "ltr_heuristic"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        try:
            self.validate_inputs(question, contexts, answer)

            query_terms = set(_tokenize(question))
            answer_terms = set(_tokenize(answer))
            gt_terms = set(_tokenize(ground_truth or ""))

            weights = kwargs.get(
                "weights",
                {
                    "query_overlap": 0.45,
                    "answer_overlap": 0.35,
                    "ground_truth_overlap": 0.20,
                },
            )

            def overlap(a: set[str], text: str) -> float:
                terms = set(_tokenize(text))
                if not a or not terms:
                    return 0.0
                return len(a & terms) / len(a)

            ranked = []

            for idx, ctx in enumerate(contexts):
                score = (
                    weights["query_overlap"] * overlap(query_terms, ctx)
                    + weights["answer_overlap"] * overlap(answer_terms, ctx)
                    + weights["ground_truth_overlap"] * overlap(gt_terms, ctx)
                )
                ranked.append((idx, score))

            ranked.sort(key=lambda x: x[1], reverse=True)
            top_score = ranked[0][1] if ranked else 0.0

            return MetricResult(
                metric_name=self.name,
                score=float(top_score),
                raw={
                    "ranked_contexts": [
                        {"index": idx, "score": score, "context": contexts[idx]}
                        for idx, score in ranked
                    ]
                },
            )
        except Exception as exc:
            return self.error_result(str(exc))


RETRIEVAL_RANKER_REGISTRY = {
    "bm25": BM25Metric,
    "rrf": ReciprocalRankFusionMetric,
    "mmr": MMRMetric,
    "ltr_heuristic": LearningToRankHeuristicMetric,
}