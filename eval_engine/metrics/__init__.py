"""
eval_engine/metrics/__init__.py — Complete MetricsRegistry

Metrics by implementation status:

IMPLEMENTED (Phase 1-3 + Tier 1):
    recall_at_k, mean_reciprocal_rank, ndcg
    compression_delta, deduplication_delta, entity_coverage
    ragas_grounding, ragas_consolidation_delta
    llamaguard_safety
    per_stage_ablation, threshold_compute_budget
    single_session_poisoning, query_perturbation
    calibration, hallucination_rate, trust_score, multi_session_persistence
    bm25, rrf, mmr, ltr_heuristic

SCAFFOLD — Tier 2 (interface locked, implementation deferred):
    consistency, constitutional_eval, model_written_eval, source_reliability

SCAFFOLD — Tier 3 (interface locked, frontier research):
    goal_misgeneralization, deceptive_alignment
"""
from __future__ import annotations
import importlib, logging
from typing import TYPE_CHECKING

from eval_engine.metrics.base import BaseMetric, MetricResult
# Implemented
from eval_engine.metrics.retrieval_metrics import RecallAtKMetric, MRRMetric, NDCGMetric
from eval_engine.metrics.graph_metrics import CompressionDeltaMetric, DeduplicationDeltaMetric, EntityCoverageMetric
from eval_engine.metrics.ragas_consolidation_delta import RAGASConsolidationDelta, ConsolidationDeltaResult
from eval_engine.metrics.ragas_grounding import RAGASGroundingMetric
from eval_engine.metrics.llamaguard import LlamaGuardMetric
from eval_engine.metrics.per_stage_ablation import PerStageAblationMetric
from eval_engine.metrics.threshold_compute_budget import ThresholdComputeBudgetMetric
from eval_engine.metrics.single_session_poisoning import SingleSessionPoisoningMetric
from eval_engine.metrics.query_perturbation import QueryPerturbationMetric
from eval_engine.metrics.calibration import CalibrationMetric
from eval_engine.metrics.hallucination import HallucinationMetric
from eval_engine.metrics.trust_score import TrustScoreMetric
from eval_engine.metrics.multi_session_persistence import MultiSessionPersistenceMetric
from eval_engine.metrics.retrieval_rankers import  BM25Metric,  ReciprocalRankFusionMetric, MMRMetric, LearningToRankHeuristicMetric
# Tier 2 scaffolds
from eval_engine.metrics.consistency import ConsistencyMetric
from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
from eval_engine.metrics.source_reliability import SourceReliabilityMetric
# Tier 3 scaffolds
from eval_engine.metrics.goal_misgeneralization import GoalMisgeneralizationMetric
from eval_engine.metrics.deceptive_alignment import DeceptiveAlignmentMetric

if TYPE_CHECKING:
    from eval_engine.config import MetricConfig

logger = logging.getLogger(__name__)

_BUILTIN_REGISTRY: dict[str, type[BaseMetric]] = {
    # --- IMPLEMENTED ---
    "recall_at_k":                  RecallAtKMetric,
    "mean_reciprocal_rank":         MRRMetric,
    "ndcg":                         NDCGMetric,
    "compression_delta":            CompressionDeltaMetric,
    "deduplication_delta":          DeduplicationDeltaMetric,
    "entity_coverage":              EntityCoverageMetric,
    "ragas_grounding":              RAGASGroundingMetric,
    "ragas_consolidation_delta":    RAGASConsolidationDelta,
    "llamaguard_safety":            LlamaGuardMetric,
    "per_stage_ablation":           PerStageAblationMetric,
    "threshold_compute_budget":     ThresholdComputeBudgetMetric,
    "single_session_poisoning":     SingleSessionPoisoningMetric,
    "query_perturbation":           QueryPerturbationMetric,
    "calibration":                  CalibrationMetric,
    "hallucination_rate":           HallucinationMetric,
    "trust_score":                  TrustScoreMetric,
    "multi_session_persistence":    MultiSessionPersistenceMetric,
    "bm25":                         BM25Metric,
    "rrf":                          ReciprocalRankFusionMetric,
    "mmr":                          MMRMetric,
    "ltr_heuristic":                LearningToRankHeuristicMetric,
    # --- TIER 2 SCAFFOLDS ---
    "consistency":                  ConsistencyMetric,
    "constitutional_eval":          ConstitutionalEvalMetric,
    "model_written_eval":           ModelWrittenEvalMetric,
    "source_reliability":           SourceReliabilityMetric,
    # --- TIER 3 SCAFFOLDS ---
    "goal_misgeneralization":       GoalMisgeneralizationMetric,
    "deceptive_alignment":          DeceptiveAlignmentMetric,
}

# Merge 47-algorithm expansion registry (algorithms 1-47; 48-50 excluded intentionally)
from eval_engine.metrics.algorithm_expansion import ALGORITHM_EXPANSION_REGISTRY
_BUILTIN_REGISTRY.update(ALGORITHM_EXPANSION_REGISTRY)

class MetricsRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseMetric]] = dict(_BUILTIN_REGISTRY)

    def register(self, name: str, cls: type[BaseMetric]) -> None:
        if not issubclass(cls, BaseMetric):
            raise TypeError(f"{cls} must subclass BaseMetric")
        self._registry[name] = cls

    def get(self, name: str, **kwargs) -> BaseMetric:
        if name not in self._registry:
            raise KeyError(f"Metric '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name](**kwargs)

    def from_config(self, metric_cfg: "MetricConfig") -> BaseMetric:
        kwargs = dict(metric_cfg.kwargs)
        if metric_cfg.name.value in ("ndcg", "recall_at_k"):
            kwargs.setdefault("k", metric_cfg.k)
        plugin_path = kwargs.pop("plugin_path", None)
        if plugin_path:
            module_path, class_name = plugin_path.rsplit(":", 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            self.register(cls.name, cls)
        return self.get(metric_cfg.name.value, **kwargs)

    def scaffolds(self) -> list[str]:
        """Return names of scaffold metrics not yet implemented."""
        scaffold_names = []
        for name, cls in self._registry.items():
            instance = cls.__new__(cls)
            # Scaffolds return metadata with status='scaffold'
            if hasattr(cls, '__doc__') and cls.__doc__ and 'SCAFFOLD' in cls.__doc__:
                scaffold_names.append(name)
        return sorted(scaffold_names)

    @property
    def available(self) -> list[str]:
        return sorted(self._registry.keys())

registry = MetricsRegistry()

__all__ = [
    "BaseMetric", "MetricResult", "ConsolidationDeltaResult",
    "RecallAtKMetric", "MRRMetric", "NDCGMetric",
    "CompressionDeltaMetric", "DeduplicationDeltaMetric", "EntityCoverageMetric",
    "RAGASGroundingMetric", "RAGASConsolidationDelta", "LlamaGuardMetric",
    "PerStageAblationMetric", "ThresholdComputeBudgetMetric",
    "SingleSessionPoisoningMetric", "QueryPerturbationMetric",
    "CalibrationMetric", "HallucinationMetric",
    "TrustScoreMetric", "MultiSessionPersistenceMetric",
    "ConsistencyMetric", "ConstitutionalEvalMetric",
    "ModelWrittenEvalMetric", "SourceReliabilityMetric",
    "GoalMisgeneralizationMetric", "DeceptiveAlignmentMetric",
    "MetricsRegistry", "registry", "bm25", "rrf", "mmr", "ltr_heuristic",
]
