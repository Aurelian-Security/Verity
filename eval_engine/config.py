"""
eval_engine/config.py

Pydantic v2 configuration schema for the eval-engine SDK.
All experiment configs (YAML or programmatic) are validated against
these models before any execution begins.

Replaces raw YAML experiment configs.
Five-test consolidation evaluation suite maps to EvalConfig.tests.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SupportedModel(str, Enum):
    CLAUDE_SONNET = "claude-sonnet-4-6"
    CLAUDE_HAIKU = "claude-haiku-4-5"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    OLLAMA_MISTRAL = "ollama/mistral"
    OLLAMA_LLAMA3 = "ollama/llama3"


class RAGArchitecture(str, Enum):
    CENTRALIZED = "centralized"       # Single-node centralized RAG baseline
    DECENTRALIZED = "decentralized"   # Multi-node decentralized RAG deployment
    CONSOLIDATION = "consolidation"             # Memory-consolidation RAG architecture


class TestName(str, Enum):
    # Consolidation evaluation suite — five core tests
    RAGAS_CONSOLIDATION_DELTA = "ragas_consolidation_delta"
    PER_STAGE_ABLATION = "per_stage_ablation"
    THRESHOLD_COMPUTE_BUDGET = "threshold_compute_budget"
    SINGLE_SESSION_POISONING = "single_session_poisoning"
    QUERY_PERTURBATION = "query_perturbation"
    # Standard retrieval harness tests
    RAGAS_GROUNDING = "ragas_grounding"
    LLAMAGUARD_SAFETY = "llamaguard_safety"
    NDCG = "ndcg"
    RECALL_AT_K = "recall_at_k"
    # Reserved — longitudinal evaluation scope
    SESSION_LEVEL = "session_level"        # Deferred: longitudinal scope
    LONGITUDINAL = "longitudinal"          # Deferred: longitudinal scope


class StatisticalTest(str, Enum):
    WILCOXON = "wilcoxon"          # Pre-registered significance test
    COHENS_D = "cohens_d"          # Effect size — pre-registered
    PEARSON = "pearson"            # RAM-to-FNR correlation
    MANN_WHITNEY = "mann_whitney"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    """Hard ceiling on LLM judge spend per run. Halts before exceeding."""
    max_usd: float = Field(default=5.00, gt=0, description="Max spend in USD before halt")
    warn_at_pct: float = Field(default=0.80, ge=0.1, le=1.0, description="Warn when this % of budget consumed")
    track_tokens: bool = Field(default=True, description="Log per-call token counts to JSONL ledger")


class RetryConfig(BaseModel):
    """Adaptive rate-limiting and failure retry settings."""
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_base_seconds: float = Field(default=2.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)
    retry_on_status: list[int] = Field(default=[429, 500, 502, 503])


class AsyncConfig(BaseModel):
    """Concurrency settings for the async EvalRunner."""
    max_concurrent_queries: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=50, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)


class StatisticsConfig(BaseModel):
    """Statistical analysis settings."""
    tests: list[StatisticalTest] = Field(
        default=[StatisticalTest.WILCOXON, StatisticalTest.COHENS_D]
    )
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level")
    effect_size_threshold: float = Field(default=0.5, description="Cohen's d threshold for practical significance")
    save_figures: bool = Field(default=True)
    figures_dir: Path = Field(default=Path("outputs/figures"))


class MetricConfig(BaseModel):
    """Config for a single metric plugin."""
    name: TestName
    enabled: bool = True
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # NDCG-specific
    k: int = Field(default=10, ge=1, description="K for Recall@K and NDCG@K")

    # Consolidation delta-specific
    pre_consolidation_snapshot: Path | None = Field(
        default=None,
        description="Path to pre-consolidation knowledge base snapshot"
    )
    post_consolidation_snapshot: Path | None = Field(
        default=None,
        description="Path to post-consolidation knowledge base snapshot"
    )

    # Poisoning-specific
    poison_rate: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Fraction of corpus to poison for single_session_poisoning test"
    )

    # Query perturbation-specific
    perturbation_tool: str | None = Field(
        default="textattack",
        description="Tool for query perturbation: textattack | rgb | ares"
    )

    @field_validator("pre_consolidation_snapshot", "post_consolidation_snapshot", mode="before")
    @classmethod
    def coerce_path(cls, v: Any) -> Path | None:
        return Path(v) if v is not None else None


class DatasetConfig(BaseModel):
    """Input dataset configuration."""
    path: Path
    format: str = Field(default="json", pattern="^(json|jsonl|csv|parquet)$")
    question_col: str = Field(default="question")
    context_col: str = Field(default="contexts")
    answer_col: str = Field(default="answer")
    ground_truth_col: str = Field(default="ground_truth")
    sample_n: int | None = Field(default=None, description="Run on random sample of N rows (None = full dataset)")

    @field_validator("path", mode="before")
    @classmethod
    def coerce_path(cls, v: Any) -> Path:
        return Path(v)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """
    Root configuration for a single verity run.

    Loaded from YAML or constructed programmatically.
    All fields validated before execution begins — fail fast, not mid-run.

    Example YAML:
        experiment_id: consolidation_eval_run1
        architecture: consolidation
        model: claude-sonnet-4-6
        dataset:
          path: datasets/consolidation_rag_eval_set.json
        metrics:
          - name: ragas_consolidation_delta
            pre_consolidation_snapshot: snapshots/pre/
            post_consolidation_snapshot: snapshots/post/
          - name: ndcg
            k: 10
    """

    experiment_id: str = Field(..., description="Unique identifier for this run")
    architecture: RAGArchitecture
    model: SupportedModel | str = Field(..., description="LLM judge model string")
    dataset: DatasetConfig
    metrics: list[MetricConfig] = Field(default_factory=list, min_length=1)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    async_cfg: AsyncConfig = Field(default_factory=AsyncConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    output_dir: Path = Field(default=Path("outputs"))
    notes: str = Field(default="", description="Free-text run notes (logged to output manifest)")

    @model_validator(mode="after")
    def validate_consolidation_delta_snapshots(self) -> "EvalConfig":
        """
        If ragas_consolidation_delta is enabled, both snapshot paths must be set.
        Catches misconfigured consolidation evaluation runs at load time.
        """
        for m in self.metrics:
            if m.name == TestName.RAGAS_CONSOLIDATION_DELTA and m.enabled:
                if m.pre_consolidation_snapshot is None or m.post_consolidation_snapshot is None:
                    raise ValueError(
                        "ragas_consolidation_delta requires both "
                        "pre_consolidation_snapshot and post_consolidation_snapshot"
                    )
        return self

    @model_validator(mode="after")
    def warn_out_of_scope_tests(self) -> "EvalConfig":
        """Flag reserved Longitudinal Eval Suite tests if accidentally included."""
        out_of_scope = {TestName.SESSION_LEVEL, TestName.LONGITUDINAL}
        for m in self.metrics:
            if m.name in out_of_scope and m.enabled:
                raise ValueError(
                    f"Test '{m.name}' is reserved for longitudinal evaluation scope. "
                    "Set enabled=false or remove from config."
                )
        return self

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load and validate config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize validated config back to YAML."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    @property
    def enabled_metrics(self) -> list[MetricConfig]:
        return [m for m in self.metrics if m.enabled]
