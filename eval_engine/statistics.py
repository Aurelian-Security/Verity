"""
eval_engine/statistics.py

General-Purpose Statistical Analysis Module

PURPOSE:
    Provides post-run statistical analysis for eval-engine RunResult outputs.
    Designed to be metric-agnostic and paper-agnostic — works for Consolidation RAG
    Consolidation Eval Suite, rag-eval-harness, and the derivative paper pipeline.

INCLUDED TESTS:
    Significance:
        - Wilcoxon signed-rank test (paired, non-parametric)
          Use: pre/post consolidation delta significance
        - Mann-Whitney U test (unpaired, non-parametric)
          Use: centralized vs. decentralized architecture comparison
        - Paired t-test (parametric, assumes normality)
          Use: large samples where normality holds

    Effect Size:
        - Cohen's d (standardized mean difference)
          Use: practical significance of any score difference
        - Cohen's d paired (for repeated measures / pre-post designs)
        - Cliff's delta (non-parametric effect size)
          Use: when distributions are non-normal or ordinal

    Correlation:
        - Pearson r (linear correlation, parametric)
          Use: hardware metric → performance metric relationships
        - Spearman rho (rank correlation, non-parametric)
          Use: when linearity cannot be assumed

    Distribution:
        - Shapiro-Wilk normality test
          Use: decide between parametric and non-parametric tests
        - Descriptive statistics (mean, median, std, IQR, min, max)

OUTPUT:
    StatResult dataclass — consistent container for all tests.
    StatReport — aggregates multiple StatResults into a summary dict
                 suitable for JSON export and paper results tables.

DERIVATIVE PIPELINE NOTE:
    Papers 2-4 (adversarial robustness, alignment) will compare score
    distributions across attack conditions. Mann-Whitney U and Cliff's delta
    are the recommended defaults for those comparisons since score
    distributions under adversarial conditions are unlikely to be normal.

USAGE:
    from eval_engine.statistics import StatEngine, StatReport

    engine = StatEngine(alpha=0.05)

    # Pre/post delta significance (Consolidation Eval Suite Test 1)
    result = engine.wilcoxon(pre_scores, post_scores, label="faithfulness_delta")

    # Architecture comparison (rag-eval-harness)
    result = engine.mann_whitney(centralized_scores, decentralized_scores, label="ndcg_arch")

    # Effect size
    d = engine.cohens_d(pre_scores, post_scores, paired=True, label="faithfulness_d")

    # Full report from RunResult
    report = StatReport.from_run_results(pre_result, post_result, metric="ndcg")
    report.save(output_dir / "stats_report.json")
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StatResult:
    """
    Standardized container for a single statistical test result.

    All tests return one of these. StatReport aggregates multiple results.
    """
    label: str                          # Human-readable identifier for this test
    test_name: str                      # e.g. "wilcoxon", "cohens_d", "pearson"
    statistic: float | None             # Test statistic (W, U, t, r, d, etc.)
    p_value: float | None               # p-value (None for effect size / descriptive)
    effect_size: float | None           # Effect size where applicable
    significant: bool | None            # True if p < alpha (None for non-significance tests)
    alpha: float = 0.05                 # Significance threshold used
    n: int = 0                          # Sample size
    metadata: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""            # Plain-language interpretation for write-up

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"[{self.test_name}] {self.label}"]
        if self.statistic is not None:
            parts.append(f"stat={self.statistic:.4f}")
        if self.p_value is not None:
            parts.append(f"p={self.p_value:.4f}")
        if self.effect_size is not None:
            parts.append(f"effect={self.effect_size:.4f}")
        if self.significant is not None:
            parts.append("SIGNIFICANT" if self.significant else "not significant")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------

def _cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples (pooled SD)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled_sd = math.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired/repeated measures (difference scores)."""
    diff = b - a
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0.0
    return float(np.mean(diff) / sd_diff)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta — non-parametric effect size.
    Range: [-1, 1]. |d| > 0.474 = large, > 0.33 = medium, > 0.147 = small.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (n_a * n_b)


def _interpret_cohens_d(d: float) -> str:
    abs_d = abs(d)
    direction = "increase" if d > 0 else "decrease"
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    return f"{magnitude} {direction} (d={d:.3f})"


def _interpret_cliffs_delta(d: float) -> str:
    abs_d = abs(d)
    direction = "A > B" if d > 0 else "B > A"
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return f"{magnitude} effect, {direction} (delta={d:.3f})"


def _interpret_correlation(r: float) -> str:
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r < 0.1:
        magnitude = "negligible"
    elif abs_r < 0.3:
        magnitude = "weak"
    elif abs_r < 0.5:
        magnitude = "moderate"
    elif abs_r < 0.7:
        magnitude = "strong"
    else:
        magnitude = "very strong"
    return f"{magnitude} {direction} correlation (r={r:.3f})"


# ---------------------------------------------------------------------------
# StatEngine
# ---------------------------------------------------------------------------

class StatEngine:
    """
    General-purpose statistical analysis engine.

    All methods return StatResult for consistent downstream handling.
    All methods validate inputs and log warnings rather than raising
    on edge cases (empty arrays, zero variance) to avoid crashing
    a batch analysis run.

    Usage:
        engine = StatEngine(alpha=0.05)
        results = [
            engine.wilcoxon(pre, post, label="faithfulness"),
            engine.cohens_d(pre, post, paired=True, label="faithfulness_d"),
            engine.shapiro(pre, label="pre_normality"),
        ]
        report = StatReport(results, experiment_id="consolidation_rag_paper1")
        report.save(output_dir / "stats.json")
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------

    def wilcoxon(
        self,
        a: list[float] | np.ndarray,
        b: list[float] | np.ndarray,
        label: str = "wilcoxon",
        alternative: str = "two-sided",
    ) -> StatResult:
        """
        Wilcoxon signed-rank test (paired, non-parametric).

        Use for: pre/post consolidation score comparisons.
        Assumes: paired observations in same order (a[i] paired with b[i]).
        H0: median difference = 0.

        Args:
            a:           Pre-condition scores
            b:           Post-condition scores
            label:       Identifier for this test in the report
            alternative: "two-sided" | "greater" | "less"
        """
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)

        if len(a_arr) != len(b_arr):
            return self._error_result(label, "wilcoxon", f"arrays must be same length: {len(a_arr)} vs {len(b_arr)}")
        if len(a_arr) < 10:
            logger.warning(f"[wilcoxon:{label}] n={len(a_arr)} — Wilcoxon is unreliable with n < 10")

        try:
            stat, p = stats.wilcoxon(a_arr, b_arr, alternative=alternative)
            significant = bool(p < self.alpha)
            diff = b_arr - a_arr
            return StatResult(
                label=label,
                test_name="wilcoxon",
                statistic=float(stat),
                p_value=float(p),
                effect_size=None,
                significant=significant,
                alpha=self.alpha,
                n=len(a_arr),
                metadata={
                    "alternative": alternative,
                    "mean_diff": float(np.mean(diff)),
                    "median_diff": float(np.median(diff)),
                },
                interpretation=(
                    f"{'Significant' if significant else 'No significant'} difference "
                    f"(W={stat:.2f}, p={p:.4f}, n={len(a_arr)}, alpha={self.alpha})"
                ),
            )
        except Exception as e:
            return self._error_result(label, "wilcoxon", str(e))

    def mann_whitney(
        self,
        a: list[float] | np.ndarray,
        b: list[float] | np.ndarray,
        label: str = "mann_whitney",
        alternative: str = "two-sided",
    ) -> StatResult:
        """
        Mann-Whitney U test (unpaired, non-parametric).

        Use for: comparing two independent groups
                 (e.g. centralized vs. decentralized RAG scores,
                  clean vs. poisoned retrieval scores in Papers 2-4).
        H0: distributions are equal.
        """
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)

        if len(a_arr) == 0 or len(b_arr) == 0:
            return self._error_result(label, "mann_whitney", "arrays must not be empty")

        try:
            stat, p = stats.mannwhitneyu(a_arr, b_arr, alternative=alternative)
            significant = bool(p < self.alpha)
            cliff = _cliffs_delta(a_arr, b_arr)
            return StatResult(
                label=label,
                test_name="mann_whitney",
                statistic=float(stat),
                p_value=float(p),
                effect_size=cliff,
                significant=significant,
                alpha=self.alpha,
                n=len(a_arr) + len(b_arr),
                metadata={
                    "n_a": len(a_arr),
                    "n_b": len(b_arr),
                    "alternative": alternative,
                    "cliffs_delta": cliff,
                },
                interpretation=(
                    f"{'Significant' if significant else 'No significant'} difference "
                    f"(U={stat:.2f}, p={p:.4f}). "
                    f"Effect: {_interpret_cliffs_delta(cliff)}"
                ),
            )
        except Exception as e:
            return self._error_result(label, "mann_whitney", str(e))

    def paired_ttest(
        self,
        a: list[float] | np.ndarray,
        b: list[float] | np.ndarray,
        label: str = "paired_ttest",
        alternative: str = "two-sided",
    ) -> StatResult:
        """
        Paired t-test (parametric).

        Use when: n > 30 and Shapiro-Wilk confirms normality.
        Otherwise prefer wilcoxon().
        """
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)

        if len(a_arr) != len(b_arr):
            return self._error_result(label, "paired_ttest", "arrays must be same length")
        if len(a_arr) < 2:
            return self._error_result(label, "paired_ttest", "n must be >= 2")

        try:
            stat, p = stats.ttest_rel(a_arr, b_arr, alternative=alternative)
            significant = bool(p < self.alpha)
            d = _cohens_d_paired(a_arr, b_arr)
            return StatResult(
                label=label,
                test_name="paired_ttest",
                statistic=float(stat),
                p_value=float(p),
                effect_size=d,
                significant=significant,
                alpha=self.alpha,
                n=len(a_arr),
                metadata={"alternative": alternative, "cohens_d_paired": d},
                interpretation=(
                    f"{'Significant' if significant else 'No significant'} difference "
                    f"(t={stat:.3f}, p={p:.4f}). "
                    f"Effect: {_interpret_cohens_d(d)}"
                ),
            )
        except Exception as e:
            return self._error_result(label, "paired_ttest", str(e))

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------

    def cohens_d(
        self,
        a: list[float] | np.ndarray,
        b: list[float] | np.ndarray,
        paired: bool = False,
        label: str = "cohens_d",
    ) -> StatResult:
        """
        Cohen's d effect size.

        Args:
            paired: True for repeated measures / pre-post designs.
                    False for independent groups.
        """
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)

        if len(a_arr) < 2 or len(b_arr) < 2:
            return self._error_result(label, "cohens_d", "n must be >= 2 for each group")

        d = _cohens_d_paired(a_arr, b_arr) if paired else _cohens_d_independent(a_arr, b_arr)
        return StatResult(
            label=label,
            test_name="cohens_d",
            statistic=d,
            p_value=None,
            effect_size=d,
            significant=None,
            alpha=self.alpha,
            n=len(a_arr),
            metadata={"paired": paired},
            interpretation=_interpret_cohens_d(d),
        )

    def cliffs_delta(
        self,
        a: list[float] | np.ndarray,
        b: list[float] | np.ndarray,
        label: str = "cliffs_delta",
    ) -> StatResult:
        """
        Cliff's delta — non-parametric effect size.

        Recommended for Papers 2-4 where adversarial score distributions
        are unlikely to be normal.
        Range: [-1, 1].
        """
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
        d = _cliffs_delta(a_arr, b_arr)
        return StatResult(
            label=label,
            test_name="cliffs_delta",
            statistic=d,
            p_value=None,
            effect_size=d,
            significant=None,
            alpha=self.alpha,
            n=len(a_arr) + len(b_arr),
            metadata={"n_a": len(a_arr), "n_b": len(b_arr)},
            interpretation=_interpret_cliffs_delta(d),
        )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def pearson(
        self,
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        label: str = "pearson",
    ) -> StatResult:
        """
        Pearson r — linear correlation (parametric).

        Use for: hardware metric → performance degradation relationships
                 (e.g. available RAM → LlamaGuard FNR).
        Assumes: linear relationship, approximately normal distributions.
        """
        x_arr, y_arr = np.array(x, dtype=float), np.array(y, dtype=float)

        if len(x_arr) != len(y_arr):
            return self._error_result(label, "pearson", "arrays must be same length")
        if len(x_arr) < 3:
            return self._error_result(label, "pearson", "n must be >= 3")

        try:
            r, p = stats.pearsonr(x_arr, y_arr)
            significant = bool(p < self.alpha)
            return StatResult(
                label=label,
                test_name="pearson",
                statistic=float(r),
                p_value=float(p),
                effect_size=float(r ** 2),   # R² = variance explained
                significant=significant,
                alpha=self.alpha,
                n=len(x_arr),
                metadata={"r_squared": float(r ** 2)},
                interpretation=(
                    f"{_interpret_correlation(r)}. "
                    f"R²={r**2:.3f} (variance explained). "
                    f"{'Significant' if significant else 'Not significant'} (p={p:.4f})"
                ),
            )
        except Exception as e:
            return self._error_result(label, "pearson", str(e))

    def spearman(
        self,
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        label: str = "spearman",
    ) -> StatResult:
        """
        Spearman rho — rank correlation (non-parametric).

        Use when: linearity cannot be assumed, or variables are ordinal.
        Preferred over Pearson when score distributions are skewed.
        """
        x_arr, y_arr = np.array(x, dtype=float), np.array(y, dtype=float)

        if len(x_arr) != len(y_arr):
            return self._error_result(label, "spearman", "arrays must be same length")
        if len(x_arr) < 3:
            return self._error_result(label, "spearman", "n must be >= 3")

        try:
            rho, p = stats.spearmanr(x_arr, y_arr)
            significant = bool(p < self.alpha)
            return StatResult(
                label=label,
                test_name="spearman",
                statistic=float(rho),
                p_value=float(p),
                effect_size=float(rho),
                significant=significant,
                alpha=self.alpha,
                n=len(x_arr),
                metadata={},
                interpretation=(
                    f"{_interpret_correlation(rho)} (rank-based). "
                    f"{'Significant' if significant else 'Not significant'} (p={p:.4f})"
                ),
            )
        except Exception as e:
            return self._error_result(label, "spearman", str(e))

    # ------------------------------------------------------------------
    # Distribution / normality
    # ------------------------------------------------------------------

    def shapiro(
        self,
        x: list[float] | np.ndarray,
        label: str = "shapiro",
    ) -> StatResult:
        """
        Shapiro-Wilk normality test.

        Use BEFORE choosing between parametric (t-test, Pearson) and
        non-parametric (Wilcoxon, Mann-Whitney, Spearman) tests.

        If p > alpha → fail to reject normality → parametric tests acceptable.
        If p < alpha → reject normality → use non-parametric tests.

        Reliable for n < 5000. For larger samples use descriptive checks.
        """
        x_arr = np.array(x, dtype=float)

        if len(x_arr) < 3:
            return self._error_result(label, "shapiro", "n must be >= 3")
        if len(x_arr) > 5000:
            logger.warning(f"[shapiro:{label}] n={len(x_arr)} > 5000 — Shapiro-Wilk unreliable. Use descriptive checks.")

        try:
            stat, p = stats.shapiro(x_arr)
            normal = bool(p >= self.alpha)
            return StatResult(
                label=label,
                test_name="shapiro",
                statistic=float(stat),
                p_value=float(p),
                effect_size=None,
                significant=not normal,  # significant = evidence AGAINST normality
                alpha=self.alpha,
                n=len(x_arr),
                metadata={"appears_normal": normal},
                interpretation=(
                    f"Distribution {'appears normal' if normal else 'does NOT appear normal'} "
                    f"(W={stat:.4f}, p={p:.4f}). "
                    f"{'Parametric tests acceptable.' if normal else 'Use non-parametric tests.'}"
                ),
            )
        except Exception as e:
            return self._error_result(label, "shapiro", str(e))

    def descriptive(
        self,
        x: list[float] | np.ndarray,
        label: str = "descriptive",
    ) -> StatResult:
        """
        Descriptive statistics — mean, median, std, IQR, min, max, n.
        Always run this first for any score distribution.
        """
        x_arr = np.array(x, dtype=float)

        if len(x_arr) == 0:
            return self._error_result(label, "descriptive", "array is empty")

        q1, q3 = float(np.percentile(x_arr, 25)), float(np.percentile(x_arr, 75))
        return StatResult(
            label=label,
            test_name="descriptive",
            statistic=float(np.mean(x_arr)),
            p_value=None,
            effect_size=None,
            significant=None,
            alpha=self.alpha,
            n=len(x_arr),
            metadata={
                "mean": float(np.mean(x_arr)),
                "median": float(np.median(x_arr)),
                "std": float(np.std(x_arr, ddof=1)),
                "min": float(np.min(x_arr)),
                "max": float(np.max(x_arr)),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
            },
            interpretation=(
                f"n={len(x_arr)}, mean={np.mean(x_arr):.4f}, "
                f"median={np.median(x_arr):.4f}, std={np.std(x_arr, ddof=1):.4f}, "
                f"IQR=[{q1:.4f}, {q3:.4f}]"
            ),
        )

    # ------------------------------------------------------------------
    # Convenience: full pre/post analysis bundle
    # ------------------------------------------------------------------

    def pre_post_bundle(
        self,
        pre_scores: list[float] | np.ndarray,
        post_scores: list[float] | np.ndarray,
        metric_label: str = "metric",
    ) -> list[StatResult]:
        """
        Run the standard pre/post analysis bundle for a single metric.

        Returns results for:
            1. Descriptive (pre)
            2. Descriptive (post)
            3. Shapiro-Wilk (pre) — decide parametric vs non-parametric
            4. Shapiro-Wilk (post)
            5. Wilcoxon signed-rank (primary significance test)
            6. Cohen's d paired (effect size)
            7. Cliff's delta (non-parametric effect size, for Papers 2-4)

        This bundle produces everything needed for a Consolidation Eval Suite results table row.
        """
        return [
            self.descriptive(pre_scores, label=f"{metric_label}_pre_descriptive"),
            self.descriptive(post_scores, label=f"{metric_label}_post_descriptive"),
            self.shapiro(pre_scores, label=f"{metric_label}_pre_normality"),
            self.shapiro(post_scores, label=f"{metric_label}_post_normality"),
            self.wilcoxon(pre_scores, post_scores, label=f"{metric_label}_wilcoxon"),
            self.cohens_d(pre_scores, post_scores, paired=True, label=f"{metric_label}_cohens_d"),
            self.cliffs_delta(pre_scores, post_scores, label=f"{metric_label}_cliffs_delta"),
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _error_result(self, label: str, test_name: str, error: str) -> StatResult:
        logger.error(f"[{test_name}:{label}] {error}")
        return StatResult(
            label=label,
            test_name=test_name,
            statistic=None,
            p_value=None,
            effect_size=None,
            significant=None,
            alpha=self.alpha,
            n=0,
            metadata={"error": error},
            interpretation=f"ERROR: {error}",
        )


# ---------------------------------------------------------------------------
# StatReport
# ---------------------------------------------------------------------------

class StatReport:
    """
    Aggregates multiple StatResult objects into a structured report.

    Produces:
        - JSON export for technical write-up and paper appendix
        - Console summary table
        - Significance summary (which tests passed/failed)

    Usage:
        report = StatReport(results, experiment_id="consolidation_rag_paper1_test1")
        report.print_summary()
        report.save(output_dir / "stats_report.json")
    """

    def __init__(
        self,
        results: list[StatResult],
        experiment_id: str = "",
        notes: str = "",
    ) -> None:
        self.results = results
        self.experiment_id = experiment_id
        self.notes = notes

    @classmethod
    def from_score_lists(
        cls,
        pre_scores: list[float],
        post_scores: list[float],
        metric_label: str,
        experiment_id: str = "",
        alpha: float = 0.05,
    ) -> "StatReport":
        """
        Convenience constructor: run the full pre/post bundle and return a report.
        Most common use case for Consolidation Eval Suite Test 1.
        """
        engine = StatEngine(alpha=alpha)
        results = engine.pre_post_bundle(pre_scores, post_scores, metric_label)
        return cls(results, experiment_id=experiment_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "notes": self.notes,
            "n_tests": len(self.results),
            "significant_tests": [r.label for r in self.results if r.significant is True],
            "non_significant_tests": [r.label for r in self.results if r.significant is False],
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path | str) -> Path:
        """Save report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"StatReport saved: {path}")
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  STATISTICAL REPORT: {self.experiment_id}")
        print(f"{'='*60}")
        for r in self.results:
            sig_marker = ""
            if r.significant is True:
                sig_marker = " *** SIGNIFICANT"
            elif r.significant is False:
                sig_marker = " (ns)"
            print(f"\n  [{r.test_name.upper()}] {r.label}{sig_marker}")
            print(f"  {r.interpretation}")
        print(f"\n{'='*60}\n")

    @property
    def any_significant(self) -> bool:
        return any(r.significant is True for r in self.results)

    @property
    def significant_labels(self) -> list[str]:
        return [r.label for r in self.results if r.significant is True]
