"""
eval_engine/reproducibility.py

Reproducibility Bundle Generator

PURPOSE:
    Captures everything needed to reproduce an eval run exactly:
        - Random seed used for sampling, poisoning, perturbation
        - Python version and platform
        - All installed package versions (dependency lockfile snapshot)
        - Git commit hash (if in a git repo)
        - EvalConfig snapshot (the exact config that produced this run)
        - Timestamp and experiment ID

    Written to: outputs/{experiment_id}/reproducibility.json

    This is what you attach to a paper submission to satisfy
    "our results are reproducible." One file, one command.

USAGE:
    from eval_engine.reproducibility import ReproducibilityBundle

    bundle = ReproducibilityBundle(
        experiment_id="consolidation_eval_run1",
        seed=42,
        config_path="configs/consolidation_eval_example.yaml",
    )
    bundle.capture()
    bundle.save(output_dir / "reproducibility.json")

    # Or from EvalConfig directly:
    bundle = ReproducibilityBundle.from_config(eval_config, seed=42)
    bundle.capture()
    bundle.save(output_dir)

PAPER NOTE:
    Include reproducibility.json as a supplementary file in submissions.
    The git_hash field ties results to a specific code state.
    If git_hash is None (repo not initialized), note this in the paper.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class ReproducibilityBundle:
    """
    Complete reproducibility snapshot for one eval run.

    Capture once per run, save alongside results.
    """
    experiment_id: str
    seed: int = 42
    config_path: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # Populated by capture()
    timestamp: float = field(default_factory=time.time)
    python_version: str = ""
    platform_info: str = ""
    git_hash: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    installed_packages: dict[str, str] = field(default_factory=dict)
    verity_version: str = ""
    captured: bool = False

    def capture(self) -> "ReproducibilityBundle":
        """
        Capture current environment state.
        Call once before running the experiment.
        """
        self.timestamp = time.time()
        self.python_version = sys.version
        self.platform_info = platform.platform()
        self.verity_version = _get_verity_version()
        self.git_hash = _get_git_hash()
        self.git_branch = _get_git_branch()
        self.git_dirty = _get_git_dirty()
        self.installed_packages = _get_installed_packages()
        self.captured = True

        if self.git_dirty:
            logger.warning(
                "[reproducibility] Git working tree is dirty (uncommitted changes). "
                "Results may not be reproducible from git_hash alone. "
                "Commit all changes before running experiments for paper submission."
            )

        if self.git_hash is None:
            logger.warning(
                "[reproducibility] Not in a git repository. "
                "git_hash will be None in reproducibility.json. "
                "Initialize a git repo for full reproducibility tracking."
            )

        logger.info(
            f"[reproducibility] Captured: "
            f"seed={self.seed}, "
            f"git={self.git_hash or 'N/A'}, "
            f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "timestamp_iso": _ts_to_iso(self.timestamp),
            "verity_version": self.verity_version,
            "python_version": self.python_version,
            "platform": self.platform_info,
            "git": {
                "hash": self.git_hash,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "config_path": self.config_path,
            "config_snapshot": self.config_snapshot,
            "installed_packages": self.installed_packages,
            "captured": self.captured,
            "reproducibility_note": (
                "To reproduce: use the same seed, git hash, and config_snapshot. "
                "Install packages matching installed_packages versions."
            ),
        }

    def save(self, output_dir: Path | str) -> Path:
        """Save reproducibility bundle to output_dir/reproducibility.json."""
        if not self.captured:
            logger.warning("[reproducibility] capture() not called — saving empty bundle.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "reproducibility.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"[reproducibility] Saved: {path}")
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  REPRODUCIBILITY BUNDLE: {self.experiment_id}")
        print(f"{'='*55}")
        print(f"  Seed:         {self.seed}")
        print(f"  Git hash:     {self.git_hash or 'N/A'}")
        print(f"  Git branch:   {self.git_branch or 'N/A'}")
        print(f"  Git dirty:    {self.git_dirty}")
        print(f"  Python:       {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"  Platform:     {platform.platform()}")
        print(f"  Verity:       {self.verity_version}")
        print(f"  Packages:     {len(self.installed_packages)} recorded")
        if self.git_dirty:
            print(f"  ⚠  WARNING: Uncommitted changes — commit before paper submission")
        print(f"{'='*55}\n")

    @classmethod
    def from_config(
        cls,
        config: Any,  # EvalConfig
        seed: int = 42,
    ) -> "ReproducibilityBundle":
        """Build bundle from EvalConfig, capturing config snapshot."""
        try:
            config_snapshot = config.model_dump(mode="json")
        except Exception:
            config_snapshot = {}

        return cls(
            experiment_id=config.experiment_id,
            seed=seed,
            config_path=None,
            config_snapshot=config_snapshot,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _get_installed_packages() -> dict[str, str]:
    """Return dict of {package_name: version} for all installed packages."""
    try:
        import importlib.metadata as importlib_metadata
        packages = {}
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name", "")
            version = dist.metadata.get("Version", "")
            if name:
                packages[name.lower()] = version
        return dict(sorted(packages.items()))
    except Exception as e:
        logger.warning(f"[reproducibility] Could not capture installed packages: {e}")
        return {}


def _get_verity_version() -> str:
    try:
        from eval_engine import __version__
        return __version__
    except Exception:
        return "unknown"


def _ts_to_iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    """
    Set random seed globally across all libraries used by Verity.
    Call at the start of any run where reproducibility matters.

    Covers: Python random, numpy, (torch if available).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"[reproducibility] Torch seed set: {seed}")
    except (ImportError, OSError, Exception):
        pass

    logger.info(f"[reproducibility] Global seed set: {seed}")
