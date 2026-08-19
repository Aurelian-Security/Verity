"""
eval_engine/dataset_manifest.py

Dataset Manifest Generator — Benchmark Versioning

PURPOSE:
    Content-hashes evaluation datasets at load time, producing a
    dataset_manifest.json that uniquely identifies the exact data
    that produced a set of results.

    Prevents the "which dataset produced this result?" problem —
    the most common reproducibility failure in ML evaluation papers.

    Also tracks dataset lineage for poisoning/consolidation pipelines:
        clean_corpus → poisoned_corpus → consolidated_corpus
    Each transformation is recorded with its own content hash.

OUTPUTS:
    dataset_manifest.json per run:
        {
          "dataset_id": "sha256:a3f9...",    ← content hash (version ID)
          "path": "datasets/eval_set.json",
          "row_count": 500,
          "columns": ["question", "contexts", "answer", "ground_truth"],
          "schema": {...},
          "file_size_bytes": 142832,
          "created_at": "2025-...",
          "lineage": [],                      ← transformation chain
          "version_tag": null,                ← optional human label
        }

LINEAGE TRACKING:
    For poisoning/consolidation pipelines, record each transformation:

        manifest = DatasetManifest.from_file("datasets/eval_set.json")
        manifest.add_lineage_step(
            step_name="single_session_poisoning",
            poison_rate=0.1,
            attack_type="factual_substitution",
            output_path="datasets/poisoned_eval_set.json",
        )
        manifest.save(output_dir)

PAPER NOTE:
    Include dataset_manifest.json in supplementary materials.
    The dataset_id (sha256 hash) is what reviewers use to verify
    they're running the same eval set as the paper.
    Use version_tag to label benchmark releases: "v1.0", "v1.1".
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lineage step
# ---------------------------------------------------------------------------

@dataclass
class LineageStep:
    """Single transformation in a dataset lineage chain."""
    step_name: str
    timestamp: float = field(default_factory=time.time)
    input_dataset_id: str = ""
    output_path: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "timestamp_iso": _ts_to_iso(self.timestamp),
            "input_dataset_id": self.input_dataset_id,
            "output_path": self.output_path,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """
    Content-addressed dataset descriptor.

    Uniquely identifies a dataset by its content hash (SHA-256).
    Records schema, row count, columns, and transformation lineage.
    """
    path: str
    dataset_id: str = ""          # SHA-256 content hash — the version ID
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    schema: dict[str, str] = field(default_factory=dict)
    file_size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    version_tag: str | None = None
    lineage: list[LineageStep] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)  # first 3 rows for inspection

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        version_tag: str | None = None,
        include_sample: bool = True,
    ) -> "DatasetManifest":
        """
        Build manifest from a dataset file.
        Computes SHA-256 hash of file contents.

        Supports: JSON, JSONL, CSV, Parquet
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        file_size = path.stat().st_size
        dataset_id = _hash_file(path)
        records = _load_records(path)

        columns = list(records[0].keys()) if records else []
        schema = _infer_schema(records[:10] if records else [])
        sample = records[:3] if include_sample else []

        manifest = cls(
            path=str(path),
            dataset_id=dataset_id,
            row_count=len(records),
            columns=columns,
            schema=schema,
            file_size_bytes=file_size,
            created_at=time.time(),
            version_tag=version_tag,
            sample_rows=sample,
        )

        logger.info(
            f"[dataset_manifest] {path.name}: "
            f"{len(records)} rows, "
            f"{len(columns)} columns, "
            f"id={dataset_id[:16]}..."
        )
        return manifest

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        source_path: str = "in_memory",
        version_tag: str | None = None,
    ) -> "DatasetManifest":
        """
        Build manifest from an in-memory record list.
        Hash computed from JSON serialization of records.
        """
        serialized = json.dumps(records, sort_keys=True, default=str).encode()
        dataset_id = "sha256:" + hashlib.sha256(serialized).hexdigest()
        columns = list(records[0].keys()) if records else []
        schema = _infer_schema(records[:10] if records else [])

        return cls(
            path=source_path,
            dataset_id=dataset_id,
            row_count=len(records),
            columns=columns,
            schema=schema,
            file_size_bytes=len(serialized),
            created_at=time.time(),
            version_tag=version_tag,
            sample_rows=records[:3],
        )

    def add_lineage_step(
        self,
        step_name: str,
        output_path: str = "",
        **parameters: Any,
    ) -> "DatasetManifest":
        """
        Record a transformation step in the lineage chain.

        Usage for poisoning pipeline:
            manifest.add_lineage_step(
                "single_session_poisoning",
                output_path="datasets/poisoned.json",
                poison_rate=0.1,
                attack_type="factual_substitution",
                seed=42,
            )
        """
        step = LineageStep(
            step_name=step_name,
            input_dataset_id=self.dataset_id,
            output_path=output_path,
            parameters=parameters,
        )
        self.lineage.append(step)
        logger.info(f"[dataset_manifest] Lineage step added: {step_name}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "version_tag": self.version_tag,
            "row_count": self.row_count,
            "columns": self.columns,
            "schema": self.schema,
            "file_size_bytes": self.file_size_bytes,
            "created_at_iso": _ts_to_iso(self.created_at),
            "lineage": [s.to_dict() for s in self.lineage],
            "sample_rows": self.sample_rows,
            "manifest_note": (
                "dataset_id is SHA-256 of file contents. "
                "Use this to verify you are running the same dataset as the paper."
            ),
        }

    def save(self, output_dir: Path | str) -> Path:
        """Save manifest to output_dir/dataset_manifest.json."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "dataset_manifest.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"[dataset_manifest] Saved: {path}")
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  DATASET MANIFEST")
        print(f"{'='*55}")
        print(f"  Path:         {self.path}")
        print(f"  Dataset ID:   {self.dataset_id[:32]}...")
        print(f"  Version tag:  {self.version_tag or 'untagged'}")
        print(f"  Rows:         {self.row_count:,}")
        print(f"  Columns:      {', '.join(self.columns)}")
        print(f"  File size:    {self.file_size_bytes:,} bytes")
        if self.lineage:
            print(f"  Lineage:")
            for step in self.lineage:
                print(f"    → {step.step_name}")
        print(f"{'='*55}\n")

    @property
    def short_id(self) -> str:
        """First 12 chars of dataset_id for display."""
        return self.dataset_id[:12] + "..."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from JSON, JSONL, CSV, or Parquet."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        elif suffix == ".jsonl":
            records = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif suffix == ".csv":
            import pandas as pd
            return pd.read_csv(path).to_dict(orient="records")
        elif suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(path).to_dict(orient="records")
        else:
            logger.warning(f"[dataset_manifest] Unknown extension {suffix} — trying JSON")
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        logger.error(f"[dataset_manifest] Failed to load {path}: {e}")
        return []


def _infer_schema(records: list[dict[str, Any]]) -> dict[str, str]:
    """Infer column types from first N records."""
    if not records:
        return {}
    schema: dict[str, str] = {}
    for key in records[0].keys():
        types = set()
        for rec in records:
            val = rec.get(key)
            types.add(type(val).__name__)
        schema[key] = "/".join(sorted(types))
    return schema


def _ts_to_iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
