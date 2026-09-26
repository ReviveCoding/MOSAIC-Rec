"""Shared contracts for read-only MOSAIC-Rec p2 posthoc diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_PROTOCOL = "mosaic-rec-2026-09-23-p2"
SOURCE_COMMIT = "b9bdc1bcb6abf912687ffb304f426107349cd06f"
EVIDENCE_CLASS = "POSTHOC_EXPLORATORY_READ_ONLY"
ANALYSIS_VERSION = "1.0.0"
BOOTSTRAP_SEED_BASE = 2026092601


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def metadata(
    *, script: Path, source_hashes: dict[str, str], row_count: int
) -> dict[str, Any]:
    return {
        "evidence_class": EVIDENCE_CLASS,
        "source_protocol": SOURCE_PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "protected_scoring_rerun": False,
        "canonical_p2_mutated": False,
        "generated_utc": generated_utc(),
        "analysis_version": ANALYSIS_VERSION,
        "row_count": int(row_count),
        "script_path": script.as_posix(),
        "script_sha256": sha256_file(script),
        "source_file_sha256": source_hashes,
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_parquet_new(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    script: Path,
    source_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    table = pa.Table.from_pylist(rows)
    meta = metadata(script=script, source_hashes=source_hashes, row_count=len(rows))
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"mosaic_posthoc_metadata"] = json.dumps(
        meta, sort_keys=True, separators=(",", ":")
    ).encode()
    pq.write_table(
        table.replace_schema_metadata(schema_metadata), path, compression="zstd"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value
