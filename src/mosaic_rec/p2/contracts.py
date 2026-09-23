"""Fail-closed p2 identities, hashing, protected access, and atomic receipts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID

P1_IMMUTABLE_HASHES = {
    "reports/FINAL_STATUS.md": (
        "79994c8583e2b56bb823c08a32dd8da2b2b850c5409f1e31b6906b881dd6f553"
    ),
    "evidence/claims.json": (
        "41153619f71023fb2ae2ebf8168d85be05376ee72badec2b81d2c56777cdf218"
    ),
    "evidence/stage_status.json": (
        "021f340d36a4b5bec5066b74620f9efa76e9d59c3a6928acc4e9d58a3cc593cd"
    ),
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    """Create a terminal receipt without permitting overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def replace_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a living p2 control file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_p1_immutable(repo: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in P1_IMMUTABLE_HASHES.items():
        actual = sha256_file(repo / relative)
        if actual != expected:
            raise RuntimeError(f"Closed p1 artifact changed: {relative}")
        observed[relative] = actual
    return observed


def load_stage_status(repo: Path) -> dict[str, Any]:
    status = load_json(repo / "evidence/p2/stage_status.json")
    if status.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("p2 stage status protocol mismatch")
    return status


def assert_outcome_access(repo: Path, role: str) -> None:
    """Guard model-outcome access; materializing source role rows is separate."""
    status = load_stage_status(repo)
    state = status["scientific_state"]
    if role in {"FIT_VISIBLE", "VALIDATION"}:
        return
    if role == "QUALIFICATION":
        if state not in {"QUALIFICATION", "FROZEN"}:
            raise PermissionError("p2 QUALIFICATION model outcomes are sealed")
        return
    if role == "LOCKED_FINAL":
        if state != "LOCKED_EXECUTION":
            raise PermissionError("p2 LOCKED_FINAL model outcomes are sealed")
        manifest = repo / "evidence/p2/e05_freeze_manifest.json"
        expected = status.get("frozen_manifest_sha256")
        if not expected or not manifest.is_file():
            raise PermissionError("p2 LOCKED_FINAL requires a frozen manifest")
        if sha256_file(manifest) != expected:
            raise PermissionError("p2 frozen manifest hash mismatch")
        return
    raise ValueError(f"Unknown p2 role: {role}")
