"""Protocol role-access and state-transition guards for protected outcomes."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any


class ScientificState(str, Enum):
    PILOT = "PILOT"
    DEVELOPMENT = "DEVELOPMENT"
    QUALIFICATION = "QUALIFICATION"
    FROZEN = "FROZEN"
    LOCKED_EXECUTION = "LOCKED_EXECUTION"
    ANALYSIS = "ANALYSIS"
    CLOSED = "CLOSED"


ORDER = tuple(ScientificState)
REQUIRED_FREEZE_FIELDS = (
    "protocol_id",
    "git_commit",
    "environment_sha256",
    "data_sha256",
    "split_sha256",
    "cold_cohort_sha256",
    "feature_sha256",
    "baseline_sequence",
    "baseline_cold",
    "mosaic_finalist",
    "seed_manifest",
    "hpo_selection",
    "metric_definitions",
    "candidate_availability",
    "noninferiority_margin",
    "promotion_gates",
    "robustness_definitions",
    "ablation_family",
)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_freeze_manifest(path: Path, protocol_id: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("Freeze manifest must be a JSON object")
    missing = [
        field
        for field in REQUIRED_FREEZE_FIELDS
        if field not in manifest or manifest[field] in (None, "", [], {})
    ]
    if missing:
        raise ValueError(f"Freeze manifest lacks required fields: {', '.join(missing)}")
    if manifest["protocol_id"] != protocol_id:
        raise ValueError("Freeze manifest protocol identity mismatch")
    margin = manifest["noninferiority_margin"]
    if not isinstance(margin, (int, float)) or not 0 < margin < 1:
        raise ValueError(
            "Warm noninferiority margin must be a predeclared number in (0, 1)"
        )
    return manifest


def assert_role_access(
    stage_status: dict[str, Any], role: str, frozen_manifest_path: Path | None = None
) -> None:
    """Fail closed before reading any protected role outcome rows."""
    state = ScientificState(stage_status["scientific_state"])
    if role in ("FIT", "FIT_VISIBLE", "VALIDATION"):
        return
    if role == "QUALIFICATION":
        if ORDER.index(state) < ORDER.index(ScientificState.QUALIFICATION):
            raise PermissionError("QUALIFICATION outcomes are sealed in this state")
        return
    if role != "LOCKED_FINAL":
        raise ValueError(f"Unknown outcome role: {role}")
    if ORDER.index(state) < ORDER.index(ScientificState.LOCKED_EXECUTION):
        raise PermissionError(
            "LOCKED_FINAL outcomes are sealed before locked execution"
        )
    if frozen_manifest_path is None or not frozen_manifest_path.is_file():
        raise PermissionError("LOCKED_FINAL requires an existing frozen manifest")
    digest = stage_status.get("frozen_manifest_sha256")
    if not digest or digest != manifest_sha256(frozen_manifest_path):
        raise PermissionError("Frozen manifest identity mismatch")
    validate_freeze_manifest(frozen_manifest_path, stage_status["protocol_id"])


def assert_transition(
    stage_status: dict[str, Any],
    target: ScientificState,
    *,
    frozen_manifest_path: Path | None = None,
    locked_execution_receipts: int = 0,
) -> None:
    current = ScientificState(stage_status["scientific_state"])
    if ORDER.index(target) != ORDER.index(current) + 1:
        raise ValueError(
            f"Invalid protocol transition {current.value} -> {target.value}"
        )
    if (
        target in (ScientificState.QUALIFICATION, ScientificState.FROZEN)
        and locked_execution_receipts
    ):
        raise ValueError("Protected execution occurred before qualification/freeze")
    if target == ScientificState.FROZEN:
        if frozen_manifest_path is None or not frozen_manifest_path.is_file():
            raise ValueError("Freeze transition requires a complete manifest")
        validate_freeze_manifest(frozen_manifest_path, stage_status["protocol_id"])
    if target == ScientificState.LOCKED_EXECUTION:
        if frozen_manifest_path is None:
            raise ValueError("Locked execution requires frozen manifest identity")
        digest = stage_status.get("frozen_manifest_sha256")
        if not digest or digest != manifest_sha256(frozen_manifest_path):
            raise ValueError("Frozen manifest identity mismatch")
