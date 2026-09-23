from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic_rec.score_qualification import (
    candidate_from_manifest,
    qualification_state_guard,
)

ROOT = Path(__file__).resolve().parents[1]


def manifest() -> dict:
    return json.loads((ROOT / "configs/qualification_selection_p1.json").read_text())


def test_manifest_freezes_candidate_and_seed_identity() -> None:
    frozen = manifest()
    assert len(frozen["candidates"]) == 7
    assert frozen["state_transition"]["authorized_roles"] == ["QUALIFICATION"]
    assert frozen["state_transition"]["prohibited_roles"] == ["LOCKED_FINAL"]
    for candidate in frozen["candidates"]:
        assert [row["seed"] for row in candidate["checkpoints"]] == [
            20260924,
            20260925,
            20260926,
        ]


def test_candidate_identity_must_be_unique() -> None:
    frozen = manifest()
    assert candidate_from_manifest(frozen, "B7_BERT4REC_T3")["architecture"] == (
        "bert4rec"
    )
    with pytest.raises(ValueError, match="not unique"):
        candidate_from_manifest(frozen, "ABSENT")


def test_qualification_guard_closes_after_e05_freeze() -> None:
    with pytest.raises(RuntimeError, match="closed after E05"):
        qualification_state_guard(ROOT, manifest())
