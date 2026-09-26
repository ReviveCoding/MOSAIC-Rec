from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import assert_outcome_access, load_stage_status
from mosaic_rec.p2.data import controlled_zero_selected
from mosaic_rec.p2.features import _copy_new
from mosaic_rec.p2.runtime import load_histories


def test_protocol_identity_and_protected_roles() -> None:
    repo = Path(__file__).resolve().parents[2]
    status = load_stage_status(repo)
    assert status["protocol_id"] == PROTOCOL_ID
    assert status["qualification_access_count"] == 1
    assert status["locked_final_access_count"] == 0
    assert_outcome_access(repo, "VALIDATION")
    assert_outcome_access(repo, "QUALIFICATION")
    with pytest.raises(PermissionError):
        assert_outcome_access(repo, "LOCKED_FINAL")


def test_preregistered_configs_are_p2_only() -> None:
    repo = Path(__file__).resolve().parents[2]
    for path in (repo / "configs/p2").glob("*.json"):
        payload = json.loads(path.read_text())
        assert payload["protocol_id"] == PROTOCOL_ID
        assert "mosaic-rec-2026-09-20-p1" not in json.dumps(payload.get("outputs", {}))


def test_controlled_zero_selection_is_deterministic_and_distinct() -> None:
    item_ids = [f"i{index}" for index in range(10000)]
    first = [item for item in item_ids if controlled_zero_selected(item)]
    second = [item for item in item_ids if controlled_zero_selected(item)]
    assert first == second
    assert 400 <= len(first) <= 600


def test_feature_copy_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_bytes(b"feature")
    _copy_new(source, target)
    assert target.read_bytes() == b"feature"
    with pytest.raises(FileExistsError):
        _copy_new(source, target)


def test_candidate_semantics_amendment_keeps_target() -> None:
    repo = Path(__file__).resolve().parents[2]
    amendment = json.loads(
        (repo / "configs/p2/amendment_002_candidate_semantics.json").read_text()
    )
    assert amendment["candidate_mask"]["same_for_all_models"] is True
    assert "except the target item" in amendment["candidate_mask"]["history"]


def test_history_loader_retains_full_fit_history_for_masking(tmp_path: Path) -> None:
    path = tmp_path / "fit.parquet"
    pq.write_table(
        pa.table(
            {
                "user_id": ["u1"] * 60,
                "item_id": [f"i{index}" for index in range(60)],
                "timestamp": list(range(60)),
            }
        ),
        path,
    )
    histories = load_histories(path, {f"i{index}": index for index in range(60)}, 50)
    assert len(histories["u1"]) == 60
