"""A CUDA infrastructure retry is distinct, narrow, and auditable."""

from __future__ import annotations

import json

from scripts.run_e01_seed_qualification_p1 import infrastructure_retry_eligible


def test_cuda_unknown_without_checkpoint_is_retry_eligible(tmp_path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "status": "FAIL_LOCAL_PILOT",
                "exception": "CUDA_ERROR_UNKNOWN from cuStreamSynchronize",
            }
        )
    )
    receipt = {
        "status": "E01_SEED_FAIL",
        "returncode": 1,
        "checkpoint": None,
        "validation_slices": None,
        "raw_receipt": str(raw),
    }
    assert infrastructure_retry_eligible(receipt)


def test_scientific_failure_is_not_retry_eligible(tmp_path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"status": "FAIL_LOCAL_PILOT", "exception": "NaN"}))
    receipt = {
        "status": "E01_SEED_FAIL",
        "returncode": 1,
        "checkpoint": None,
        "validation_slices": None,
        "raw_receipt": str(raw),
    }
    assert not infrastructure_retry_eligible(receipt)
