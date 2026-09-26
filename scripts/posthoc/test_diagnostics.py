from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from diagnostics import (
    age_bucket,
    assert_paired,
    concentration,
    history_bucket,
    jaccard,
    oracle_values,
    paired_bootstrap,
    recency_bucket,
    support_bucket,
    temporal_quartiles,
    validate_output_path,
    win_tie_loss,
)


def test_slice_boundaries() -> None:
    assert [support_bucket(x) for x in (0, 1, 2, 3, 5, 6, 10, 11, 20, 21, 50, 51)] == [
        "0",
        "1-2",
        "1-2",
        "3-5",
        "3-5",
        "6-10",
        "6-10",
        "11-20",
        "11-20",
        "21-50",
        "21-50",
        ">50",
    ]
    assert [age_bucket(x) for x in (0.5, 1, 7, 8, 30, 31, 90, 91, 365, 366)] == [
        "<1 day",
        "1-7",
        "1-7",
        "8-30",
        "8-30",
        "31-90",
        "31-90",
        "91-365",
        "91-365",
        ">365",
    ]
    assert history_bucket(50) == "21-50" and history_bucket(51) == ">50"
    assert recency_bucket(1) == "<=1 day" and recency_bucket(1.1) == "1-7"


def test_paired_identity_and_accounting() -> None:
    frame = pd.DataFrame(
        {"user_id": ["a"], "query_timestamp": [1], "target_item_id": ["x"]}
    )
    assert_paired([frame, frame.copy()])
    with pytest.raises(ValueError, match="not paired"):
        assert_paired([frame, frame.assign(target_item_id="y")])
    result = win_tie_loss(np.array([1.0, 0.0, -1.0, 0.0]))
    assert (result["win_n"], result["tie_n"], result["loss_n"]) == (1, 2, 1)
    assert win_tie_loss(np.asarray([]))["n"] == 0


def test_oracle_jaccard_and_concentration() -> None:
    oracle, regret = oracle_values(
        np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([0.5, 3.0])
    )
    assert np.array_equal(oracle, [1.0, 2.0])
    assert np.array_equal(regret, [0.5, -1.0])
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert concentration(np.array([4.0, 3.0, 2.0, 1.0]))["top_1"] == pytest.approx(0.4)


def test_bootstrap_and_temporal_quartiles_are_deterministic() -> None:
    values = np.array([0.0, 1.0, -1.0, 0.5])
    assert paired_bootstrap(values, "unit", draws=200) == paired_bootstrap(
        values, "unit", draws=200
    )
    boundaries, labels = temporal_quartiles(np.arange(8))
    assert boundaries == [1, 3, 5]
    assert labels.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_canonical_paths_are_not_writable_targets(tmp_path: Path) -> None:
    repo = tmp_path
    allowed = repo / "analysis/posthoc/result.json"
    validate_output_path(allowed, repo)
    with pytest.raises(PermissionError):
        validate_output_path(repo / "evidence/p2/stage_status.json", repo)
