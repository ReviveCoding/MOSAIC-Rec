from __future__ import annotations

import numpy as np

from mosaic_rec.p2.classic import _exact_target_ranks, _model_scores


def test_unknown_classic_model_is_rejected(tmp_path) -> None:
    try:
        _model_scores("UNKNOWN", ["i1"], tmp_path / "missing", {"i1": 0})
    except ValueError as exc:
        assert str(exc) == "UNKNOWN"
    else:
        raise AssertionError("unknown model was accepted")


def test_popularity_tie_order_is_item_id_ascending() -> None:
    item_ids = ["i2", "i1", "i3"]
    scores = np.asarray([2, 2, 1])
    order = sorted(range(3), key=lambda i: (-int(scores[i]), item_ids[i]))
    assert order == [1, 0, 2]


def test_exact_target_ranks_respect_time_and_history_masks() -> None:
    scores = np.asarray([2, 1, 0, 0])
    order = np.asarray([0, 1, 2, 3])
    position = np.asarray([0, 1, 2, 3])
    first_seen = np.asarray([1, 1, 1, 10])
    ranks = _exact_target_ranks(
        scores=scores,
        order=order,
        order_position=position,
        first_seen=first_seen,
        query_timestamps=[5, 10],
        targets=[2, 3],
        histories=[{0}, {1}],
    )
    assert ranks == [2, 3]
