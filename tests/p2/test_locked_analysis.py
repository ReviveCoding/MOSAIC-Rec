from __future__ import annotations

import numpy as np

from mosaic_rec.p2.locked_analysis import item_cluster_bootstrap, percentile_bootstrap


def test_paired_bootstrap_is_deterministic() -> None:
    values = np.array([0.1, -0.2, 0.3, 0.0])
    first = percentile_bootstrap(values, draws=200, seed=7)
    assert first == percentile_bootstrap(values, draws=200, seed=7)
    assert first["mean"] == np.mean(values)


def test_item_cluster_bootstrap_keeps_cluster_rows_together() -> None:
    values = np.array([1.0, 1.0, -1.0])
    result = item_cluster_bootstrap(
        values, np.array(["a", "a", "b"]), draws=200, seed=11
    )
    assert result["clusters"] == 2
    assert result["mean"] == np.mean(values)
