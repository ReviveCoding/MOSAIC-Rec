from __future__ import annotations

import pytest

from mosaic_rec.p2.seed_summary import distribution


def test_seed_distribution_uses_sample_sd_and_no_best_seed() -> None:
    result = distribution([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
    assert result["coefficient_of_variation"] == 0.5
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_zero_mean_cv_is_not_meaningful() -> None:
    result = distribution([0.0, 0.0, 0.0])
    assert result["coefficient_of_variation"] is None
    assert result["sample_sd"] == pytest.approx(0.0)
