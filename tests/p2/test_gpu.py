from __future__ import annotations

from unittest.mock import patch

import pytest

from mosaic_rec.p2.gpu import gpu_activity, sustained_idle


def test_gpu_activity_parses_single_gpu() -> None:
    with patch("subprocess.check_output", return_value="0, 0\n"):
        assert gpu_activity() == (0, 0)


def test_idle_fails_closed_when_wsl_hides_busy_process() -> None:
    with (
        patch("mosaic_rec.p2.gpu.compute_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.gpu_activity", return_value=(4444, 20)),
        pytest.raises(RuntimeError, match="materially active"),
    ):
        sustained_idle(seconds=1, interval=1)


def test_idle_accepts_sustained_zero_activity() -> None:
    with (
        patch("mosaic_rec.p2.gpu.compute_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.gpu_activity", return_value=(0, 0)),
        patch("mosaic_rec.p2.gpu.time.sleep"),
    ):
        sustained_idle(seconds=5, interval=5)
