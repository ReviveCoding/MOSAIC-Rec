from __future__ import annotations

from unittest.mock import patch

import pytest

from mosaic_rec.p2.gpu import (
    compute_engine_pids,
    display_is_active,
    gpu_activity,
    sustained_idle,
)


def test_gpu_activity_parses_single_gpu() -> None:
    with patch("subprocess.check_output", return_value="0, 0\n"):
        assert gpu_activity() == (0, 0)


def test_display_active_parses_enabled() -> None:
    with patch("subprocess.check_output", return_value="Enabled\n"):
        assert display_is_active()


def test_compute_engine_pids_parses_compute_rows() -> None:
    output = """# gpu pid type sm mem enc dec jpg ofa command
0 1234 C 90 20 - - - - python
0 5678 G 10 5 - - - - desktop
"""
    with patch("subprocess.check_output", return_value=output):
        assert compute_engine_pids() == ["1234"]


def test_idle_fails_closed_when_wsl_hides_busy_process() -> None:
    with (
        patch("mosaic_rec.p2.gpu.compute_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.compute_engine_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.gpu_activity", return_value=(4444, 20)),
        patch("mosaic_rec.p2.gpu.display_is_active", return_value=False),
        pytest.raises(RuntimeError, match="materially active"),
    ):
        sustained_idle(seconds=1, interval=1)


def test_idle_accepts_sustained_zero_activity() -> None:
    with (
        patch("mosaic_rec.p2.gpu.compute_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.compute_engine_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.gpu_activity", return_value=(0, 0)),
        patch("mosaic_rec.p2.gpu.display_is_active", return_value=False),
        patch("mosaic_rec.p2.gpu.time.sleep"),
    ):
        sustained_idle(seconds=5, interval=5)


def test_idle_accepts_display_activity_without_compute_process() -> None:
    with (
        patch("mosaic_rec.p2.gpu.compute_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.compute_engine_pids", return_value=[]),
        patch("mosaic_rec.p2.gpu.gpu_activity", return_value=(2048, 40)),
        patch("mosaic_rec.p2.gpu.display_is_active", return_value=True),
        patch("mosaic_rec.p2.gpu.time.sleep"),
    ):
        sustained_idle(seconds=5, interval=5)
