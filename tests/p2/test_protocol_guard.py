from pathlib import Path

import pytest

from mosaic_rec.p2.protocol_guard import prequalification_guard


def test_clean_repository_prequalification_guard() -> None:
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="Protected access count"):
        prequalification_guard(repo)
