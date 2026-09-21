"""Validated runtime override for predeclared development HPO trials."""

from __future__ import annotations

import math
import os


def learning_rate(default: float) -> float:
    raw = os.environ.get("MOSAIC_HPO_LEARNING_RATE")
    if raw is None:
        return default
    value = float(raw)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("Invalid MOSAIC_HPO_LEARNING_RATE")
    return value
