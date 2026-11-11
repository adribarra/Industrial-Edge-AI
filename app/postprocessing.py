"""Helpers for choosing the PASS/FAIL threshold.

The threshold has to come from scores of good validation images only, never from the test
images.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def calibrate_threshold(normal_scores: Sequence[float], percentile: float = 100.0,
                        margin: float = 0.0) -> float:
    """Pick a threshold from the scores of good validation images.

    It takes the given percentile of the scores and multiplies it by (1 + margin).
    With percentile=100 the threshold is the highest good score, so no validation image
    is flagged. A lower percentile catches more defects but also flags some good parts.
    """
    s = np.asarray(list(normal_scores), dtype=np.float64)
    if s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("normal_scores must be a non-empty list of finite values")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    return float(np.percentile(s, percentile) * (1.0 + margin))
