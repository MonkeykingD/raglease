"""Exact scalar linear percentiles for short, positive float64 histories."""
from collections import deque
import math

import numpy as np


def history_percentile(values, percentile):
    # All other inputs keep NumPy's dtype, shape, nonfinite and error semantics.
    # Positive doubles avoid signed-zero ordering and infinite interpolation.
    if (type(values) not in (list, tuple, deque) or not 0 < len(values) <= 256
            or type(percentile) not in (int, float, np.float64)
            or not 0 <= percentile <= 100):
        return np.percentile(values, percentile)
    if any(type(v) not in (float, np.float64) or not math.isfinite(v) or v <= 0
           for v in values):
        return np.percentile(values, percentile)
    ordered = sorted(values)
    position = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    a, b = float(ordered[lower]), float(ordered[upper])
    weight = position - lower
    difference = b - a
    # Preserve NumPy's two-sided linear interpolation evaluation order.
    if weight >= 0.5:
        return b - difference * (1.0 - weight)
    return a + difference * weight
