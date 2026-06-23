# morie.fn -- function file (rootcoder007/morie)
"""Lag operator (k-th lag)."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def lagop(x: Union[Sequence, np.ndarray], k: int = 1) -> np.ndarray:
    """k-th lag of a series. First k entries are NaN (no prior data).

    Lᵏ xₜ = xₜ₋ₖ
    """
    if k < 0:
        raise ValueError("k must be non-negative.")
    a = np.asarray(x, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    if k < a.size:
        out[k:] = a[: a.size - k]
    return out
