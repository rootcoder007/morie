# morie.fn -- function file (rootcoder007/morie)
"""Cosine similarity."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def cossim(x: Union[Sequence, np.ndarray], y: Union[Sequence, np.ndarray]) -> float:
    """Cosine similarity: x·y / (||x|| ||y||)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero-norm vector.")
    return float(np.dot(a, b) / (na * nb))
