# morie.fn -- function file (rootcoder007/morie)
"""Matrix trace (sum of diagonal)."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def trace(A: Union[Sequence, np.ndarray]) -> float:
    """Trace of a square matrix: Σᵢ Aᵢᵢ."""
    M = np.asarray(A, dtype=float)
    if M.shape[0] != M.shape[1]:
        raise ValueError("matrix must be square.")
    return float(np.trace(M))
