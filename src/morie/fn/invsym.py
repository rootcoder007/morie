# morie.fn -- function file (rootcoder007/morie)
"""Inverse of a symmetric matrix (pinv-safe)."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def invsym(A: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Pseudo-inverse of a symmetric matrix.

    Robust to singularity; use np.linalg.pinv internally.
    """
    M = np.asarray(A, dtype=float)
    return np.linalg.pinv(M)
