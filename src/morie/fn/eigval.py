# morie.fn -- function file (rootcoder007/morie)
"""Eigenvalues of a symmetric matrix."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def eigval(A: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Eigenvalues of a symmetric matrix (ascending order)."""
    M = np.asarray(A, dtype=float)
    return np.linalg.eigvalsh(M)
