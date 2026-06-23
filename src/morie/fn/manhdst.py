# morie.fn -- function file (rootcoder007/morie)
"""Manhattan (L₁) distance."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def manhdst(x: Union[Sequence, np.ndarray], y: Union[Sequence, np.ndarray]) -> float:
    """Manhattan (L₁) distance: Σ |xᵢ − yᵢ|."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sum(np.abs(a - b)))
