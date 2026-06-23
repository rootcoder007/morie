# morie.fn -- function file (rootcoder007/morie)
"""Cross product of centered vectors."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def crsspr(x: Union[Sequence, np.ndarray], y: Union[Sequence, np.ndarray]) -> float:
    """Σᵢ (xᵢ − x̄)(yᵢ − ȳ) -- cross-product of centered series.

    Numerator of covariance and Pearson r.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sum((a - a.mean()) * (b - b.mean())))
