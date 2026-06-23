# morie.fn -- function file (rootcoder007/morie)
"""Leave-one-out jackknife estimates."""

from collections.abc import Callable, Sequence
from typing import Union

import numpy as np


def jackone(x: Union[Sequence, np.ndarray], stat: Callable) -> np.ndarray:
    """Leave-one-out jackknife: stat applied to each n−1-subset.

    Returns array of n estimates. Use to compute jackknife SE.
    """
    a = np.asarray(x)
    n = a.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = stat(np.delete(a, i))
    return out
