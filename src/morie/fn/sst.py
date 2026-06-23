# morie.fn -- function file (rootcoder007/morie)
"""Total sum of squares."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def sst(y: Union[Sequence[float], np.ndarray]) -> float:
    """Total sum of squares: Σᵢ (yᵢ − ȳ)².

    Decomposition: SST = SSR + SSE. Used by R², ANOVA F-stat, etc.
    """
    a = np.asarray(y, dtype=float)
    return float(np.sum((a - a.mean()) ** 2))
