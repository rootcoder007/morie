# morie.fn -- function file (rootcoder007/morie)
"""Softmax over a vector."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def softmx(x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Softmax: σᵢ(x) = exp(xᵢ) / Σⱼ exp(xⱼ).

    Numerically stable -- subtracts max(x) before exponentiating.
    """
    a = np.asarray(x, dtype=float)
    e = np.exp(a - a.max())
    return e / e.sum()
