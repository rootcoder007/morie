# morie.fn -- slice s04 (rootcoder007/morie)
"""Max pooling operation for CNNs.

Source consulted: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, Section 13.4.  "The max pooling operation
summarizes the input as the maximum within a rectangular neighborhood",
the window is slid by the stride, and the output length in the l-th
layer is

    L^(l+1) = (L^(l) - P^(l)) / S^(l) + 1.

The worked example of Fig. 13.6 is quoted in the text: with a 2-by-2
filter and stride 1 the first output is 7 -- "the maximum value of the
four elements that conform to the bounds of the filter (3, 3, 7, 4)" --
the second is 5, "the max of 3, 4, 4, and 5", and the last is 6, "a max
of 3, 2, 5, and 6".  Those three printed windows are the anchor.

This is the one-dimensional case, y[i] = max(x[i*S : i*S+P]); the book's
two-dimensional map is the same operation applied along both axes.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["max_pooling"]


def max_pooling(x, kernel, stride):
    """One-dimensional max pooling.

    Parameters
    ----------
    x : array-like
        The activation map to pool.
    kernel : int
        Window width P; must be a positive integer no larger than len(x).
    stride : int
        Step S; must be a positive integer.

    Returns
    -------
    estimate : the first pooled value
    pooled   : the pooled map, of length (n - P) / S + 1 rounded down
    argmax   : the index in x that supplied each pooled value
    """
    v = k.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("max_pooling: x is empty")
    P = int(kernel)
    S = int(stride)
    if P <= 0:
        raise ValueError("max_pooling: kernel must be a positive integer")
    if S <= 0:
        raise ValueError("max_pooling: stride must be a positive integer")
    if P > n:
        raise ValueError("max_pooling: kernel is wider than the input")
    m = (n - P) // S + 1
    pooled = []
    where = []
    for i in range(m):
        a = i * S
        best = a
        for j in range(a + 1, a + P):
            if v[j] > v[best]:
                best = j
        pooled.append(v[best])
        where.append(best)
    return RichResult(
        title="Max pooling",
        summary_lines=[("input", n), ("kernel", P), ("stride", S), ("output", m)],
        payload={
            "estimate": pooled[0],
            "pooled": pooled,
            "argmax": where,
            "n": m,
            "method": "y[i] = max(x[i*S : i*S+P]), Chapter 13 Sect. 13.4 with L' = (L-P)/S + 1",
        },
    )


def cheatsheet():
    return "maxpl: Max pooling operation for CNNs"


# compact alias per ledger/NAMING.md
maxpooling = max_pooling
