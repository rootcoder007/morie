# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.8: the KronA layer output."""

from . import _array_core as np

from ._richresult import RichResult
from .km062 import _tuned

__all__ = ["kamath_ch4_krona_output"]


def kamath_ch4_krona_output(X, W, A_k, B_k, s):
    """Y = X W + s X [A_k (x) B_k] = X (W + s [A_k (x) B_k]).

    Distributivity means Eq 4.8 IS Eq 4.9's merged weight applied to X,
    so the merge is delegated to km062 and multiplied through -- one
    definition, two views, no drift. Both routes are returned so the
    caller can see they agree.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.8, printed
    p. 153.

    Examples
    --------
    >>> out = kamath_ch4_krona_output([[1.0, 0.0]],
    ...     [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...     [[1.0]], 2.0)
    >>> out["Y"]
    [[3.0, 1.0]]
    >>> out["base"], out["adapter_term"]
    ([[1.0, 1.0]], [[2.0, 0.0]])
    """
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    Wt, Wm, K, s = _tuned(W, A_k, B_k, s)
    if Xm.shape[1] != Wm.shape[0]:
        raise ValueError(
            f"X has width {Xm.shape[1]} but W has {Wm.shape[0]} rows.")
    Y = Xm @ Wt
    return RichResult(payload={
        "Y": [[float(v) for v in row] for row in Y],
        "base": [[float(v) for v in row] for row in Xm @ Wm],
        "adapter_term": [[float(v) for v in row] for row in s * (Xm @ K)],
        "s": s, "estimate": float(Y[0, 0]), "n": int(Xm.shape[0]),
        "method": "KronA layer output (Kamath Eq 4.8)"})


def cheatsheet():
    return "km061: Y = X W + s X (A_k (x) B_k) via the merged weight"
