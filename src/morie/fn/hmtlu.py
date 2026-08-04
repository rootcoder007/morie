# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Threshold logic unit: step activation of weighted sum."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_tlu"]


def geron_tlu(x, w, b=0.0):
    """
    Threshold logic unit: step activation of weighted sum.

    Formula: y = step(w^T x + b)

    McCulloch-Pitts / perceptron unit. The Heaviside step fires on
    ``z >= 0`` (``step(0) = 1``), which is the convention that makes a
    single TLU with ``w = (1, 1), b = -1.5`` compute logical AND.

    Parameters
    ----------
    x : array-like
        One input vector of length d, or a (n, d) batch.
    w : array-like
        Weight vector of length d.
    b : float, default 0.0
        Bias (negative threshold).

    Returns
    -------
    result : RichResult
        Keys: y, z, estimate, n, method.

    Examples
    --------
    AND gate: fires only when both inputs are 1.

    >>> r = geron_tlu([[0, 0], [0, 1], [1, 0], [1, 1]], [1.0, 1.0], -1.5)
    >>> [int(v) for v in r["y"]]
    [0, 0, 0, 1]
    >>> [float(v) for v in r["z"]]
    [-1.5, -0.5, -0.5, 0.5]
    >>> int(geron_tlu([1.0], [1.0], 0.0)["y"][0])
    1

    References
    ----------
    Géron Ch 9
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("geron_tlu: x must be a non-empty vector or (n, d) batch")
    wv = np.asarray(w, dtype=float).ravel()
    if wv.size != X.shape[1]:
        raise ValueError(f"geron_tlu: x has {X.shape[1]} features but w has {wv.size} weights")
    bias = float(b)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(wv)) and np.isfinite(bias)):
        raise ValueError("geron_tlu: x, w and b must all be finite")

    z = X @ wv + bias
    y = (z >= 0.0).astype(int)

    return RichResult(
        title="Threshold logic unit",
        summary_lines=[("Inputs", int(X.shape[0])), ("Fired", int(np.sum(y)))],
        interpretation="A TLU is a linear separator: the decision boundary is the hyperplane w^T x + b = 0.",
        payload={
            "y": y,
            "z": z,
            "w": wv,
            "b": bias,
            "estimate": float(np.mean(y)),
            "n": int(X.shape[0]),
            "method": "Threshold logic unit with Heaviside step (step(0) = 1)",
        },
    )


def cheatsheet():
    return "hmtlu: Threshold logic unit: step activation of weighted sum"


# compact alias per ledger/NAMING.md
gerontlu = geron_tlu
