# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Threshold Logic Unit: weighted sum with a step activation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_threshold_logic_unit"]

_METHOD = "Threshold logic unit (Heaviside step)"


def geron_threshold_logic_unit(x, w, b=0.0):
    r"""Rosenblatt's perceptron unit.

    .. math::
        h(\mathbf{x}) = \mathrm{heaviside}(\mathbf{w}^{\top}\mathbf{x} + b),
        \qquad \mathrm{heaviside}(z) = \begin{cases}
            1 & z \ge 0\\ 0 & z < 0\end{cases}

    The step is why a TLU cannot be trained by gradient descent: its
    derivative is zero everywhere it is defined.  It is also why a single
    TLU can only carve one hyperplane -- the XOR failure that stalled
    neural networks for a decade.  Both facts are visible in the payload:
    ``margin`` is the pre-activation, and it moves smoothly while the
    output does not.

    Parameters
    ----------
    x : array-like, shape (n,) or (m, n)
    w : array-like, shape (n,)
    b : float, optional
        Bias; the threshold is at ``-b``.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``margin``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 9, Threshold Logic Unit section.

    Examples
    --------
    An AND gate: fires only when both inputs are 1.

    >>> X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> r = geron_threshold_logic_unit(X, [1.0, 1.0], b=-1.5)
    >>> r["output"]
    [0, 0, 0, 1]

    Exactly on the boundary counts as firing:

    >>> geron_threshold_logic_unit([1.0, 0.5], [1.0, 1.0], b=-1.5)["output"]
    1
    """
    A = np.asarray(x, dtype=float)
    wv = np.asarray(w, dtype=float).ravel()
    single = A.ndim == 1
    A = np.atleast_2d(A)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"x must be a non-empty (n,) or (m, n) array, got shape {A.shape}.")
    if wv.size != A.shape[1]:
        raise ValueError(f"w has {wv.size} weights but x has {A.shape[1]} features.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(wv)):
        raise ValueError("x and w must be finite.")
    b = float(b)
    if not np.isfinite(b):
        raise ValueError(f"b must be finite, got {b}.")

    z = A @ wv + b
    out = (z >= 0).astype(int)
    est = int(out[0]) if single else out.tolist()
    return RichResult(
        title="Threshold logic unit",
        summary_lines=[("Inputs", int(A.shape[0])), ("Firing rate", float(out.mean()))],
        payload={
            "output": est,
            "margin": float(z[0]) if single else z.tolist(),
            "estimate": est,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtlu: h(x) = 1 iff w.x + b >= 0; step has zero gradient, one hyperplane only (no XOR)"
