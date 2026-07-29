# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D max pooling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_max_pooling"]

_METHOD = "2D max pooling"


def geron_max_pooling(X, k=2, stride=None):
    r"""Take the maximum of each ``k x k`` window.

    .. math::
        Y[i, j] = \max_{0 \le p, q < k}
        X[i s + p,\; j s + q]

    Output size is :math:`\lfloor (H-k)/s \rfloor + 1`; partial windows
    at the edge are dropped, never padded with zeros, because a zero
    would be a real (and wrong) candidate for the maximum.

    Max rather than mean (:mod:`morie.fn.grapl`) buys a small amount of
    translation invariance: shift the input by less than the stride and
    the strongest activation in each window usually stays the strongest,
    so the output is unchanged.  The cost is that everything but the
    winner is discarded -- ``argmax_indices`` records which position won
    each window, which is exactly what a pooling layer must remember to
    route gradients back.

    Parameters
    ----------
    X : array-like, shape (H, W)
    k : int, optional
        Window size, default 2.
    stride : int, optional
        Step, defaults to ``k`` (non-overlapping windows).

    Returns
    -------
    RichResult
        Payload keys ``output``, ``output_shape``, ``argmax_indices``
        (flat index within each window), ``reduction_factor``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 12, Max Pooling section.

    Examples
    --------
    A 2x2 window over a 2x2 input keeps only the largest value:

    >>> r = geron_max_pooling([[1.0, 2.0], [3.0, 4.0]], k=2)
    >>> r["output"]
    [[4.0]]
    >>> r["reduction_factor"]
    4.0

    Overlapping windows (stride 1) on a 3x3 input give a 2x2 map:

    >>> X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    >>> geron_max_pooling(X, k=2, stride=1)["output"]
    [[5.0, 5.0], [9.0, 4.0]]

    Negative inputs stay negative -- no zero padding sneaks in:

    >>> geron_max_pooling([[-3.0, -1.0], [-9.0, -4.0]], k=2)["output"]
    [[-1.0]]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (H, W), got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    s = k if stride is None else int(stride)
    if s < 1:
        raise ValueError(f"stride must be a positive integer, got {s}.")
    H, W = A.shape
    if k > H or k > W:
        raise ValueError(f"window {k}x{k} does not fit in a {H}x{W} input.")

    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1
    Y = np.empty((out_h, out_w))
    arg = np.empty((out_h, out_w), dtype=int)
    for i in range(out_h):
        for j in range(out_w):
            win = A[i * s:i * s + k, j * s:j * s + k]
            Y[i, j] = win.max()
            arg[i, j] = int(win.argmax())

    return RichResult(
        title="Max pooling",
        summary_lines=[("Input", (H, W)), ("Output", (out_h, out_w)),
                       ("k / stride", f"{k} / {s}")],
        payload={
            "output": Y.tolist(),
            "output_shape": (int(out_h), int(out_w)),
            "argmax_indices": arg.tolist(),
            "reduction_factor": float(A.size) / float(Y.size),
            "k": k,
            "stride": s,
            "estimate": Y.tolist(),
            "n": int(Y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmpl: Y[i,j] = max of the kxk window at stride s; argmax kept for the backward pass"
