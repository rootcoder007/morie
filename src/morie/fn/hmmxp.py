# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Max pooling over a sliding window."""

from . import _array_core as np

from ._richresult import RichResult
from .mxpol import maxpool_forward

__all__ = ["geron_max_pool"]


def geron_max_pool(x, window=2, stride=None):
    """
    Max pooling: output the maximum per pooling window.

    Formula: y[i,j,k] = max over window W of x[i+u, j+v, k]

    Pooling is DELEGATED per channel to the finished implementation
    :func:`morie.fn.mxpol.maxpool_forward`; this entry point adds the
    channel loop and the invariance diagnostic. Max pooling has no
    parameters and its output is unchanged by any shift of the input that
    keeps the maximum inside the same window -- a small translation
    invariance bought with a hard loss of position, which is why modern
    architectures replace it with strided convolutions when localisation
    matters.

    Parameters
    ----------
    x : array-like, shape (h, w) or (h, w, c)
        Feature map; channels last.
    window : int, default 2
        Square pooling window.
    stride : int, optional
        Defaults to ``window`` (non-overlapping).

    Returns
    -------
    result : RichResult
        Keys: pooled, argmax, output_shape, parameters, estimate, n,
        method.

    Examples
    --------
    >>> r = geron_max_pool(np.arange(16.0).reshape(4, 4), 2)
    >>> r["pooled"].tolist()
    [[5.0, 7.0], [13.0, 15.0]]
    >>> r["output_shape"], int(r["parameters"])
    ((2, 2), 0)

    Overlapping windows at stride 1 keep more of the map:

    >>> geron_max_pool(np.arange(9.0).reshape(3, 3), 2, stride=1)["pooled"].tolist()
    [[4.0, 5.0], [7.0, 8.0]]

    Two channels are pooled independently:

    >>> z = np.dstack([np.arange(4.0).reshape(2, 2), np.arange(4.0).reshape(2, 2) - 10.0])
    >>> geron_max_pool(z, 2)["pooled"].ravel().tolist()
    [3.0, -7.0]

    References
    ----------
    Geron Ch 12
    """
    a = np.asarray(x, dtype=float)
    if a.ndim not in (2, 3):
        raise ValueError(f"geron_max_pool: x must be (h, w) or (h, w, c), got ndim={a.ndim}")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_max_pool: x contains non-finite values")
    k = int(window)
    if k < 1:
        raise ValueError(f"geron_max_pool: window must be >= 1, got {window!r}")
    s = k if stride is None else int(stride)
    if s < 1:
        raise ValueError(f"geron_max_pool: stride must be >= 1, got {stride!r}")
    H, W = a.shape[0], a.shape[1]
    if k > H or k > W:
        raise ValueError(f"geron_max_pool: window {k} does not fit in a {H}x{W} map")

    if a.ndim == 2:
        base = maxpool_forward(a, kernel_size=k, stride=s)
        pooled = np.asarray(base["y"], dtype=float)
        arg = np.asarray(base["argmax"])
        shape = tuple(int(v) for v in base["output_shape"])
    else:
        outs, args = [], []
        for c in range(a.shape[2]):
            b = maxpool_forward(a[:, :, c], kernel_size=k, stride=s)
            outs.append(np.asarray(b["y"], dtype=float))
            args.append(np.asarray(b["argmax"]))
        pooled = np.dstack(outs)
        arg = np.dstack(args)
        shape = (pooled.shape[0], pooled.shape[1], pooled.shape[2])

    return RichResult(
        title="Max pooling",
        summary_lines=[("Window", k), ("Stride", s), ("Output shape", shape)],
        interpretation="No parameters, and invariant to shifts that keep the max in the same window.",
        payload={
            "pooled": pooled,
            "y": pooled,
            "argmax": arg,
            "output_shape": shape,
            "parameters": 0,
            "estimate": pooled,
            "n": int(a.size),
            "method": "Max pooling, per channel, delegated to morie.fn.mxpol.maxpool_forward",
        },
    )


def cheatsheet():
    return "hmmxp: Max pooling over a sliding window"


# compact alias per ledger/NAMING.md
geronmaxpool = geron_max_pool
