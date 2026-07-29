# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feature map output of a conv layer after activation."""

import numpy as np

from ._richresult import RichResult
from .grcvf import geron_conv2d_forward

__all__ = ["geron_feature_map"]

_ACTS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "identity": lambda z: z,
    "tanh": np.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
}


def geron_feature_map(x, K, b=0.0, activation="relu", stride=1, padding=0):
    """
    Feature map output of a conv layer after activation.

    Formula: F = phi(conv(x, K) + b)

    The convolution and bias are DELEGATED to
    :func:`morie.fn.grcvf.geron_conv2d_forward` (cross-correlation, as in
    every DL library), so the shape arithmetic lives in one place. This
    module applies the activation and reports what the activation costs
    you: with ReLU, ``sparsity`` is the fraction of the map that has been
    zeroed, and any unit that is zero here contributes no gradient.

    Several filters can be stacked by passing ``K`` with shape
    ``(F, kh, kw)`` (or ``(F, C, kh, kw)`` for multi-channel input); the
    result is then a ``(F, oh, ow)`` stack.

    Parameters
    ----------
    x : array-like, shape (H, W) or (C, H, W)
        Input map.
    K : array-like
        One filter, or a stack of filters along the leading axis.
    b : float or array-like, default 0.0
        Bias; one value, or one per filter.
    activation : {"relu", "identity", "tanh", "sigmoid"}, default "relu"
    stride, padding : int or pair of int

    Returns
    -------
    result : RichResult
        Keys: feature_map, pre_activation, out_shape, n_filters,
        sparsity, max_response, argmax, estimate, n, method.

    Examples
    --------
    A diagonal 2x2 filter on a 3x3 ramp sums the two diagonal neighbours
    of each window; ReLU leaves all-positive values alone:

    >>> X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    >>> r = geron_feature_map(X, [[1.0, 0.0], [0.0, 1.0]])
    >>> r["feature_map"]
    [[6.0, 8.0], [12.0, 14.0]]
    >>> r["out_shape"]
    (2, 2)
    >>> r["sparsity"]
    0.0

    A negative bias pushes part of the map below zero, and ReLU clips it:

    >>> r2 = geron_feature_map(X, [[1.0, 0.0], [0.0, 1.0]], b=-9.0)
    >>> r2["feature_map"]
    [[0.0, 0.0], [3.0, 5.0]]
    >>> round(r2["sparsity"], 6)
    0.5

    Two filters give a two-slice stack:

    >>> r3 = geron_feature_map([[1.0, 2.0]], [[[1.0]], [[2.0]]])
    >>> r3["n_filters"], r3["out_shape"]
    (2, (2, 1, 2))
    >>> r3["feature_map"]
    [[[1.0, 2.0]], [[2.0, 4.0]]]

    References
    ----------
    Géron Ch 12
    """
    if activation not in _ACTS:
        raise ValueError(f"geron_feature_map: activation must be one of {sorted(_ACTS)}, got {activation!r}")
    Xa = np.asarray(x, dtype=float)
    Ka = np.asarray(K, dtype=float)
    in_ch = 1 if Xa.ndim == 2 else (Xa.shape[0] if Xa.ndim == 3 else None)
    if in_ch is None:
        raise ValueError(f"geron_feature_map: x must be 2-D or 3-D, got shape {Xa.shape}")

    if Ka.ndim == 2 or (Ka.ndim == 3 and in_ch > 1 and Ka.shape[0] == in_ch):
        filters = [Ka]
    elif Ka.ndim in (3, 4):
        filters = [Ka[i] for i in range(Ka.shape[0])]
    else:
        raise ValueError(f"geron_feature_map: K must be 2-D, 3-D or 4-D, got shape {Ka.shape}")

    F = len(filters)
    bias = np.atleast_1d(np.asarray(b, dtype=float))
    if bias.size == 1:
        bias = np.repeat(bias, F)
    if bias.size != F:
        raise ValueError(f"geron_feature_map: b has {bias.size} entries but there are {F} filters")

    maps = []
    for i, k in enumerate(filters):
        conv = geron_conv2d_forward(Xa, k, b=float(bias[i]), stride=stride, padding=padding)
        maps.append(np.asarray(conv["Y"], dtype=float))
    Z = np.stack(maps, axis=0)
    A = _ACTS[activation](Z)
    single = F == 1
    out = A[0] if single else A
    pre = Z[0] if single else Z
    shape = tuple(int(v) for v in out.shape)
    flat_arg = int(np.argmax(A))

    return RichResult(
        title="Feature map",
        summary_lines=[("Output shape", shape), ("Filters", F), ("Activation", activation)],
        interpretation="Units that ReLU has zeroed pass no gradient back to their filter.",
        payload={
            "feature_map": out.tolist(),
            "pre_activation": pre.tolist(),
            "out_shape": shape,
            "n_filters": int(F),
            "activation": activation,
            "sparsity": float(np.mean(A == 0.0)),
            "max_response": float(A.max()),
            "argmax": tuple(int(v) for v in np.unravel_index(flat_arg, A.shape)),
            "estimate": float(A.mean()),
            "n": int(A.size),
            "method": "F = phi(conv(x, K) + b); convolution delegated to grcvf",
        },
    )


def cheatsheet():
    return "hmfmap: Feature map output of a conv layer after activation"
