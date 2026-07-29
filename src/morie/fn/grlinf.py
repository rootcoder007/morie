# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Affine (linear) layer forward pass."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_linear_layer_forward"]

_METHOD = "Affine layer forward pass"


def geron_linear_layer_forward(X, W, b):
    r"""One dense layer, no activation.

    .. math::
        Y = X W^{\mathsf T} + b

    ``W`` is stored ``(out_features, in_features)`` -- the ``nn.Linear``
    layout, which is why the transpose appears.  Storing it this way
    means each *row* of ``W`` is one output unit's weight vector, so
    row-wise operations such as max-norm regularisation
    (:mod:`morie.fn.grmnr`) act on the right thing.

    Every dense stage in this tranche routes through here: ``grmlpf``
    (one call per hidden layer), ``grffn`` (both transformer
    projections), ``grmlr`` and ``grmlc`` (output heads).

    Parameters
    ----------
    X : array-like, shape (in,) or (m, in)
        One instance or a batch.
    W : array-like, shape (out, in)
    b : array-like, shape (out,) or scalar

    Returns
    -------
    RichResult
        Payload keys ``output`` (1-D for a single instance, else 2-D),
        ``preactivation_norm``, ``in_features``, ``out_features``,
        ``n_parameters``, ``estimate``, ``n``, ``method``.

    Examples
    --------
    Two inputs into three units, the third unit summing both and adding
    a bias of 1:

    >>> W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> r = geron_linear_layer_forward([1.0, 2.0], W, [0.0, 0.0, 1.0])
    >>> r["output"]
    [1.0, 2.0, 4.0]
    >>> r["n_parameters"]
    9

    A batch keeps its rows:

    >>> geron_linear_layer_forward([[1.0, 0.0], [0.0, 1.0]], W, 0.0)["output"]
    [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    """
    X = np.asarray(X, dtype=float)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if X.ndim not in (1, 2):
        raise ValueError(f"X must be 1-D (one instance) or 2-D (a batch), got ndim {X.ndim}.")
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D of shape (out, in), got shape {W.shape}.")
    out_f, in_f = W.shape
    batch = X.ndim == 2
    Xm = X if batch else X.reshape(1, -1)
    if Xm.shape[1] != in_f:
        raise ValueError(
            f"X has {Xm.shape[1]} features but W expects {in_f} "
            f"(W is stored as (out, in) = {W.shape})."
        )
    b_arr = np.asarray(b, dtype=float).ravel()
    if b_arr.size == 1:
        b_arr = np.full(out_f, float(b_arr[0]))
    if b_arr.size != out_f:
        raise ValueError(f"b has {b_arr.size} entries but the layer has {out_f} outputs.")
    if not (np.all(np.isfinite(Xm)) and np.all(np.isfinite(W)) and np.all(np.isfinite(b_arr))):
        raise ValueError("X, W and b must all be finite.")

    Y = Xm @ W.T + b_arr

    return RichResult(
        title="Linear layer forward",
        summary_lines=[("in -> out", f"{in_f} -> {out_f}"), ("Batch", int(Xm.shape[0]))],
        payload={
            "output": Y.tolist() if batch else Y[0].tolist(),
            "preactivation_norm": float(np.linalg.norm(Y)),
            "in_features": int(in_f),
            "out_features": int(out_f),
            "n_parameters": int(W.size + out_f),
            "estimate": Y.tolist() if batch else Y[0].tolist(),
            "n": int(Xm.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlinf: Y = X W^T + b, W stored (out, in) as in nn.Linear"
