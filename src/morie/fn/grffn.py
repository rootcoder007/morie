# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transformer position-wise feed-forward network."""

from . import _array_core as np

from ._richresult import RichResult
from .grlinf import geron_linear_layer_forward

__all__ = ["geron_transformer_feedforward"]

_METHOD = "Transformer position-wise feed-forward sublayer"


def geron_transformer_feedforward(x, W1, b1, W2, b2):
    r"""The FFN sublayer of a transformer block.

    .. math::
        \mathrm{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2

    "Position-wise" means the same two matrices are applied to every
    token independently -- there is no mixing across positions here, all
    of that happened in the attention sublayer.  So a ``(T, d)`` input
    is just ``T`` independent forward passes, and this function checks
    that by construction: the result for row ``t`` never depends on the
    other rows.

    Both affine stages delegate to
    :func:`morie.fn.grlinf.geron_linear_layer_forward`.  That function
    stores weights ``(out, in)`` while the transformer convention writes
    ``x W_1``, i.e. ``(in, out)``, so ``W1`` and ``W2`` are transposed on
    the way in.

    Parameters
    ----------
    x : array-like, shape (d,) or (T, d)
    W1 : array-like, shape (d, d_ff)
    b1 : array-like, shape (d_ff,) or scalar
    W2 : array-like, shape (d_ff, d)
    b2 : array-like, shape (d,) or scalar

    Returns
    -------
    RichResult
        Payload keys ``output``, ``hidden`` (post-ReLU), ``d_model``,
        ``d_ff``, ``expansion_ratio``, ``sparsity`` (fraction of hidden
        units the ReLU zeroed), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Feed-Forward sublayer section.

    Examples
    --------
    Identity ``W1`` clips the negative coordinate, then ``W2`` reads the
    survivors (the sublayer is shape-preserving, as the residual
    connection around it requires):

    >>> W1 = [[1.0, 0.0], [0.0, 1.0]]
    >>> W2 = [[2.0, 0.0], [3.0, 0.0]]
    >>> r = geron_transformer_feedforward([1.0, -1.0], W1, [0.0, 0.0], W2, [1.0, 1.0])
    >>> r["hidden"]
    [1.0, 0.0]
    >>> r["output"]
    [3.0, 1.0]
    >>> r["sparsity"]
    0.5
    """
    x_arr = np.asarray(x, dtype=float)
    W1 = np.atleast_2d(np.asarray(W1, dtype=float))
    W2 = np.atleast_2d(np.asarray(W2, dtype=float))
    if W1.shape[1] != W2.shape[0]:
        raise ValueError(
            f"W1 maps to {W1.shape[1]} hidden units but W2 expects {W2.shape[0]}."
        )
    if W2.shape[1] != W1.shape[0]:
        raise ValueError(
            f"the sublayer must be shape-preserving: W1 takes d={W1.shape[0]} but "
            f"W2 returns d={W2.shape[1]}."
        )

    first = geron_linear_layer_forward(x_arr, W1.T, b1)
    H = np.maximum(np.asarray(first["output"], dtype=float), 0.0)
    second = geron_linear_layer_forward(H, W2.T, b2)
    Y = np.asarray(second["output"], dtype=float)

    d_model = int(W1.shape[0])
    d_ff = int(W1.shape[1])

    return RichResult(
        title="Transformer feed-forward sublayer",
        summary_lines=[("d_model", d_model), ("d_ff", d_ff),
                       ("Expansion", d_ff / d_model)],
        payload={
            "output": Y.tolist(),
            "hidden": H.tolist(),
            "d_model": d_model,
            "d_ff": d_ff,
            "expansion_ratio": float(d_ff) / float(d_model),
            "sparsity": float(np.mean(H == 0.0)),
            "n_parameters": first["n_parameters"] + second["n_parameters"],
            "estimate": Y.tolist(),
            "n": int(x_arr.shape[0]) if x_arr.ndim == 2 else 1,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grffn: FFN(x) = relu(x W1 + b1) W2 + b2, position-wise (delegates to grlinf)"
