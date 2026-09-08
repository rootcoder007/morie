# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-attention: Q, K and V all projected from the same sequence."""

from . import _array_core as np

from ._richresult import RichResult
from .grsdpa import attend

__all__ = ["geron_self_attention"]

_METHOD = "Self-attention (single head)"


def geron_self_attention(X, WQ, WK, WV, mask=None):
    r"""Project one sequence three ways, then attend to itself.

    .. math::
        \mathrm{SA}(X) = \mathrm{softmax}\!\left(
            \frac{(X W_Q)(X W_K)^{\top}}{\sqrt{d_k}}\right) (X W_V)

    The attention kernel itself is delegated to :mod:`morie.fn.grsdpa`
    (``attend``) -- this module only supplies the three projections and
    the "same sequence on both sides" contract that makes it *self*
    attention.

    Parameters
    ----------
    X : array-like, shape (T, d_model)
    WQ, WK : array-like, shape (d_model, d_k)
    WV : array-like, shape (d_model, d_v)
    mask : array-like of bool, shape (T, T), optional
        Pass a lower-triangular mask for causal self-attention.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``weights``, ``Q``, ``K``, ``V``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Self-Attention section.

    Examples
    --------
    Identity projections on an orthonormal 2-token sequence reduce this
    to plain scaled dot-product attention:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_self_attention(I, I, I, I)
    >>> round(r["weights"][0][0], 6)
    0.669762

    A causal mask stops token 0 from seeing token 1:

    >>> r2 = geron_self_attention(I, I, I, I, mask=[[True, False], [True, True]])
    >>> r2["weights"][0]
    [1.0, 0.0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"X must be a non-empty (T, d_model) matrix, got shape {X.shape}.")
    mats = {}
    for name, W in (("WQ", WQ), ("WK", WK), ("WV", WV)):
        A = np.atleast_2d(np.asarray(W, dtype=float))
        if A.ndim != 2 or A.shape[0] != X.shape[1]:
            raise ValueError(
                f"{name} must have {X.shape[1]} rows to match d_model, got shape {A.shape}."
            )
        mats[name] = A
    if mats["WQ"].shape[1] != mats["WK"].shape[1]:
        raise ValueError(
            f"WQ maps to d_k={mats['WQ'].shape[1]} but WK maps to {mats['WK'].shape[1]}."
        )

    Q = X @ mats["WQ"]
    K = X @ mats["WK"]
    V = X @ mats["WV"]
    out, W = attend(Q, K, V, mask)

    return RichResult(
        title="Self-attention",
        summary_lines=[("Tokens", int(X.shape[0])), ("d_k", int(Q.shape[1]))],
        payload={
            "output": out.tolist(),
            "weights": W.tolist(),
            "Q": Q.tolist(),
            "K": K.tolist(),
            "V": V.tolist(),
            "estimate": out.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsa: SA(X) = softmax(XWq (XWk)^T/sqrt(d_k)) XWv; kernel delegated to grsdpa.attend"
