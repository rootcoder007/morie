# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled dot-product attention (Vaswani et al. 2017, Eq 1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["scaled_dot_product_attention"]


def _softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    ``mask`` is additive (0 keep, -inf drop) or boolean (True keep).
    Rows of the attention matrix sum to 1 by construction, and the
    tests assert it rather than assume it.

    References: Vaswani et al. (2017), *Attention Is All You Need*,
    Section 3.2.1.

    Examples
    --------
    >>> out = scaled_dot_product_attention([[1.0, 0.0]], [[1.0, 0.0],
    ...     [0.0, 1.0]], [[1.0], [0.0]])
    >>> round(out["output"][0][0], 6)
    0.669762
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Q and K must share d_k; got {Q.shape[1]} and {K.shape[1]}.")
    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"K and V must have the same number of rows; got {K.shape[0]} "
            f"and {V.shape[0]}.")
    dk = Q.shape[1]
    scores = Q @ K.T / np.sqrt(dk)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != scores.shape:
            raise ValueError(
                f"mask shape {m.shape} does not match scores "
                f"{scores.shape}.")
        if m.dtype == bool:
            scores = np.where(m, scores, -np.inf)
        else:
            scores = scores + m
    A = _softmax_rows(scores)
    out = A @ V
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "attention": [[float(v) for v in row] for row in A],
        "estimate": float(out[0, 0]), "d_k": dk, "n": Q.shape[0],
        "method": "Scaled dot-product attention (Vaswani et al. 2017)"})


def cheatsheet():
    return "attsdp: softmax(QK^T/sqrt(d_k))V with additive or boolean mask"
