# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled dot-product attention."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_scaled_dot_product_attention", "attend"]

_METHOD = "Scaled dot-product attention"


def attend(Q, K, V, mask=None):
    """Core attention kernel: returns ``(output, weights)``.

    Shared by every attention-based module in this tranche (self-attention,
    encoder/decoder blocks, Swin, PVT, Perceiver) so the scaling and the
    masking convention live in exactly one place.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K and V must each be 2-D (seq_len, dim).")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(f"Q has d_k={Q.shape[1]} but K has d_k={K.shape[1]}.")
    if K.shape[0] != V.shape[0]:
        raise ValueError(f"K has {K.shape[0]} rows but V has {V.shape[0]}.")
    for name, A in (("Q", Q), ("K", K), ("V", V)):
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} contains non-finite values.")
    d_k = K.shape[1]
    if d_k == 0:
        raise ValueError("d_k is zero; the 1/sqrt(d_k) scaling is undefined.")

    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != scores.shape:
            raise ValueError(f"mask shape {m.shape} != score shape {scores.shape}.")
        keep = m.astype(bool) if m.dtype != bool else m
        if not keep.any(axis=1).all():
            raise ValueError("mask blocks every key for at least one query; softmax undefined.")
        scores = np.where(keep, scores, -np.inf)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(shifted)
    W = e / e.sum(axis=1, keepdims=True)
    return W @ V, W


def geron_scaled_dot_product_attention(Q, K, V, mask=None):
    r"""Attention as a softmax-weighted average of value rows.

    .. math::
        \mathrm{Attn}(Q, K, V)
          = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V

    The :math:`1/\sqrt{d_k}` is not cosmetic: for unit-variance
    independent entries the dot product of two length-:math:`d_k` vectors
    has variance :math:`d_k`, so without the scaling the softmax saturates
    as the head width grows and the gradient dies.

    Parameters
    ----------
    Q : array-like, shape (n_q, d_k)
    K : array-like, shape (n_k, d_k)
    V : array-like, shape (n_k, d_v)
    mask : array-like of bool, shape (n_q, n_k), optional
        True = attend, False = block. A row of all-False raises.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``weights``, ``scores``, ``d_k``,
        ``estimate`` (output), ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Scaled Dot-Product Attention.

    Examples
    --------
    Orthonormal queries/keys, ``d_k = 2``: the diagonal score is
    ``1/sqrt(2) = 0.7071`` and the off-diagonal is 0, so the first row
    weights are ``e^0.7071 / (e^0.7071 + 1)``:

    >>> r = geron_scaled_dot_product_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...                                        [[1.0, 0.0], [0.0, 1.0]])
    >>> round(r["weights"][0][0], 6)
    0.669762
    >>> round(r["output"][0][0], 6)
    0.669762

    Masking the second key forces all mass onto the first:

    >>> r2 = geron_scaled_dot_product_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...                                         [[1.0, 0.0], [0.0, 1.0]],
    ...                                         mask=[[True, False]])
    >>> r2["weights"][0]
    [1.0, 0.0]
    """
    out, W = attend(Q, K, V, mask)
    Kd = np.atleast_2d(np.asarray(K, dtype=float))
    scores = np.atleast_2d(np.asarray(Q, dtype=float)) @ Kd.T / np.sqrt(Kd.shape[1])
    return RichResult(
        title="Scaled dot-product attention",
        summary_lines=[("Queries", int(out.shape[0])), ("d_k", int(Kd.shape[1]))],
        payload={
            "output": out.tolist(),
            "weights": W.tolist(),
            "scores": scores.tolist(),
            "d_k": int(Kd.shape[1]),
            "estimate": out.tolist(),
            "n": int(out.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsdpa: Attn(Q,K,V) = softmax(QK^T/sqrt(d_k)) V; mask True=attend; attend() is the shared kernel"
