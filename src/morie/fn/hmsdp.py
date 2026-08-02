# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled dot-product attention."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_scaled_dot_product"]


def geron_scaled_dot_product(Q, K, V, d_k=None, mask=None):
    """
    Scaled dot-product attention.

    Formula: Att(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    The row-wise softmax is delegated to
    :func:`morie.fn.hmsftm.geron_softmax_function` (max-shift stabilised)
    rather than reimplemented. Masking is applied to the *scores*, before
    the softmax, by setting blocked entries to -inf; a query row with no
    visible key is an error, not a NaN row.

    Parameters
    ----------
    Q : array-like
        Queries, shape (T_q, d_k). A 1-D query is read as one row.
    K : array-like
        Keys, shape (T_k, d_k).
    V : array-like
        Values, shape (T_k, d_v).
    d_k : int, optional
        Scaling dimension; defaults to the key width ``K.shape[1]``.
    mask : array-like of bool/0-1, optional
        Shape (T_q, T_k) or (T_k,). True/1 means "visible"; False/0
        blocks that key. A lower-triangular mask gives causal attention.

    Returns
    -------
    result : RichResult
        Keys: Y, attention, scores, estimate, n, method.

    Examples
    --------
    A zero query scores every key equally, so attention is uniform and the
    output is the mean of the values:

    >>> r = geron_scaled_dot_product([[0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 10.0]])
    >>> [round(float(v), 12) for v in r["attention"][0]]
    [0.5, 0.5]
    >>> [round(float(v), 12) for v in r["Y"][0]]
    [0.5, 5.0]

    A causal mask hides the future, so the first row attends only to itself:

    >>> Qm = [[1.0, 0.0], [0.0, 1.0]]
    >>> rm = geron_scaled_dot_product(Qm, Qm, [[1.0], [2.0]], mask=[[1, 0], [1, 1]])
    >>> [round(float(v), 12) for v in rm["attention"][0]]
    [1.0, 0.0]
    >>> float(rm["Y"][0][0])
    1.0

    References
    ----------
    Géron Ch 15
    """
    Qa = np.asarray(Q, dtype=float)
    Ka = np.asarray(K, dtype=float)
    Va = np.asarray(V, dtype=float)
    if Qa.ndim == 1:
        Qa = Qa.reshape(1, -1)
    if Ka.ndim == 1:
        Ka = Ka.reshape(1, -1)
    if Va.ndim == 1:
        Va = Va.reshape(-1, 1)
    for nm, A in (("Q", Qa), ("K", Ka), ("V", Va)):
        if A.ndim != 2 or A.size == 0:
            raise ValueError(f"geron_scaled_dot_product: {nm} must be a non-empty 2-D matrix")
        if not np.all(np.isfinite(A)):
            raise ValueError(f"geron_scaled_dot_product: {nm} contains non-finite values")
    if Qa.shape[1] != Ka.shape[1]:
        raise ValueError(
            f"geron_scaled_dot_product: Q has width {Qa.shape[1]} but K has width {Ka.shape[1]}; "
            "queries and keys must share a dimension"
        )
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(
            f"geron_scaled_dot_product: K has {Ka.shape[0]} rows but V has {Va.shape[0]}; "
            "each key needs exactly one value"
        )
    dk = int(Ka.shape[1]) if d_k is None else int(d_k)
    if dk < 1:
        raise ValueError(f"geron_scaled_dot_product: d_k must be >= 1, got {dk}")

    scores = Qa @ Ka.T / np.sqrt(float(dk))
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 1:
            m = np.broadcast_to(m.reshape(1, -1), scores.shape)
        if m.shape != scores.shape:
            raise ValueError(
                f"geron_scaled_dot_product: mask has shape {m.shape} but the score matrix is {scores.shape}"
            )
        keep = m.astype(bool)
        if not np.all(keep.any(axis=1)):
            blind = np.flatnonzero(~keep.any(axis=1)).tolist()
            raise ValueError(
                f"geron_scaled_dot_product: query row(s) {blind} have no visible key; attention is undefined"
            )
        scores = np.where(keep, scores, -np.inf)

    attn = np.empty_like(scores)
    for i in range(scores.shape[0]):
        row = scores[i]
        finite = np.isfinite(row)
        a = np.zeros_like(row)
        if int(finite.sum()) == 1:
            a[finite] = 1.0
        else:
            a[finite] = np.asarray(geron_softmax_function(row[finite])["p"], dtype=float)
        attn[i] = a
    Y = attn @ Va

    return RichResult(
        title="Scaled dot-product attention",
        summary_lines=[("Queries", int(Qa.shape[0])), ("Keys", int(Ka.shape[0])), ("d_k", dk)],
        interpretation=(
            "Dividing by sqrt(d_k) keeps the logit variance near 1 as the head width grows, which stops "
            "the softmax from saturating into a hard argmax."
        ),
        payload={
            "Y": Y,
            "output": Y,
            "attention": attn,
            "scores": scores,
            "d_k": dk,
            "estimate": float(np.max(attn)),
            "n": int(Qa.shape[0]),
            "method": "Scaled dot-product attention with pre-softmax masking",
        },
    )


def cheatsheet():
    return "hmsdp: Scaled dot-product attention"
