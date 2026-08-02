# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head attention."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["geron_multi_head_attention"]

_METHOD = "Multi-head attention"


def geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h, mask=None):
    r"""Run ``h`` attention heads in parallel and mix them.

    .. math::
        \mathrm{MHA}(Q,K,V) = \mathrm{Concat}(\text{head}_1, \dots,
        \text{head}_h)\, W_O,\qquad
        \text{head}_i = \mathrm{Attn}(Q W_Q^i, K W_K^i, V W_V^i)

    The projections cut ``d_model`` into ``h`` slices of ``d_model/h``,
    so multi-head attention costs the same as one full-width head --
    the heads are bought by *narrowing*, not by adding compute.  What
    they buy is several attention patterns at once: one head can track
    the syntactic subject while another tracks the previous token, which
    a single softmax cannot do because it has one distribution to spend.

    Each head is delegated to
    :func:`morie.fn.attsdp.scaled_dot_product_attention`, so the
    ``sqrt(d_k)`` scaling and masking semantics are shared with the
    single-head implementation rather than re-derived.

    Parameters
    ----------
    Q : array-like, shape (Tq, d_model)
    K, V : array-like, shape (Tk, d_model)
    WQ, WK, WV : array-like, shape (d_model, d_model)
    WO : array-like, shape (d_model, d_out)
    h : int
        Number of heads; must divide ``d_model``.
    mask : array-like, optional
        Passed through to each head (additive or boolean).

    Returns
    -------
    RichResult
        Payload keys ``output``, ``head_outputs``, ``attention_weights``
        (one matrix per head), ``concat``, ``d_head``, ``n_heads``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Multi-Head Attention section (Vaswani et al. 2017).

    Examples
    --------
    One head with identity projections is exactly single-head attention
    -- the same 0.669762 that ``attsdp`` reports:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> Kv = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_multi_head_attention([[1.0, 0.0]], Kv, Kv, I, I, I, I, h=1)
    >>> round(r["output"][0][0], 6)
    0.669762
    >>> r["d_head"]
    2

    Two heads over the same width give two 1-wide heads, and each
    attention row is still a distribution:

    >>> r2 = geron_multi_head_attention([[1.0, 0.0]], Kv, Kv, I, I, I, I, h=2)
    >>> r2["n_heads"], r2["d_head"]
    (2, 1)
    >>> [round(sum(row), 10) for A in r2["attention_weights"] for row in A]
    [1.0, 1.0]
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Ka = np.atleast_2d(np.asarray(K, dtype=float))
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(f"K has {Ka.shape[0]} positions but V has {Va.shape[0]}.")
    d_model = Qa.shape[1]
    if Ka.shape[1] != d_model or Va.shape[1] != d_model:
        raise ValueError(
            f"Q, K and V must share d_model; got {d_model}, {Ka.shape[1]}, {Va.shape[1]}."
        )
    h = int(h)
    if h < 1:
        raise ValueError(f"h must be a positive integer, got {h}.")
    if d_model % h:
        raise ValueError(
            f"h = {h} does not divide d_model = {d_model}; heads must partition the "
            f"model width exactly."
        )
    d_head = d_model // h

    mats = {}
    for name, M, cols in (("WQ", WQ, d_model), ("WK", WK, d_model),
                          ("WV", WV, d_model), ("WO", WO, None)):
        A = np.atleast_2d(np.asarray(M, dtype=float))
        if A.shape[0] != d_model:
            raise ValueError(f"{name} must have {d_model} rows, got shape {A.shape}.")
        if cols is not None and A.shape[1] != cols:
            raise ValueError(f"{name} must have {cols} columns, got shape {A.shape}.")
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} must be finite.")
        mats[name] = A
    if not (np.all(np.isfinite(Qa)) and np.all(np.isfinite(Ka)) and np.all(np.isfinite(Va))):
        raise ValueError("Q, K and V must be finite.")

    Qp, Kp, Vp = Qa @ mats["WQ"], Ka @ mats["WK"], Va @ mats["WV"]
    heads, weights = [], []
    for i in range(h):
        sl = slice(i * d_head, (i + 1) * d_head)
        res = scaled_dot_product_attention(Qp[:, sl], Kp[:, sl], Vp[:, sl], mask=mask)
        heads.append(np.asarray(res["output"], dtype=float))
        weights.append(res["attention"])
    concat = np.concatenate(heads, axis=1)
    out = concat @ mats["WO"]

    return RichResult(
        title="Multi-head attention",
        summary_lines=[("Heads", h), ("d_head", d_head),
                       ("Queries", int(Qa.shape[0]))],
        payload={
            "output": out.tolist(),
            "head_outputs": [H.tolist() for H in heads],
            "attention_weights": weights,
            "concat": concat.tolist(),
            "d_head": int(d_head),
            "d_model": int(d_model),
            "n_heads": h,
            "estimate": out.tolist(),
            "n": int(Qa.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmha: h heads of width d_model/h (each via attsdp), concatenated then mixed by WO"
