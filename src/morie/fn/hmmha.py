# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head attention: concat heads, then linear projection."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["geron_multihead_attention"]

_METHOD = "Multi-head attention (split, attend, concat, project)"


def geron_multihead_attention(Q, K, V, n_heads, W_O=None, mask=None):
    """
    Multi-head attention: concat heads, then linear projection.

    Formula: MHA(Q,K,V) = Concat(head_1,...,head_h) W_O

    The heads are formed by **splitting the model dimension**, not by
    replicating it: with ``d_model = 8`` and ``n_heads = 2`` each head
    sees 4 dimensions, so the total work is the same as one big head and
    the ``1/sqrt(d_k)`` scale inside each head uses the *head* width,
    not ``d_model``.  Getting that wrong is the classic multi-head bug --
    it makes the softmax far too flat.

    Each head's attention is delegated to
    :func:`morie.fn.attsdp.scaled_dot_product_attention`, which already
    handles the scale, the mask and the row-stochastic softmax.

    Parameters
    ----------
    Q : array-like, shape (n_q, d_model)
        Queries.
    K, V : array-like, shape (n_kv, d_model)
        Keys and values.  ``V`` may have its own width as long as it is
        divisible by ``n_heads``.
    n_heads : int
        Number of heads; must divide the query/key width.
    W_O : array-like, shape (d_v, d_out), optional
        Output projection.  Defaults to the identity, which leaves the
        concatenated heads unprojected.
    mask : array-like, optional
        Passed through to each head, shape ``(n_q, n_kv)``.

    Returns
    -------
    result : RichResult
        Keys: output, head_outputs, attention_weights, d_head,
        estimate, n, method.

    Examples
    --------
    One head is ordinary scaled dot-product attention:

    >>> Q = [[1.0, 0.0]]
    >>> K = [[1.0, 0.0], [0.0, 1.0]]
    >>> V = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_multihead_attention(Q, K, V, n_heads=1)
    >>> from morie.fn.attsdp import scaled_dot_product_attention as sdpa
    >>> ref = np.asarray(sdpa(Q, K, V)["output"])
    >>> bool(np.allclose(r["output"], ref))
    True

    Two heads over a width-2 model give each head one dimension, so
    ``d_head`` is 1 and each head's scores are scaled by ``sqrt(1)``:

    >>> t = geron_multihead_attention(Q, K, V, n_heads=2)
    >>> t["d_head"]
    1
    >>> t["output"].shape
    (1, 2)

    Attention rows are probability distributions in every head:

    >>> bool(np.allclose([np.sum(A, axis=1) for A in t["attention_weights"]], 1.0))
    True

    A head count that does not divide the width is refused:

    >>> geron_multihead_attention(Q, K, V, n_heads=3)
    Traceback (most recent call last):
        ...
    ValueError: geron_multihead_attention: n_heads=3 does not divide the query/key width 2

    References
    ----------
    Géron Ch 15
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Ka = np.atleast_2d(np.asarray(K, dtype=float))
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    for name, arr in (("Q", Qa), ("K", Ka), ("V", Va)):
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"geron_multihead_attention: {name} must be a non-empty 2-D array, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_multihead_attention: {name} contains non-finite values")
    h = int(n_heads)
    if h < 1:
        raise ValueError(f"geron_multihead_attention: n_heads must be at least 1, got {n_heads!r}")
    if Qa.shape[1] != Ka.shape[1]:
        raise ValueError(
            f"geron_multihead_attention: Q has width {Qa.shape[1]} but K has {Ka.shape[1]}; they share d_k"
        )
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(
            f"geron_multihead_attention: K has {Ka.shape[0]} rows but V has {Va.shape[0]}"
        )
    if Qa.shape[1] % h:
        raise ValueError(
            f"geron_multihead_attention: n_heads={h} does not divide the query/key width {Qa.shape[1]}"
        )
    if Va.shape[1] % h:
        raise ValueError(
            f"geron_multihead_attention: n_heads={h} does not divide the value width {Va.shape[1]}"
        )

    d_head = Qa.shape[1] // h
    d_vhead = Va.shape[1] // h
    outs = []
    attns = []
    for i in range(h):
        qs = slice(i * d_head, (i + 1) * d_head)
        vs = slice(i * d_vhead, (i + 1) * d_vhead)
        head = scaled_dot_product_attention(Qa[:, qs], Ka[:, qs], Va[:, vs], mask=mask)
        outs.append(np.asarray(head["output"], dtype=float))
        attns.append(np.asarray(head["attention"], dtype=float))
    concat = np.concatenate(outs, axis=1)

    if W_O is None:
        Wo = np.eye(concat.shape[1])
    else:
        Wo = np.atleast_2d(np.asarray(W_O, dtype=float))
        if Wo.shape[0] != concat.shape[1]:
            raise ValueError(
                f"geron_multihead_attention: the concatenated heads have width {concat.shape[1]} "
                f"but W_O has {Wo.shape[0]} rows"
            )
        if not np.all(np.isfinite(Wo)):
            raise ValueError("geron_multihead_attention: W_O contains non-finite values")
    out = concat @ Wo

    return RichResult(
        title="Multi-head attention",
        summary_lines=[
            ("Heads", h),
            ("Head width d_k", d_head),
            ("Output shape", str(out.shape)),
        ],
        interpretation=(
            "Heads split d_model rather than duplicating it, so each head's softmax is scaled by "
            "sqrt(d_head) -- using sqrt(d_model) instead would flatten every distribution."
        ),
        payload={
            "output": out,
            "concat": concat,
            "head_outputs": outs,
            "attention_weights": attns,
            "d_head": int(d_head),
            "n_heads": h,
            "estimate": float(np.linalg.norm(out)),
            "n": int(Qa.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmha: multi-head attention -- split d_model into heads, delegate each to attsdp, concat, project"
