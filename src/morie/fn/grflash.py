# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FlashAttention: tiled softmax(QK^T)V with online normalisation."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["geron_flash_attention_tile"]

_METHOD = "FlashAttention tiled online-softmax attention"


def geron_flash_attention_tile(Q, K, V, block_size=2):
    r"""Compute attention block by block, never materialising ``QK^T``.

    For each key block the running maximum ``m``, the running
    denominator ``l`` and the running output ``O`` are corrected:

    .. math::
        m^{\text{new}} &= \max(m, \tilde m) \\
        l^{\text{new}} &= e^{m - m^{\text{new}}} l
        + e^{\tilde m - m^{\text{new}}}\tilde l \\
        O^{\text{new}} &= \frac{e^{m-m^{\text{new}}} l\, O
        + e^{\tilde m - m^{\text{new}}}\, \tilde P V_j}{l^{\text{new}}}

    The rescaling factors are what make this *exact*, not an
    approximation: every partial sum already accumulated is corrected by
    the same exponential shift the new block introduces, so the final
    result is the ordinary softmax attention to the last bit.  The gain
    is memory -- the ``T x T`` score matrix is never held, only one
    ``block_size`` tile at a time, which is why the peak memory is
    linear rather than quadratic in sequence length.

    The claim is checked, not asserted: the direct result from
    :func:`morie.fn.attsdp.scaled_dot_product_attention` is computed
    alongside and the maximum discrepancy is reported as
    ``max_abs_error``.

    Parameters
    ----------
    Q : array-like, shape (Tq, d)
    K, V : array-like, shape (Tk, d)
    block_size : int, optional
        Key-block width, at least 1. Default 2.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``reference_output``, ``max_abs_error``,
        ``n_blocks``, ``row_max``, ``row_denominator``,
        ``peak_score_elements`` (scores held at once, vs ``Tq * Tk`` for
        the direct form), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 17, FlashAttention section (Dao et al. 2022).

    Examples
    --------
    Tiled and direct agree to machine precision:

    >>> Kv = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> r = geron_flash_attention_tile([[1.0, 0.0]], Kv, Kv, block_size=2)
    >>> r["max_abs_error"] < 1e-12
    True
    >>> r["n_blocks"]
    2

    The block size changes only the schedule, never the answer:

    >>> r1 = geron_flash_attention_tile([[1.0, 0.0]], Kv, Kv, block_size=1)
    >>> r3 = geron_flash_attention_tile([[1.0, 0.0]], Kv, Kv, block_size=3)
    >>> max(abs(a - b) for ra, rb in zip(r1["output"], r3["output"])
    ...     for a, b in zip(ra, rb)) < 1e-12
    True

    And it saves memory: 2 score elements held instead of 3.

    >>> r["peak_score_elements"], r["full_score_elements"]
    (2, 3)
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Ka = np.atleast_2d(np.asarray(K, dtype=float))
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    if Qa.shape[1] != Ka.shape[1]:
        raise ValueError(f"Q and K must share d_k; got {Qa.shape[1]} and {Ka.shape[1]}.")
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(f"K has {Ka.shape[0]} rows but V has {Va.shape[0]}.")
    if not (np.all(np.isfinite(Qa)) and np.all(np.isfinite(Ka)) and np.all(np.isfinite(Va))):
        raise ValueError("Q, K and V must be finite.")
    bs = int(block_size)
    if bs < 1:
        raise ValueError(f"block_size must be a positive integer, got {bs}.")

    Tq, dk = Qa.shape
    Tk, dv = Va.shape
    scale = 1.0 / np.sqrt(dk)

    m = np.full(Tq, -np.inf)
    l = np.zeros(Tq)
    O = np.zeros((Tq, dv))
    n_blocks = 0
    for start in range(0, Tk, bs):
        Kj = Ka[start:start + bs]
        Vj = Va[start:start + bs]
        S = (Qa @ Kj.T) * scale                    # (Tq, <=bs) -- the only tile held
        m_tilde = S.max(axis=1)
        P = np.exp(S - m_tilde[:, None])
        l_tilde = P.sum(axis=1)
        m_new = np.maximum(m, m_tilde)
        a = np.exp(np.where(np.isfinite(m), m, 0.0) - m_new) * np.isfinite(m)
        b = np.exp(m_tilde - m_new)
        l_new = a * l + b * l_tilde
        O = (a[:, None] * l[:, None] * O + b[:, None] * (P @ Vj)) / l_new[:, None]
        m, l = m_new, l_new
        n_blocks += 1

    ref = scaled_dot_product_attention(Qa, Ka, Va)
    R = np.asarray(ref["output"], dtype=float)
    err = float(np.max(np.abs(O - R)))

    return RichResult(
        title="FlashAttention (tiled)",
        summary_lines=[("Blocks", n_blocks), ("Block size", bs),
                       ("Max abs error vs direct", err)],
        payload={
            "output": O.tolist(),
            "reference_output": R.tolist(),
            "max_abs_error": err,
            "n_blocks": int(n_blocks),
            "row_max": m.tolist(),
            "row_denominator": l.tolist(),
            "peak_score_elements": int(Tq * min(bs, Tk)),
            "full_score_elements": int(Tq * Tk),
            "block_size": bs,
            "estimate": O.tolist(),
            "n": int(Tq),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grflash: online-softmax tiling, exact vs attsdp, peak memory O(Tq * block) not O(Tq*Tk)"
