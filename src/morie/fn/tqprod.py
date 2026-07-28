# morie.fn -- function file (rootcoder007/morie)
"""Asymmetric QJL inner-product estimator (TurboQuant eq 4)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["turboquant_qjl_product_estimator"]


def turboquant_qjl_product_estimator(q, signs_k, norm_k, S):
    r"""Recover an inner product from one-bit sketched keys.

    TurboQuant equation (4):

    .. math::
       \mathrm{ProdQJL}(q, k) = \frac{1}{m}\sqrt{\frac{\pi}{2}}\,
         \lVert k\rVert_2 \, \langle Sq,\ \mathrm{sign}(Sk)\rangle

    The estimator is ASYMMETRIC and that is the point: the key is
    stored as :math:`m` sign bits plus one float norm, while the query
    is kept in full precision and merely projected. Attention reads
    every cached key against each new query, so the compression has to
    sit on the key side; sketching the query too would throw away
    precision on the one vector that is not being stored.

    The constant :math:`\sqrt{\pi/2}` is forced, not tuned. For a
    Gaussian row :math:`s`,

    .. math::
       \mathbb{E}\big[\langle s, q\rangle\,
       \mathrm{sign}(\langle s, k\rangle)\big]
       = \sqrt{\tfrac{2}{\pi}}\,
         \frac{\langle q, k\rangle}{\lVert k \rVert_2},

    so multiplying by :math:`\sqrt{\pi/2}\lVert k\rVert` is exactly
    what removes the bias -- see :func:`~morie.fn.tqunb`.

    Parameters
    ----------
    q : array-like, shape (d,)
        Query, full precision.
    signs_k : array-like of {-1, +1}, shape (n, m) or (m,)
        Sign bits of the projected keys.
    norm_k : array-like, shape (n,) or scalar
        Stored key norms.
    S : array-like, shape (m, d)
        Shared JL projection.

    Returns
    -------
    RichResult
        ``estimate`` (one per key), ``sketch_bits``, ``full_bits``,
        ``compression``, ``relative_error`` when norms allow a bound.

    References
    ----------
    Zandieh, Daliri and Han (2024), "QJL: 1-bit quantized JL transform
    for KV cache quantization with zero overhead", arXiv:2406.03482,
    equation (4).
    Zandieh et al. (2026), TurboQuant, ICLR, arXiv:2504.19874.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> out = turboquant_qjl_product_estimator([1.0, 0.0], [[1, 1]], 1.0, S)
    >>> bool(out["estimate"].size == 1)
    True
    """
    qv = np.asarray(q, dtype=float).ravel()
    Sm = np.atleast_2d(np.asarray(S, dtype=float))
    m, d = Sm.shape
    if qv.size != d:
        raise ValueError(
            "q has dimension %d, S expects %d." % (qv.size, d)
        )
    G = np.atleast_2d(np.asarray(signs_k, dtype=float))
    if G.shape[1] != m:
        raise ValueError(
            "signs_k has %d columns, S has %d rows." % (G.shape[1], m)
        )
    if not np.all(np.isin(G, (-1.0, 1.0))):
        raise ValueError("signs_k must contain only -1 and +1.")
    nk = np.atleast_1d(np.asarray(norm_k, dtype=float)).ravel()
    if nk.size == 1:
        nk = np.full(G.shape[0], float(nk[0]))
    if nk.size != G.shape[0]:
        raise ValueError(
            "norm_k has %d entries for %d keys." % (nk.size, G.shape[0])
        )
    if np.any(nk < 0):
        raise ValueError("key norms must be non-negative.")

    Sq = Sm @ qv
    est = np.sqrt(np.pi / 2.0) / m * nk * (G @ Sq)
    # a one-bit sketch plus one float32 norm, against d float16 entries
    sketch = m + 32
    full = 16 * d
    return RichResult(
        payload={
            "estimate": est,
            "inner_product": est,
            "projected_query": Sq,
            "constant": float(np.sqrt(np.pi / 2.0)),
            "constant_note": (
                "sqrt(pi/2) is forced by E[<s,q> sign(<s,k>)] = "
                "sqrt(2/pi) <q,k>/||k||, not chosen"
            ),
            "asymmetry_note": (
                "keys are stored as m sign bits plus a norm, queries stay in "
                "full precision; attention reads every cached key against "
                "each new query, so the compression belongs on the key side"
            ),
            "sketch_bits": int(sketch),
            "full_bits": int(full),
            "compression": float(full / sketch),
            "m": int(m),
            "d": int(d),
            "n_keys": int(G.shape[0]),
            "method": "Asymmetric QJL inner-product estimator (eq 4)",
        }
    )


def cheatsheet():
    return (
        "tqprod: inner products from one-bit sketched keys with a "
        "full-precision query, constant sqrt(pi/2)"
    )
