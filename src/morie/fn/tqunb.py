# morie.fn -- function file (rootcoder007/morie)
"""Unbiasedness of the ProdQJL estimator (TurboQuant Lemma 3.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["turboquant_prodqjl_unbiasedness"]


def turboquant_prodqjl_unbiasedness(q, k, m=64, trials=2000, seed=0):
    r"""Verify Lemma 3.2 numerically rather than asserting it.

    The lemma states that the ProdQJL estimator is unbiased,

    .. math::
       \mathbb{E}\left[\frac{1}{m}\sqrt{\frac{\pi}{2}}\,
         \lVert k\rVert_2\,\langle Sq,\ \mathrm{sign}(Sk)\rangle\right]
       = \langle q, k\rangle .

    The proof rests on one fact about a Gaussian row :math:`s`: the
    sign of :math:`\langle s, k\rangle` depends only on the DIRECTION
    of :math:`k`, so

    .. math::
       \mathbb{E}\big[\langle s,q\rangle\,
       \mathrm{sign}(\langle s,k\rangle)\big]
       = \sqrt{\tfrac{2}{\pi}}\,\frac{\langle q,k\rangle}{\lVert k\rVert}.

    Unbiasedness is the property that matters downstream. A biased
    inner-product estimate would shift every attention score in the
    same direction, and the softmax would not cancel it; an unbiased
    one with variance :math:`O(1/m)` merely adds noise that averages
    out across heads and positions.

    This function draws ``trials`` independent projections and reports
    the empirical mean against the exact product, together with the
    standard error, so the check is a measurement rather than a claim.

    Parameters
    ----------
    q, k : array-like, shape (d,)
    m : int
        Sketch dimension.
    trials : int
        Independent draws of ``S``.
    seed : int

    Returns
    -------
    RichResult
        ``exact``, ``mean_estimate``, ``bias``, ``se``, ``z``,
        ``unbiased``, ``variance``, ``variance_scaling``.

    References
    ----------
    Zandieh, Daliri and Han (2024), arXiv:2406.03482, Lemma 3.2.

    Examples
    --------
    >>> out = turboquant_prodqjl_unbiasedness([1.0, 0.0], [1.0, 0.0],
    ...                                       m=32, trials=400)
    >>> bool(out["unbiased"])
    True
    """
    qv = np.asarray(q, dtype=float).ravel()
    kv = np.asarray(k, dtype=float).ravel()
    if qv.size != kv.size:
        raise ValueError(
            "q and k must have the same dimension, got %d and %d."
            % (qv.size, kv.size)
        )
    d = qv.size
    m = int(m)
    if m < 1:
        raise ValueError("m must be positive, got %d." % m)
    nk = float(np.linalg.norm(kv))
    if nk == 0:
        raise ValueError("k must be non-zero.")
    exact = float(qv @ kv)

    rng = np.random.default_rng(int(seed))
    T = int(trials)
    ests = np.empty(T)
    c = np.sqrt(np.pi / 2.0) / m * nk
    for t in range(T):
        S = rng.normal(size=(m, d))
        ests[t] = c * (np.sign(S @ kv) @ (S @ qv))
    mean = float(ests.mean())
    se = float(ests.std(ddof=1) / np.sqrt(T))
    z = float((mean - exact) / se) if se > 0 else np.nan
    return RichResult(
        payload={
            "estimate": mean,
            "exact": exact,
            "mean_estimate": mean,
            "bias": float(mean - exact),
            "se": se,
            "z": z,
            "unbiased": bool(abs(z) < 4.0) if np.isfinite(z) else False,
            "z_note": (
                "standardised bias; |z| under about 4 is consistent with "
                "exact unbiasedness at this number of trials"
            ),
            "variance": float(ests.var(ddof=1)),
            "variance_scaling": float(ests.var(ddof=1) * m),
            "scaling_note": (
                "variance times m should be roughly constant in m, which is "
                "the O(1/m) rate the lemma implies"
            ),
            "downstream_note": (
                "bias would shift every attention score the same way and the "
                "softmax would not cancel it; unbiased noise averages out "
                "across heads and positions"
            ),
            "m": m,
            "d": int(d),
            "trials": T,
            "method": "Numerical check of ProdQJL unbiasedness (Lemma 3.2)",
        }
    )


def cheatsheet():
    return (
        "tqunb: measures the ProdQJL bias against the exact inner product "
        "and checks the O(1/m) variance rate"
    )
