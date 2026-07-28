# morie.fn -- function file (rootcoder007/morie)
"""Estimated true preferential ordering from k rankings."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_concordance_preference"]


def _rank(v):
    """Average ranks, ties shared."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(v.size, dtype=float)
    r[order] = np.arange(1, v.size + 1, dtype=float)
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    for u in np.nonzero(cnt > 1)[0]:
        m = inv == u
        r[m] = r[m].mean()
    return r


def gibbons_concordance_preference(rankings):
    r"""Estimate the agreed-upon ordering behind :math:`k` rankings.

    Given a :math:`k \times n` table of rankings of :math:`n` objects
    by :math:`k` observers, the estimated true preferential ordering is
    obtained by ranking the objects according to their COLUMN SUMS --
    the rank totals.

    Gibbons and Chakraborti's justification is an optimality one, not
    an appeal to convenience: this estimate is best in the sense that
    if the Spearman coefficient is computed between the estimated
    ordering and each of the :math:`k` observed rankings, the average
    of those :math:`k` coefficients is a MAXIMUM. ``rho_bar`` reports
    that maximised average, and ``rho_by_observer`` the individual
    values it averages, so an observer who dissents from the consensus
    is visible rather than buried in it.

    The ordering is only worth reading when the rankings actually
    agree. Kendall's :math:`W` is returned alongside for that reason:
    under independence its expectation is :math:`1/k`, not zero, so a
    small positive :math:`W` is what disagreement looks like and not
    evidence of weak consensus.

    Parameters
    ----------
    rankings : array-like, shape (k, n)
        Row per observer, column per object. Values are ranks, or any
        scores that will be ranked within each row.

    Returns
    -------
    RichResult
        ``order`` (object indices best-first), ``preference_rank``
        (rank of each object), ``rank_sums``, ``W``, ``rho_bar``,
        ``rho_by_observer``, ``expected_W_under_independence``.

    References
    ----------
    Gibbons and Chakraborti (2011), *Nonparametric Statistical
    Inference*, 5th ed., section 12.4.3, pp. 456-458, and section
    12.4.1 for the relationship used by ``rho_bar``.

    Examples
    --------
    >>> r = [[1, 2, 3], [1, 2, 3], [2, 1, 3]]
    >>> out = gibbons_concordance_preference(r)
    >>> [float(v) for v in out["preference_rank"]]
    [1.0, 2.0, 3.0]
    """
    R = np.atleast_2d(np.asarray(rankings, dtype=float))
    k, n = R.shape
    if k < 2:
        raise ValueError("need at least 2 rankings, got %d." % k)
    if n < 2:
        raise ValueError("need at least 2 objects, got %d." % n)
    if np.any(~np.isfinite(R)):
        raise ValueError("rankings contain non-finite values.")
    # rank within each observer, so raw scores are accepted too
    Rk = np.vstack([_rank(row) for row in R])

    sums = Rk.sum(axis=0)
    pref = _rank(sums)                       # small rank total = preferred
    order = np.argsort(sums, kind="mergesort")

    mu = k * (n + 1) / 2.0
    S = float(np.sum((sums - mu) ** 2))
    denom = k ** 2 * (n ** 3 - n) / 12.0
    W = S / denom if denom > 0 else np.nan

    rho_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            rho_pairs.append(_spearman(Rk[i], Rk[j]))
    rho_av = float(np.mean(rho_pairs)) if rho_pairs else np.nan
    rho_obs = np.array([_spearman(Rk[i], pref) for i in range(k)])

    return RichResult(
        payload={
            "estimate": pref,
            "preference_rank": pref,
            "order": order,
            "rank_sums": sums,
            "W": float(W),
            "rho_av_pairwise": rho_av,
            "rho_bar": float(np.mean(rho_obs)),
            "rho_by_observer": rho_obs,
            "optimality_note": (
                "ranking by the column sums maximises the average Spearman "
                "correlation between the estimate and the k observed "
                "rankings; no other ordering does better"
            ),
            "expected_W_under_independence": 1.0 / k,
            "W_note": (
                "under independence E[W] = 1/k, not 0, so a small positive W "
                "is what disagreement looks like"
            ),
            "k": int(k),
            "n": int(n),
            "method": "Estimated true preferential ordering from k rankings",
        }
    )


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def cheatsheet():
    return (
        "gb_wsp: consensus ordering from rank totals -- the ordering that "
        "maximises average Spearman agreement with the k observers"
    )
