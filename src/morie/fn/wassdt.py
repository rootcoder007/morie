"""1-Wasserstein distance on the line."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["wasserstein_1d"]


def wasserstein_1d(p, q, support=None):
    """
    1-Wasserstein distance (1D)

    Formula: W1(p, q) = integral |F_p(x) - F_q(x)| dx

    On the real line the Kantorovich-Rubinstein dual of the p = 1
    transport problem,

        W1(p, q) = sup_{Lip(f) <= 1} ( integral f dp - integral f dq ),

    has the closed form given above: the transport cost is the area
    between the two cumulative distribution functions.  No linear
    programme is needed, which is why this function has no solver.

    Two input conventions are accepted.

    * ``support`` given -- ``p`` and ``q`` are probability masses placed
      on the common ``support`` grid.  Both are normalised to sum to one
      and the integral is evaluated exactly on that grid: the CDFs are
      step functions constant between consecutive support points, so

          W1 = sum_i |cumsum(p)_i - cumsum(q)_i| * (s_{i+1} - s_i).

    * ``support`` omitted -- ``p`` and ``q`` are raw samples and the
      empirical distributions are compared.  This is delegated to
      ``_stats_core.wasserstein_distance``, which applies the same
      closed form to the pooled order statistics.

    Parameters
    ----------
    p : array-like
        Probability masses on ``support``, or a sample.
    q : array-like
        Probability masses on ``support``, or a sample.
    support : array-like, optional
        Common support points.  Must be strictly increasing and the same
        length as ``p`` and ``q``.

    Returns
    -------
    result : RichResult
        Keys: distance, n_support, method.

    References
    ----------
    Kantorovich L V & Rubinstein G S (1958).  On a space of completely
    additive functions.  Vestnik Leningrad. Univ. 13(7), 52-59.  The
    duality theorem stated there gives the sup-over-1-Lipschitz form; the
    one-dimensional closed form is Theorem 2.18 of Villani C (2009),
    Optimal Transport: Old and New, Springer.
    """
    if support is None:
        pv = np.atleast_1d(np.asarray(p, dtype=float))
        qv = np.atleast_1d(np.asarray(q, dtype=float))
        if len(pv) == 0 or len(qv) == 0:
            raise ValueError("both samples must be non-empty")
        d = stats.wasserstein_distance(pv.tolist(), qv.tolist())
        return RichResult(
            payload={
                "distance": float(d),
                "n_support": len(pv) + len(qv),
                "method": "1-Wasserstein distance (empirical, 1D)",
            }
        )

    s = [float(v) for v in np.atleast_1d(np.asarray(support, dtype=float)).tolist()]
    pv = [float(v) for v in np.atleast_1d(np.asarray(p, dtype=float)).tolist()]
    qv = [float(v) for v in np.atleast_1d(np.asarray(q, dtype=float)).tolist()]
    n = len(s)
    if len(pv) != n or len(qv) != n:
        raise ValueError("p, q and support must have the same length")
    if n < 2:
        raise ValueError("need at least two support points")
    for i in range(n - 1):
        if not s[i + 1] > s[i]:
            raise ValueError("support must be strictly increasing")
    if any(v < 0.0 for v in pv) or any(v < 0.0 for v in qv):
        raise ValueError("probability masses must be non-negative")
    sp = 0.0
    sq = 0.0
    for i in range(n):
        sp += pv[i]
        sq += qv[i]
    if sp <= 0.0 or sq <= 0.0:
        raise ValueError("probability masses must have positive total mass")

    cp = 0.0
    cq = 0.0
    total = 0.0
    for i in range(n - 1):
        cp += pv[i] / sp
        cq += qv[i] / sq
        gap = cp - cq
        if gap < 0.0:
            gap = -gap
        total += gap * (s[i + 1] - s[i])
    return RichResult(
        payload={
            "distance": float(total),
            "n_support": n,
            "method": "1-Wasserstein distance (1D, gridded)",
        }
    )


def cheatsheet():
    return "wassdt: 1-Wasserstein distance on the line"


# compact alias per ledger/NAMING.md
wasserstein1d = wasserstein_1d
