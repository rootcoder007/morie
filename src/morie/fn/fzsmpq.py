# morie.fn -- function file (rootcoder007/morie)
"""Sample quantile from the empirical distribution function."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["smpqnt", "fauzi_sample_quantile"]


def smpqnt(x, p=0.5):
    r"""Sample quantile from the empirical distribution function.

    .. math:: \hat Q(p) = \inf\{t : F_n(t) \ge p\},

    the generalised inverse of the empirical df. Concretely the order
    statistic :math:`X_{(\lceil np\rceil)}`, and the estimator Chapter 3
    measures the kernel quantile estimator against.

    Its asymptotic variance is :math:`p(1-p)/(nf^2(Q(p)))` (Eq. 3.2), the
    same first-order quantity the kernel estimator attains -- Remark 3.3
    is explicit that a kernel can only MATCH the sample quantile to first
    order, so any gain has to show up in the Edgeworth term. That is the
    reason Chapter 3 exists.

    No interpolation and no plotting-position convention: the definition
    is a strict infimum, so this is R's ``type = 1`` and NOT the default
    ``type = 7``. Using an interpolating quantile here would silently
    change the estimand.

    Parameters
    ----------
    x : array-like
        Sample.
    p : float or array-like, default 0.5
        Probabilities in ``(0, 1]``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``index``, ``p``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (3.1) and the display defining the sample quantile in Sec. 3.2.
    """
    xv = np.sort(np.asarray(x, dtype=float).ravel())
    n = xv.size
    if n < 1:
        raise ValueError("need at least one observation.")
    pv = np.atleast_1d(np.asarray(p, dtype=float))
    if np.any(pv <= 0) or np.any(pv > 1):
        raise ValueError("probabilities must lie in (0, 1].")
    idx = [int(np.ceil(float(q) * n)) for q in pv]
    idx = [max(1, min(n, i)) for i in idx]
    return RichResult(
        payload={
            "estimate": [float(xv[i - 1]) for i in idx],
            "index": idx,
            "p": [float(q) for q in pv],
            "n": int(n),
            "method": "sample quantile, inf{t: F_n(t) >= p}",
        }
    )


fauzi_sample_quantile = smpqnt


def cheatsheet():
    return "fzsmpq: sample quantile as a strict infimum (type 1), the Ch 3 benchmark"


# CANONICAL TEST
# >>> smpqnt([1.0, 2.0, 3.0, 4.0], p=0.5)['estimate'][0] == 2.0
# True
