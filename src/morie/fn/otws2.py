# morie.fn -- function file (rootcoder007/morie)
"""Univariate Wasserstein distance from order statistics."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_wasserstein_p_1d"]


def ot_wasserstein_p_1d(x, y, p=2):
    """Wasserstein distance on the line, by sorting.

    In one dimension the optimal coupling is the monotone one, so the
    whole linear program collapses to a pair of sorts: the k-th smallest
    goes to the k-th smallest.  This is not an approximation, and it is
    the reason every sliced method exists.

    Formula: ``W_p^p = (1/n) sum_i |x_(i) - y_(i)|^p`` for two
    equal-weight samples -- Bobkov & Ledoux (2019) Section 2; Peyre &
    Cuturi (2019) Remark 2.30.

    Parameters
    ----------
    x, y : array-like
        Two samples of equal length.
    p : float, default 2
        Exponent, positive.

    Returns
    -------
    RichResult
        ``Wp``, ``Wp_p`` (the un-rooted cost), ``n``, ``p``.

    References
    ----------
    Bobkov, S. and Ledoux, M. (2019).  One-dimensional empirical
    measures, order statistics, and Kantorovich transport distances.
    Memoirs of the American Mathematical Society 261(1259).
    doi:10.1090/memo/1259.
    """
    w = ot.wp1d(x, y, float(p))
    n = len(core.vec(x))
    return RichResult(payload={
        "Wp": w, "Wp_p": w ** float(p), "n": n, "p": float(p),
        "method": "Univariate Wasserstein distance"})


def cheatsheet():
    return "otws2: univariate Wasserstein distance from sorted samples"
