# morie.fn -- function file (rootcoder007/morie)
"""Equivalence of the boundary-free and empirical KS statistics (Theorem 5.6)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkseq", "fauzi_thm5_6_bdfree_ks_equiv"]


def bfkseq(empirical, smoothed, tol=0.05, h=None, n=None):
    r"""Equivalence of the boundary-free and empirical KS statistics (Theorem 5.6).

    Theorem 5.6: under :math:`H_0: F_X = F` on the support
    :math:`\Omega`,

    .. math:: |KS_n - \tilde{KS}| \to_p 0,

    with :math:`\tilde{KS}` built from the BOUNDARY-FREE estimator (5.5).

    The proof is short and worth knowing, because it explains why the rate
    is what it is. Both statistics are suprema, so their difference is
    bounded by :math:`\sup_x|\tilde F_X(x) - F_n(x)|`. Transforming to
    :math:`Y_i = g^{-1}(X_i)`, that becomes
    :math:`\sup_y|\tilde F_Y(y) - F_{n,Y}(y)|`, which is known to be
    :math:`o_p(n^{-1/2})` for the ordinary kernel estimator on the whole
    line. The bijection carries the result back unchanged because
    :math:`\tilde F_Y(g^{-1}(x)) = \tilde F_X(x)` and
    :math:`F_{n,Y}(g^{-1}(x)) = F_n(x)` EXACTLY -- the same
    change-of-variable identity that made (5.5) work in the first place.

    So the convergence is at rate :math:`n^{-1/2}`, faster than the
    :math:`n^{-1/2}` scale of the statistics themselves is coarse; the
    difference vanishes on the scale at which the test operates.

    Parameters
    ----------
    empirical : float
        The empirical statistic ``KS_n``.
    smoothed : float
        The boundary-free statistic.
    tol : float, default 0.05
        Tolerance against which the difference is reported.
    h : float, optional
        Bandwidth; when given with ``n`` the ``h = o(n^{-1/4})``
        condition is checked.
    n : int, optional
        Sample size.

    Returns
    -------
    RichResult
        Keys ``difference``, ``close``, ``tol``, ``bwok``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.6.
    """
    tol = float(tol)
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    d = abs(float(empirical) - float(smoothed))
    if h is None or n is None:
        bwok = None
    else:
        bwok = bool(float(h) < float(int(n)) ** -0.25)
    return RichResult(
        payload={
            "difference": float(d),
            "close": bool(d < tol),
            "tol": tol,
            "bwok": bwok,
            "method": "boundary-free vs empirical KS equivalence (Theorem 5.6)",
        }
    )


fauzi_thm5_6_bdfree_ks_equiv = bfkseq


def cheatsheet():
    return "fzt56: Theorem 5.6: empirical and boundary-free statistics have the same limit, so the same critical values"


# CANONICAL TEST
# >>> r = bfkseq(empirical=0.20, smoothed=0.21, h=0.05, n=1000)
# >>> r['close'] and r['bwok']
# True
