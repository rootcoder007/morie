# morie.fn -- function file (rootcoder007/morie)
"""Equivalence of the boundary-free and empirical CvM statistics (Theorem 5.7)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfcvmeq", "fauzi_thm5_7_bdfree_cvm_equiv"]


def bfcvmeq(empirical, smoothed, tol=0.05, h=None, n=None):
    r"""Equivalence of the boundary-free and empirical CvM statistics (Theorem 5.7).

    Theorem 5.7: under :math:`H_0: F_X = F` on :math:`\Omega`,

    .. math:: |CvM_n - \tilde{CvM}| \to_p 0,

    with :math:`\tilde{CvM}` built from the boundary-free estimator (5.5).

    The proof needs something Theorem 5.6 did not: it ASSUMES the
    bandwidth satisfies :math:`h = o(n^{-1/4})`. That is not decoration.
    A supremum difference is controlled by a uniform bound, but the CvM
    difference is an integral of a SQUARED discrepancy multiplied by
    :math:`n`, so a bias of order :math:`h^2` contributes
    :math:`nh^4`, and only :math:`h = o(n^{-1/4})` makes that vanish.

    The same :math:`n^{-1/4}` threshold appears in (3.8) for the quantile
    Edgeworth expansion and in Theorem 5.9 for the smoothed Wilcoxon test,
    and for the same reason each time: whenever a squared bias is
    multiplied by ``n``, undersmoothing becomes compulsory.

    ``h`` may be passed so the condition is checked rather than assumed;
    ``bwok`` reports it.

    Parameters
    ----------
    empirical : float
        The empirical statistic ``CvM_n``.
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
    Fauzi and Maesono (2023), Theorem 5.7.
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
            "method": "boundary-free vs empirical CvM equivalence (Theorem 5.7)",
        }
    )


fauzi_thm5_7_bdfree_cvm_equiv = bfcvmeq


def cheatsheet():
    return "fzt57: Theorem 5.7: empirical and boundary-free statistics have the same limit, so the same critical values"


# CANONICAL TEST
# >>> r = bfcvmeq(empirical=0.20, smoothed=0.21, h=0.05, n=1000)
# >>> r['close'] and r['bwok']
# True
