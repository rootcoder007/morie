# morie.fn -- function file (rootcoder007/morie)
"""Mantel-Haenszel pooled odds ratio -- an alias for :mod:`mhors`.

The audit at ``ledger/wave2/DUPMAP.tsv`` records ``mamh`` as a duplicate
of ``mhors``, and it is: the same estimator with the same
Robins-Breslow-Greenland variance.  Rather than carry the arithmetic
twice -- two copies agree with each other at 1e-9 forever and neither
is ever checked against the other -- this module only adapts the calling
convention.  ``mhors`` takes a list of ``(a, b, c, d)`` tables; the name
``mamh`` is documented as taking four parallel vectors.
"""

from . import _s03core as core
from .mhors import mantel_haenszel_or

__all__ = ["ma_mantel_haenszel"]


def ma_mantel_haenszel(a, b, c, d, confidence=0.95):
    """Pool odds ratios across strata without modelling the strata.

    Fitting a stratum effect per table costs a parameter per table, and
    with sparse tables the maximum-likelihood estimate is badly biased --
    the classic Neyman-Scott problem.  The Mantel-Haenszel weights step
    around it: they are the weights that stay consistent both when the
    strata are few and large and when they are many and small, which no
    likelihood-based weighting achieves at once.

    Formula: ``OR_MH = sum(a_k d_k/n_k) / sum(b_k c_k/n_k)``, with the
    Robins-Breslow-Greenland variance for its logarithm -- Mantel &
    Haenszel (1959); Robins, Breslow & Greenland (1986).

    This is an alias.  The estimator lives in ``morie.fn.mhors``; here the
    four cell vectors are zipped into the list of tables it expects.

    Parameters
    ----------
    a, b, c, d : array-like
        Per-stratum cells: exposed cases, exposed non-cases, unexposed
        cases, unexposed non-cases.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes
        Whatever ``mhors.mantel_haenszel_or`` returns, unchanged.

    References
    ----------
    Mantel, N. and Haenszel, W. (1959).  Journal of the National Cancer
    Institute 22(4):719-748.  doi:10.1093/jnci/22.4.719.  Robins, J.,
    Breslow, N. and Greenland, S. (1986).  Biometrics 42(2):311-323.
    doi:10.2307/2531052.
    """
    A = [float(t) for t in core.vec(a)]
    B = [float(t) for t in core.vec(b)]
    C = [float(t) for t in core.vec(c)]
    D = [float(t) for t in core.vec(d)]
    if not (len(B) == len(C) == len(D) == len(A)):
        raise ValueError("the four cell vectors must have equal length")
    return mantel_haenszel_or([(A[i], B[i], C[i], D[i])
                               for i in range(len(A))], confidence)


def cheatsheet():
    return "mamh: Mantel-Haenszel pooled odds ratio (alias of mhors)"
