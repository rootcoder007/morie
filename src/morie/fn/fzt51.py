# morie.fn -- function file (rootcoder007/morie)
"""Equivalence of the naive kernel and empirical goodness-of-fit statistics (Theorem 5.1)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kerngofeq", "fauzi_thm5_1_naive_kernel_equiv"]


def kerngofeq(ks_emp, ks_kernel, cvm_emp, cvm_kernel, tol=0.05):
    r"""Equivalence of the naive kernel and empirical goodness-of-fit statistics (Theorem 5.1).

    Theorem 5.1: under :math:`H_0: F_X = F`,

    .. math:: |KS_n - \widehat{KS}| \to_p 0
              \quad\text{and}\quad
              |CvM_n - \widehat{CvM}| \to_p 0,

    where the hatted statistics use the NAIVE kernel distribution function
    estimator of (5.3)-(5.4).

    The consequence Sec. 5.1 draws is the practical one: the smoothed
    statistics have the very same Kolmogorov and Cramer-von Mises limiting
    distributions, so the SAME critical values are used. Smoothing is not
    a new test, it is a better-calibrated computation of the same test.

    This routine reports the two differences and compares them against a
    caller-supplied tolerance. It is a convergence DIAGNOSTIC, not a
    hypothesis test: "converges in probability to zero" is a statement
    about a sequence, and no single sample can confirm or refute it. The
    payload therefore returns ``close``, not a p-value.

    Parameters
    ----------
    ks_emp, ks_kernel : float
        The empirical and naive-kernel Kolmogorov-Smirnov statistics.
    cvm_emp, cvm_kernel : float
        The empirical and naive-kernel Cramer-von Mises statistics.
    tol : float, default 0.05
        Tolerance against which the differences are reported.

    Returns
    -------
    RichResult
        Keys ``ksdiff``, ``cvmdiff``, ``close``, ``tol``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.1.
    """
    tol = float(tol)
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    ksd = abs(float(ks_emp) - float(ks_kernel))
    cvmd = abs(float(cvm_emp) - float(cvm_kernel))
    return RichResult(
        payload={
            "ksdiff": float(ksd),
            "cvmdiff": float(cvmd),
            "close": bool(ksd < tol and cvmd < tol),
            "tol": tol,
            "method": "naive kernel vs empirical GOF equivalence (Theorem 5.1)",
        }
    )


fauzi_thm5_1_naive_kernel_equiv = kerngofeq


def cheatsheet():
    return "fzt51: Theorem 5.1: empirical and naive kernel statistics have the same limit, so the same critical values"


# CANONICAL TEST
# >>> r = kerngofeq(ks_emp=0.20, ks_kernel=0.21, cvm_emp=0.30, cvm_kernel=0.31)
# >>> r['close']
# True
