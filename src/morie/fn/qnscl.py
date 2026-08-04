# morie.fn -- function file (rootcoder007/morie)
"""Qn robust scale estimator of Rousseeuw and Croux.

Source: Rousseeuw, P. J. and Croux, C. (1993), "Alternatives to the
median absolute deviation", *Journal of the American Statistical
Association* 88(424):1273-1283.  The JASA article is paywalled and was
NOT read directly.  The estimator was taken instead from the authors'
own reference implementation, the R package **robustbase** (Rousseeuw is
an author of that package), file ``R/qnsn.R``, which is the definition
this module reproduces:

    Qn = d_n * { |x_i - x_j| ; i < j }_(k),   k = C(h, 2),  h = n%/%2 + 1

that is, the k-th order statistic of the C(n, 2) pairwise absolute
differences.  The consistency constant is

    2.21914 = 1 / (sqrt(2) * Phi^{-1}(5/8))

``qnsn.R`` records that the value 2.2219 printed in the original Fortran
implementation was slightly wrong and was corrected in 2010; the correct
constant above is used here, and ``constant`` is exposed so a caller
reproducing pre-2010 output can pass 2.2219 explicitly.

The finite-sample bias correction is quoted verbatim from ``qnsn.R``:

    n <= 12:  multiply by
        c(.399356, .99365, .51321, .84401, .61220,
          .85877, .66993, .87344, .72014, .88906, .75743)[n - 1]
    n >  12:  divide by
        (if (n %% 2) 1.60188 + (-2.1284 - 5.172/n)/n
         else        3.67561 + ( 1.9654 + (6.987 - 77/n)/n)/n) / n + 1

Selection of the k-th order statistic is done by a full sort of the
pairwise differences.  That is O(n^2 log n) rather than the O(n log n)
algorithm of the paper, and is deliberate: the fast algorithm's answer
is the same order statistic, so the sort trades time for an arithmetic
path that is identical in Python and R and therefore cannot drift.
"""

from ._richresult import RichResult

__all__ = ["qn_scale"]

_QN_SMALL = [0.399356, 0.99365, 0.51321, 0.84401, 0.61220,
             0.85877, 0.66993, 0.87344, 0.72014, 0.88906, 0.75743]


def _qn_finite_c(n):
    """robustbase Qn.finite.c(n), the n > 12 correction denominator."""
    if n % 2:
        inner = 1.60188 + (-2.1284 - 5.172 / n) / n
    else:
        inner = 3.67561 + (1.9654 + (6.987 - 77.0 / n) / n) / n
    return inner / n + 1.0


def qn_scale(y, constant=2.21914, finite_corr=True):
    """Qn scale: a k-th order statistic of the pairwise absolute differences.

    Parameters
    ----------
    y : array-like
        Sample, at least two finite values.
    constant : float
        Consistency constant.  Default 2.21914 = 1/(sqrt(2) Phi^-1(5/8)).
    finite_corr : bool
        Apply the robustbase finite-sample bias correction.

    Returns
    -------
    RichResult
        ``estimate``, ``raw`` (before the constant and the correction),
        ``k`` (the order statistic index, 1-based), ``h``, ``n_pairs``,
        ``correction``, ``n``.
    """
    x = [float(v) for v in y]
    n = len(x)
    if n < 2:
        raise ValueError("Qn needs at least two observations")
    h = n // 2 + 1
    k = h * (h - 1) // 2
    diffs = []
    for i in range(n - 1):
        xi = x[i]
        for j in range(i + 1, n):
            dv = xi - x[j]
            diffs.append(dv if dv >= 0.0 else -dv)
    diffs.sort()
    raw = diffs[k - 1]
    est = float(constant) * raw
    if finite_corr:
        corr = _QN_SMALL[n - 2] if n <= 12 else 1.0 / _qn_finite_c(n)
    else:
        corr = 1.0
    est = est * corr
    return RichResult(payload={
        "estimate": float(est), "raw": float(raw), "k": k, "h": h,
        "n_pairs": n * (n - 1) // 2, "correction": float(corr),
        "constant": float(constant), "n": n,
        "method": "Rousseeuw & Croux (1993) Qn, robustbase qnsn.R definition"})


def cheatsheet():
    return "qnscl: Rousseeuw & Croux (1993) Qn robust scale"
