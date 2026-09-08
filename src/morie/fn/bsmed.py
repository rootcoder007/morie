# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bootstrap percentile CI for the indirect effect in simple mediation."""

from . import _array_core as np

from ._richresult import RichResult
from ._rrng_core import RRandom

__all__ = ["bsmed", "bootstrap_mediation_ci"]


def _ab(x, m, y):
    # a from OLS of M on (1, X); b, c-prime from OLS of Y on (1, X, M).
    n = len(x)
    one = np.ones(n)
    Xa = np.stack([one, x], axis=1)
    ca = np.linalg.solve(Xa.T @ Xa, Xa.T @ m)
    Xb = np.stack([one, x, m], axis=1)
    cb = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return float(ca[1]), float(cb[2]), float(cb[1])


def bsmed(x, m, y, B=1000, alpha=0.05, seed=1):
    """
    Bootstrap percentile confidence interval for the indirect effect
    a*b in the simple mediation model M = i1 + a X, Y = i2 + c' X + b M.

    Procedure of Preacher and Hayes (2004), p. 722: draw B resamples of
    the n cases with replacement, recompute a*b in each, sort the B
    estimates, and take as limits the (B * alpha/2)-th and the
    (B * (1 - alpha/2) + 1)-th order statistics (their worked rule: for
    B = 1000 and alpha = .05, the 25th and 976th sorted values). The
    bootstrap point estimate is the mean of the B resampled a*b (same
    page); the sample-data a*b is reported alongside. Percentile
    intervals for indirect effects follow Shrout and Bolger (2002).

    Resampling uses the R-compatible Mersenne-Twister stream
    (``set.seed(seed)`` + ``sample.int(n, n, replace = TRUE)``), so the
    R arm reproduces the same resamples index for index.

    Parameters
    ----------
    x, m, y : array-like
        Independent variable, mediator, outcome (equal length n).
    B : int
        Number of bootstrap resamples (default 1000).
    alpha : float
        Two-sided miss probability (default 0.05).
    seed : int
        Seed for the R-compatible RNG.

    Returns
    -------
    result : RichResult
        Keys: estimate (sample a*b), boot_estimate (mean of resampled
        a*b), se (sd of the bootstrap distribution), ci_lower,
        ci_upper, a, b, c_prime, B, n, conf_level.

    References
    ----------
    Preacher, K. J. and Hayes, A. F. (2004), "SPSS and SAS procedures
    for estimating indirect effects in simple mediation models",
    Behavior Research Methods, Instruments, and Computers 36(4),
    717-731, doi:10.3758/BF03206553; bootstrap procedure p. 722.
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/
    fetched-wave3/preacher-hayes-2004-spss-sas-indirect-effects-BRM36.pdf
    Shrout, P. E. and Bolger, N. (2002), "Mediation in experimental and
    nonexperimental studies: New procedures and recommendations",
    Psychological Methods 7(4), 422-445, doi:10.1037/1082-989X.7.4.422.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if len(m) != n or len(y) != n:
        raise ValueError("x, m, y must have equal length")
    B = int(B)
    if B < 2:
        raise ValueError("B must be at least 2")
    a, b, c_prime = _ab(x, m, y)
    rng = RRandom(seed)
    boots = []
    for _ in range(B):
        idx = [i - 1 for i in rng.sample_int(n, n, replace=True)]
        ar, br, _cr = _ab(
            np.asarray([x[i] for i in idx]),
            np.asarray([m[i] for i in idx]),
            np.asarray([y[i] for i in idx]),
        )
        boots.append(ar * br)
    s = sorted(boots)
    lo_i = int(B * (alpha / 2.0))          # 1-based rank, PH2004 p.722
    hi_i = int(B * (1.0 - alpha / 2.0)) + 1
    lo_i = min(max(lo_i, 1), B)
    hi_i = min(max(hi_i, 1), B)
    bmean = float(np.mean(np.asarray(boots)))
    bse = float(np.std(np.asarray(boots), ddof=1))
    return RichResult(payload={
        "estimate": a * b,
        "boot_estimate": bmean,
        "se": bse,
        "ci_lower": float(s[lo_i - 1]),
        "ci_upper": float(s[hi_i - 1]),
        "a": a, "b": b, "c_prime": c_prime,
        "B": B, "n": n, "conf_level": 1.0 - alpha,
        "method": "Preacher-Hayes (2004) bootstrap percentile CI for a*b",
    })


bootstrap_mediation_ci = bsmed


def cheatsheet():
    return "bsmed(x, m, y, B, alpha, seed) -> percentile bootstrap CI for the indirect effect a*b."
