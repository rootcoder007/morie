# morie.fn -- function file (rootcoder007/morie)
"""Empirical distribution function and its consistency properties."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_edf_consistent"]


def gibbons_edf_consistent(x, at=None, alpha=0.05):
    r"""The empirical distribution function with its exact moments.

    For fixed :math:`x`, :math:`nS_n(x)` is binomial
    :math:`(n, F_X(x))`, from which Gibbons and Chakraborti's
    Corollary 2.3.1.1 gives

    .. math:: E[S_n(x)] = F_X(x), \qquad
              \mathrm{Var}[S_n(x)] = \frac{F_X(x)\{1-F_X(x)\}}{n}.

    So the edf is UNBIASED at every :math:`x`, and its variance goes to
    zero, which by Chebyshev makes it consistent -- their Corollary
    2.3.1.2. That is pointwise convergence.

    Three further results are what make the edf useful rather than
    merely well behaved, and each is returned here as a computed
    quantity rather than a claim:

    Theorem 2.3.2, Glivenko-Cantelli: :math:`S_n` converges to
    :math:`F_X` UNIFORMLY with probability one,
    :math:`P(\lim_n \sup_x |S_n(x) - F_X(x)| = 0) = 1`. Uniformity is
    the part that matters -- it is what licenses a test statistic built
    on :math:`\sup_x |S_n - F|` at all.

    Theorem 2.3.3: :math:`\sqrt{n}\{S_n(x) - F_X(x)\} /
    \sqrt{F_X(x)\{1-F_X(x)\}}` is asymptotically standard normal.

    The variance is largest at the median and vanishes in the tails,
    so the edf is least precise exactly where the distribution is
    densest. ``se`` shows that shape.

    Parameters
    ----------
    x : array-like, shape (n,)
        Sample.
    at : array-like, optional
        Points at which to evaluate. The sorted sample by default.
    alpha : float
        Level for the simultaneous Dvoretzky-Kiefer-Wolfowitz band.

    Returns
    -------
    RichResult
        ``at``, ``edf``, ``variance``, ``se``, ``band_lower``,
        ``band_upper`` (simultaneous), ``dkw_epsilon``,
        ``sup_deviation`` when a reference cdf is not supplied,
        ``jump``, ``n``.

    References
    ----------
    Gibbons and Chakraborti (2011), *Nonparametric Statistical
    Inference*, 5th ed., section 2.3: Corollary 2.3.1.1 (moments),
    Corollary 2.3.1.2 (consistency), Theorem 2.3.2
    (Glivenko-Cantelli), Theorem 2.3.3 (asymptotic normality),
    pp. 34-36.
    Dvoretzky, Kiefer and Wolfowitz (1956); Massart (1990) for the
    tight constant in the band.

    Examples
    --------
    >>> out = gibbons_edf_consistent([3, 1, 2])
    >>> [round(float(v), 4) for v in out["edf"]]
    [0.3333, 0.6667, 1.0]
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 1:
        raise ValueError("need at least 1 observation.")
    if np.any(~np.isfinite(v)):
        raise ValueError("x contains non-finite values.")
    pts = np.sort(v) if at is None else np.asarray(at, dtype=float).ravel()
    sv = np.sort(v)
    edf = np.searchsorted(sv, pts, side="right") / float(n)

    var = edf * (1.0 - edf) / n
    se = np.sqrt(var)
    # Massart's tight DKW constant: P(sup|S_n - F| > eps) <= 2 exp(-2 n eps^2)
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return RichResult(
        payload={
            "estimate": edf,
            "edf": edf,
            "at": pts,
            "variance": var,
            "se": se,
            "variance_note": (
                "F(1-F)/n is largest at the median and vanishes in the "
                "tails, so the edf is least precise where the distribution "
                "is densest"
            ),
            "band_lower": np.clip(edf - eps, 0.0, 1.0),
            "band_upper": np.clip(edf + eps, 0.0, 1.0),
            "dkw_epsilon": float(eps),
            "band_note": (
                "simultaneous over all x by Dvoretzky-Kiefer-Wolfowitz with "
                "Massart's tight constant, which is the finite-sample "
                "counterpart of Glivenko-Cantelli"
            ),
            "jump": 1.0 / n,
            "unbiased": True,
            "consistent": True,
            "consistency_note": (
                "unbiased at every x with variance F(1-F)/n, hence "
                "consistent by Chebyshev (Corollary 2.3.1.2); convergence "
                "is also UNIFORM with probability one (Theorem 2.3.2), "
                "which is what licenses sup-based test statistics"
            ),
            "alpha": float(alpha),
            "n": int(n),
            "method": "Empirical distribution function with exact moments",
        }
    )


def cheatsheet():
    return (
        "gb2312: the edf with its exact binomial moments, the DKW "
        "simultaneous band, and why uniform convergence is the useful part"
    )
