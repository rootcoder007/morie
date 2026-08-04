# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov-Smirnov statistic against a fully specified distribution."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ksstat", "fauzi_ks_statistic"]


def ksstat(x, cdf):
    r"""Kolmogorov-Smirnov statistic against a fully specified distribution.

    The classical statistic the chapter's smoothed versions are compared
    against:

    .. math:: KS_n = \sup_{x\in\mathbb R}|F_n(x) - F(x)|,

    with :math:`F_n` the empirical distribution function.

    Computed as :math:`\max(D^+, D^-)` over the order statistics, which is
    exact -- the supremum of a step function against a continuous one is
    always attained at a jump, so no grid search is needed and none is
    done.

    Sec. 5.1 gives the motivation for replacing :math:`F_n` here: its lack
    of smoothness makes the test over-sensitive near the centre of the
    distribution and inflates the type-I error above the nominal
    :math:`\alpha` at small ``n``. Theorems 5.1 and 5.6 then show the
    smoothed replacements have the SAME limit, so the same critical values
    apply.

    Uses the exact one-sample Kolmogorov distribution for ``n <= 40`` and
    the standard asymptotic series otherwise, so the p-value is usable at
    the small sample sizes the chapter is about.

    Parameters
    ----------
    x : array-like
        Sample.
    cdf : callable
        The fully specified null distribution ``F(t)``.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``dplus``, ``dminus``, ``p_value``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 5.1, the display preceding (5.3).
    """
    from . import _stats_core as stats

    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if not callable(cdf):
        raise ValueError("cdf must be a callable F(t).")
    fv = np.asarray([float(cdf(float(t))) for t in xs], dtype=float)
    dplus = float(np.max(np.arange(1, n + 1) / n - fv))
    dminus = float(np.max(fv - np.arange(0, n) / n))
    stat = max(dplus, dminus)
    if n <= 40:
        pval = float(1.0 - stats.ksone.cdf(stat, n))
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * stat
        pval = 2.0 * float(
            np.sum([(-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)])
        )
        pval = max(0.0, min(1.0, pval))
    return RichResult(
        payload={
            "statistic": float(stat),
            "dplus": dplus,
            "dminus": dminus,
            "p_value": float(pval),
            "n": int(n),
            "method": "Kolmogorov-Smirnov statistic against a specified F",
        }
    )


fauzi_ks_statistic = ksstat


def cheatsheet():
    return "fzksst: classical KS statistic, exact at the order statistics -- the Ch 5 baseline"


# CANONICAL TEST
# >>> r = ksstat([0.1, 0.3, 0.5, 0.7, 0.9], cdf=lambda t: min(max(t, 0.0), 1.0))
# >>> abs(r['statistic'] - 0.1) < 1e-12
# True
