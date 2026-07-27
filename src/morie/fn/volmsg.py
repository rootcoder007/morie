# morie.fn -- function file (rootcoder007/morie)
"""Markov-switching GARCH."""

from ._garch import ms_garch_fit
from ._richresult import RichResult

__all__ = ["vol_markov_switching_garch"]


def vol_markov_switching_garch(r, K=2):
    r"""Markov-switching GARCH with regime-specific (omega, alpha, beta).

    An exact MS-GARCH likelihood is intractable because the variance
    recursion depends on the entire unobserved regime path. This uses
    Gray's (1996) collapsing step -- the variance carried into the
    next period is the filtered-probability-weighted average across
    regimes -- which keeps the state finite-dimensional at the cost of
    being an approximation, stated here rather than glossed.

    Regimes are returned sorted by unconditional variance, so index 0
    is always the calm regime and the labels do not permute between
    runs.

    Parameters
    ----------
    r : array-like
        Return series, at least 100 observations.
    K : int, default 2
        Number of regimes.

    Returns
    -------
    RichResult
        keys: ``params`` (list of per-regime dicts), ``transition``
        (K x K), ``unconditional_var``, ``loglik``, ``n_regimes``,
        ``converged``, ``n``, ``method``.

    References
    ----------
    Gray, S. F. (1996). Modeling the conditional distribution of
    interest rates as a regime-switching process. *Journal of
    Financial Economics*, 42(1), 27-62.

    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*,
    57(2), 357-384.
    """
    return RichResult(payload=ms_garch_fit(r, K))


def cheatsheet():
    return "volmsg: MS-GARCH, Gray (1996) collapsed recursion, regimes sorted by variance"
