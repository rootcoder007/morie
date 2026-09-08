# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric Bayes survival via a beta process.

DUPLICATE: Hjort's beta-process posterior for right-censored survival
is already implemented in ``ghsrv`` (public name
``ghosal_survival_beta_process``), which has an R arm.  This module
aliases it rather than carrying a second copy.
"""

from .ghsrv import ghosal_survival_beta_process as _bp

__all__ = ["np_bayes_survival"]


def np_bayes_survival(time, event=None, c=1.0, lam0=None):
    """Posterior-mean survival under a beta-process prior on the hazard.

    Alias of :func:`morie.fn.ghsrv.ghosal_survival_beta_process`.

    Formula: with ``H ~ BP(c, H_0)`` the posterior is again a beta
    process and ``dH_post(t) = (c dH_0(t) + dN(t)) / (c + Y(t-))``, so
    ``S_hat(t) = prod_{s <= t} (1 - dH_post(s))``.  As ``c -> 0`` this
    becomes the Kaplan-Meier estimator.

    Parameters
    ----------
    time : array-like
        Observation times, possibly censored.
    event : array-like or None
        1 = event, 0 = censored; all events if ``None``.
    c : float, default 1.0
        Prior concentration.
    lam0 : float or None
        Exponential base hazard rate; ``1 / mean(time)`` if ``None``.

    Returns
    -------
    RichResult
        ``estimate`` (survival at the median time), ``times``,
        ``S_post``, ``H_post``, ``c``, ``lam0``, ``n``.

    References
    ----------
    Hjort, N. L. (1990).  Nonparametric Bayes estimators based on beta
    processes in models for life history data.  Annals of Statistics,
    18(3), 1259--1294.  doi:10.1214/aos/1176347749
    """
    return _bp(time, event=event, c=c, lam0=lam0)


def cheatsheet():
    return "npbsr: Beta-process survival (alias of ghsrv)"


npbayessurvival = np_bayes_survival
