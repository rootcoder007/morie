# morie.fn -- function file (rootcoder007/morie)
"""Bayesian online changepoint detection (re-export)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bayesocp', 'bayesian_online_changepoint']


def bayesocp(y, hazard=0.004, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Bayesian online changepoint detection (re-export).

    The same run-length recursion as morie.fn.bocpd. This module exists because the shelf lists the method twice under different names; keeping one implementation and one delegating name is preferable to two copies that drift.


    Formula: see bocpd

    Parameters
    ----------
    y : array-like
        Observed univariate series.
    hazard : float
        Constant hazard of the geometric run-length prior.
    mu0 : float
        Prior mean.
    kappa0 : float
        Prior mean precision.
    alpha0 : float
        Prior shape.
    beta0 : float
        Prior scale.

    Returns
    -------
    RichResult
        the payload of :func:`morie.fn.bocpd.bocpd`.

    References
    ----------
    Adams and MacKay (2007), Bayesian Online Changepoint Detection,
    arXiv:0710.3742.  Equations (2)-(5) for the recursion and the
    changepoint prior, Section 2.3 and Algorithm 1 for the
    conjugate-exponential update of the run-specific sufficient
    statistics.  Verified against the paper.
    """
    from .bocpd import bocpd as _bocpd
    return _bocpd(y, hazard=hazard, mu0=mu0, kappa0=kappa0,
                  alpha0=alpha0, beta0=beta0)


bayesian_online_changepoint = bayesocp


def cheatsheet():
    return "chgbcp: Bayesian online changepoint detection (re-export)."
