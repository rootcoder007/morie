# morie.fn -- function file (rootcoder007/morie)
"""Wald test for a scalar parameter."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["waldstat", "wasserman_wald"]


def waldstat(theta_hat, se, theta0=0.0, level=0.95):
    """Wald test of H0: theta = theta0 against a two-sided alternative.

    The test is asymptotic in one specific way worth naming: it assumes
    (theta_hat - theta0)/se is standard Normal UNDER THE NULL, with se
    evaluated at the estimate.  That is why the Wald test and the
    likelihood ratio test can disagree in finite samples even though
    they agree asymptotically, and why the Wald test is the one that
    misbehaves near a boundary of the parameter space.

    Formula: W = (theta_hat - theta0)/se; reject when |W| > z_{alpha/2};
             p = 2(1 - Phi(|W|))

    Parameters
    ----------
    theta_hat : float
        The estimate.
    se : float
        Its estimated standard error, se > 0.
    theta0 : float
        The null value.
    level : float
        Confidence level for the returned interval.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``estimate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``z_critical``, ``reject``.

    References
    ----------
    Wasserman (2004), All of Statistics, Definition 10.3 and equation
    (10.5): "the size alpha Wald test is: reject H0 when |W| > z_alpha/2
    where W = (theta_hat - theta_0)/se_hat", with Theorem 10.4 giving
    the asymptotic size.  Fetched as the full text of the book.
    """
    se = float(se)
    if se <= 0:
        raise ValueError("the standard error must be positive")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    theta_hat = float(theta_hat)
    theta0 = float(theta0)
    W = (theta_hat - theta0) / se
    p = 2.0 * (1.0 - C.pnorm(abs(W)))
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "statistic": W, "p_value": p, "estimate": theta_hat, "se": se,
        "ci_lower": theta_hat - z * se, "ci_upper": theta_hat + z * se,
        "z_critical": z, "reject": 1.0 if abs(W) > z else 0.0,
        "method": "Wald test, Wasserman Definition 10.3"})


wasserman_wald = waldstat


def cheatsheet():
    return "wsmwld: W = (theta_hat - theta0)/se; p = 2(1 - Phi(|W|))"
