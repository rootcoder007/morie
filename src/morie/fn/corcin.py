"""Pairs needed for a confidence interval of given limits for a correlation."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["correlation_ci_sample_size"]


def correlation_ci_sample_size(lower, upper, conf_level=0.95):
    r"""Pairs needed so the z-interval for :math:`\rho` spans ``lower`` to ``upper``.

    Hedderich, Sachs & Reynarowych (2023, eq 6.155): the interval
    :math:`\dot z \pm z_{1-\alpha/2}/\sqrt{n-3}` has width
    :math:`\dot z_o - \dot z_u`, so

    .. math::  n = 4\left(\frac{z_{1-\alpha/2}}{\tanh^{-1}\varrho_o - \tanh^{-1}\varrho_u}\right)^2 + 3.

    Parameters
    ----------
    lower, upper : float
        Required limits, -1 < lower < upper < 1.
    conf_level : float

    Returns
    -------
    RichResult
        ``n`` (rounded up) and ``n_exact``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eq (6.155).
    """
    lower, upper = float(lower), float(upper)
    if not -1 < lower < upper < 1:
        raise ValueError("need -1 < lower < upper < 1")
    q = float(norm.ppf(1 - (1 - conf_level) / 2))
    ne = 4 * (q / (math.atanh(upper) - math.atanh(lower))) ** 2 + 3
    return RichResult(
        title="Sample size for a correlation CI",
        summary_lines=[("n", math.ceil(ne))],
        payload={"n": math.ceil(ne), "n_exact": ne, "lower": lower, "upper": upper},
    )


def cheatsheet():
    return "corcin: n = 4 (z_{1-a/2} / (atanh upper - atanh lower))^2 + 3"
