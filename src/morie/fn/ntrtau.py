"""Sample size for testing Kendall's tau, Noether's approximation."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["noether_tau_sample_size"]


def noether_tau_sample_size(tau, alpha=0.05, power=0.8):
    r"""Pairs needed to detect Kendall's :math:`\tau` (two-sided), Noether (1987).

    Hedderich, Sachs & Reynarowych (2023, eq 7.415): with the probability
    of concordance :math:`\pi_c = (1 + \tau)/2`,

    .. math::  n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{9(\pi_c - 1/2)^2}.

    Parameters
    ----------
    tau : float
        Kendall's tau to detect, non-zero.
    alpha : float
    power : float

    Returns
    -------
    RichResult
        ``n`` (rounded up), ``n_exact``.

    References
    ----------
    Noether, G. E. (1987). Sample size determination for some common
    nonparametric tests. JASA 82, 645-647.
    """
    tau = float(tau)
    if tau == 0 or not -1 < tau < 1:
        raise ValueError("`tau` must be non-zero and in (-1, 1)")
    zs = float(norm.ppf(1 - alpha / 2)) + float(norm.ppf(power))
    ne = zs * zs / (9 * ((1 + tau) / 2 - 0.5) ** 2)
    return RichResult(
        title="Noether sample size for Kendall's tau",
        summary_lines=[("n", math.ceil(ne))],
        payload={"n": math.ceil(ne), "n_exact": ne, "tau": tau, "alpha": alpha, "power": power},
    )


def cheatsheet():
    return "ntrtau: n = (z_{1-a/2} + z_{1-b})^2 / (9 (pi_c - 1/2)^2), pi_c = (1 + tau)/2"
