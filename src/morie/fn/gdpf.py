# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gaussian differential privacy: the trade-off function.

Dong, Roth and Su (2022), "Gaussian differential privacy", JRSS B
84(1):3-37, doi:10.1111/rssb.12454 (arXiv:1905.02383, 2019).  Privacy
is expressed as the trade-off between the two error rates of the
hypothesis test that distinguishes neighbouring databases; a mechanism
is mu-GDP when that trade-off is at least that of testing N(0, 1)
against N(mu, 1),

    G_mu(alpha) = Phi( Phi^{-1}(1 - alpha) - mu ).

Their Corollary 2.13 converts to the (eps, delta) language:

    delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) Phi(-eps/mu - mu/2).

mu = 0 gives G(alpha) = 1 - alpha, perfect privacy, and delta = 0 for
every eps; both limits are exact and are what the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gaussian_dp"]


def gaussian_dp(mech, mu, alpha=None, epsilon=1.0):
    """Trade-off curve of a mu-GDP mechanism and its (eps, delta) profile.

    Parameters
    ----------
    mech : array-like or None
        Placeholder for the mechanism description; only mu is used.
    mu : float
        The GDP parameter, non-negative.
    alpha : array-like, optional
        Type I error rates at which the trade-off is evaluated.
    epsilon : float
        Epsilon at which the (eps, delta) conversion is reported.
    """
    m = float(mu)
    if m < 0:
        raise ValueError("gaussian_dp: mu must be non-negative")
    a = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9] if alpha is None else core.vec(alpha)
    if len(a) == 0:
        raise ValueError("gaussian_dp: alpha is empty")
    for v in a:
        if not 0 < v < 1:
            raise ValueError("gaussian_dp: every alpha must lie in (0, 1)")
    e = float(epsilon)
    trade = [core.pnorm(core.qnorm(1.0 - v) - m) for v in a]
    if m == 0.0:
        delta = 0.0
    else:
        delta = core.pnorm(-e / m + m / 2.0) - math.exp(e) * core.pnorm(-e / m - m / 2.0)
    if delta < 0:
        delta = 0.0
    return RichResult(
        title="Gaussian differential privacy",
        summary_lines=[("mu", m), ("epsilon", e), ("delta", delta)],
        payload={
            "estimate": delta,
            "trade_off": trade,
            "alpha": list(a),
            "delta": delta,
            "mu": m,
            "epsilon": e,
            "n": len(a),
            "method": "G_mu(alpha) = Phi(Phi^{-1}(1-alpha) - mu) with the (eps, delta) dual of Corollary 2.13, Dong, Roth & Su (2022)",
        },
    )


def cheatsheet():
    return "gdpf: Gaussian differential privacy trade-off"


# compact alias per ledger/NAMING.md
gaussiandp = gaussian_dp
