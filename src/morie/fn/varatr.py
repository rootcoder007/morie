# morie.fn -- function file (rootcoder007/morie)
"""Value at Risk and expected shortfall from a GARCH(1,1) volatility.

SOURCE.  Jorion, P. (2007), *Value at Risk: The New Benchmark for
Managing Financial Risk*, 3rd edition, McGraw-Hill, ISBN
978-0-07-146495-6.  The parametric ("delta-normal") VaR at confidence
level 1-alpha over one period is the alpha-quantile of the return
distribution taken with the sign flipped, so that a loss is reported as
a positive number:

    VaR_alpha = -(mu + sigma * z_alpha) = -mu + sigma * z_{1-alpha}

with z_alpha = Phi^{-1}(alpha).  The Gaussian expected shortfall that
goes with it is ES_alpha = -mu + sigma * phi(z_alpha)/alpha.

The conditional sigma is the one-step-ahead GARCH(1,1) standard
deviation of Bollerslev, T. (1986), "Generalized autoregressive
conditional heteroskedasticity", *Journal of Econometrics*
31(3):307-327, doi:10.1016/0304-4076(86)90063-1, Eq. (1)-(2):

    sigma^2_t = omega + a e^2_{t-1} + b sigma^2_{t-1},   e_t = y_t - mu.

VARIANCE TARGETING.  omega is not free: it is pinned to the sample
unconditional variance s^2 by omega = s^2 (1 - a - b), which is the
stationary-variance identity of Bollerslev Eq. (5) solved for omega.
That removes one parameter and guarantees the fitted process has the
sample variance.  a and b are then chosen by exhaustive search over a
fixed lattice maximising the Gaussian quasi-likelihood

    log L = -(1/2) sum_t [log 2 pi + log sigma^2_t + e^2_t / sigma^2_t].

A lattice rather than a numerical optimiser is what makes the two
language arms land on identical numbers; the lattice is stated in the
signature and is this implementation's choice, not a quotation.

Jorion (2007) is not in the local corpus and the delta-normal formula
above is the standard statement.  The module is anchored on the
degenerate a = b = 0 case, where sigma is the sample standard deviation
and VaR collapses to a closed form in ``qnorm``, computed by a different
route.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["value_at_risk"]

_AGRID = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
_BGRID = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90)


def _qml(e, s2u, a, b):
    n = len(e)
    om = s2u * (1.0 - a - b)
    s2 = s2u
    ll = 0.0
    path = []
    for t in range(n):
        if t > 0:
            s2 = om + a * e[t - 1] * e[t - 1] + b * s2
        if not (s2 > 0.0):
            return float("-inf"), [], 0.0
        ll += -0.5 * (math.log(2.0 * math.pi) + math.log(s2) + e[t] * e[t] / s2)
        path.append(s2)
    nxt = om + a * e[n - 1] * e[n - 1] + b * s2
    return ll, path, nxt


def value_at_risk(y, alpha=0.05, a_grid=None, b_grid=None):
    """One-period VaR and expected shortfall from a GARCH(1,1) fit.

    Parameters
    ----------
    y : array-like
        Return series.
    alpha : float
        Tail probability, strictly between 0 and 1.  ``alpha = 0.05``
        is the 95% VaR.
    a_grid, b_grid : sequence of float or None
        Lattices searched for the ARCH and GARCH coefficients.  Only
        pairs with ``a + b < 1`` are admissible.

    Returns
    -------
    RichResult
        ``estimate`` (= ``var``), ``var``, ``es``, ``mu``, ``sigma``,
        ``sigma2_next``, ``omega``, ``a``, ``b``, ``loglik``,
        ``sigma2_path``, ``z``, ``alpha``, ``n``.

    Raises
    ------
    ValueError
        Empty series, fewer than two observations, alpha outside (0, 1),
        a degenerate (zero-variance) series, or an empty lattice.

    References
    ----------
    Jorion, P. (2007).  Value at Risk, 3rd ed.  McGraw-Hill.
    Bollerslev, T. (1986).  Journal of Econometrics 31(3):307-327.
    doi:10.1016/0304-4076(86)90063-1.
    """
    yv = core.vec(y)
    n = len(yv)
    if n < 2:
        raise ValueError("value_at_risk: need at least two observations")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("value_at_risk: alpha must lie strictly in (0, 1)")
    mu = 0.0
    for v in yv:
        mu += v
    mu /= n
    e = [v - mu for v in yv]
    s2u = 0.0
    for v in e:
        s2u += v * v
    s2u /= n
    if not (s2u > 0.0):
        raise ValueError("value_at_risk: series has zero variance")
    ag = tuple(_AGRID if a_grid is None else [float(v) for v in a_grid])
    bg = tuple(_BGRID if b_grid is None else [float(v) for v in b_grid])
    if not ag or not bg:
        raise ValueError("value_at_risk: coefficient lattice is empty")
    best = None
    for a in ag:
        for b in bg:
            if a < 0.0 or b < 0.0 or a + b >= 1.0:
                continue
            ll, path, nxt = _qml(e, s2u, a, b)
            if ll == float("-inf"):
                continue
            if best is None or ll > best[0]:
                best = (ll, a, b, path, nxt)
    if best is None:
        raise ValueError("value_at_risk: no admissible (a, b) on the lattice")
    ll, a, b, path, s2next = best
    om = s2u * (1.0 - a - b)
    sigma = math.sqrt(s2next)
    z = core.qnorm(alpha)
    var = -(mu + sigma * z)
    dens = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    es = -mu + sigma * dens / alpha
    return RichResult(
        title="GARCH(1,1) Value at Risk",
        summary_lines=[("obs", n), ("alpha", alpha), ("VaR", var)],
        payload={
            "estimate": var,
            "var": var,
            "es": es,
            "mu": mu,
            "sigma": sigma,
            "sigma2_next": s2next,
            "omega": om,
            "a": a,
            "b": b,
            "loglik": ll,
            "sigma2_path": path,
            "z": z,
            "alpha": alpha,
            "n": n,
            "method": "Delta-normal VaR on a GARCH(1,1) conditional sigma, variance targeting (Jorion 2007; Bollerslev 1986)",
        },
    )


def cheatsheet():
    return "varatr: GARCH(1,1) Value at Risk and expected shortfall (Jorion 2007)"
