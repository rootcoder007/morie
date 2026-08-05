# morie.fn -- function file (rootcoder007/morie)
"""Peaks-over-threshold GPD fit by maximum likelihood."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_pot_fit"]


def gpd_nll(y, sigma, xi):
    """Negative log-likelihood of the GPD at (sigma, xi) for excesses y."""
    n = len(y)
    if sigma <= 0.0:
        return float("inf")
    if abs(xi) < 1e-12:
        return n * math.log(sigma) + sum(y) / sigma
    s = 0.0
    for v in y:
        z = 1.0 + xi * v / sigma
        if z <= 0.0:
            return float("inf")
        s += math.log(z)
    return n * math.log(sigma) + (1.0 / xi + 1.0) * s


def _profile(y, t):
    """Grimshaw's reparametrisation: given t = xi/sigma, xi and sigma follow."""
    n = len(y)
    s = 0.0
    for v in y:
        z = 1.0 + t * v
        if z <= 0.0:
            return None
        s += math.log(z)
    xi = s / n
    if xi == 0.0:
        return None
    return xi, xi / t, math.log(abs(t)) - math.log(abs(xi)) - xi


def evt_pot_fit(x, u):
    """
    Peaks-over-threshold GPD fit + scale-invariance check

    Formula: fit GPD on x[x>u]; rate zeta_u = N_u/n

    Maximum likelihood in Grimshaw's one-parameter reduction: with
    t = xi/sigma the profile has xi(t) = mean(log(1 + t y)) and
    sigma(t) = xi(t)/t, leaving a single bounded search.  The
    exponential (xi = 0) fit, sigma = mean(y), is evaluated separately
    and kept if its likelihood is higher.  The modified scale
    sigma - xi u is reported: it is invariant to the threshold when the
    GPD model holds, which is the check the fit is meant to support.

    Parameters
    ----------
    x : array-like
        Sample.
    u : float
        Threshold.

    Returns
    -------
    result : dict
        Keys: sigma, xi, zeta_u, estimate, n_exceed, n, nll,
        modified_scale.

    References
    ----------
    Davison & Smith (1990), JRSS B 52(3):393-442.
    Grimshaw (1993), Technometrics 35(2):185-191.
    """
    x = core.vec(x)
    n = len(x)
    if n == 0:
        raise ValueError("empty input: x has no observations")
    u = float(u)
    y = sorted(v - u for v in x if v > u)
    k = len(y)
    if k < 2:
        raise ValueError("fewer than two exceedances above u; nothing to fit")
    ymax = y[-1]
    ybar = sum(y) / k
    # exponential reference fit
    best = (gpd_nll(y, ybar, 0.0), ybar, 0.0)
    # search t over (-1/ymax, 0) and (0, tmax) on a deterministic grid,
    # then bisect on the derivative-free golden section within the best cell
    lo = -1.0 / ymax + 1e-10
    hi = 4.0 / ybar
    grid = 4000
    cells = []
    for i in range(grid + 1):
        t = lo + (hi - lo) * i / grid
        if abs(t) < 1e-12:
            continue
        pr = _profile(y, t)
        if pr is None:
            continue
        cells.append((pr[2], t))
    if cells:
        cells.sort()
        g0, t0 = cells[-1]
        step = (hi - lo) / grid
        a, b = t0 - step, t0 + step
        gr = 0.5 * (math.sqrt(5.0) - 1.0)
        for _ in range(200):
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            pc = _profile(y, c)
            pd = _profile(y, d)
            fc = pc[2] if pc else -1e300
            fd = pd[2] if pd else -1e300
            if fc > fd:
                b = d
            else:
                a = c
        t = 0.5 * (a + b)
        pr = _profile(y, t)
        if pr is not None:
            nll = gpd_nll(y, pr[1], pr[0])
            if nll < best[0]:
                best = (nll, pr[1], pr[0])
    nll, sigma, xi = best
    zeta = k / float(n)
    return RichResult(payload={
        "sigma": sigma,
        "xi": xi,
        "zeta_u": zeta,
        "estimate": xi,
        "n_exceed": k,
        "n": n,
        "nll": nll,
        "modified_scale": sigma - xi * u,
        "method": "GPD maximum likelihood on threshold exceedances",
    })


def cheatsheet():
    return "evpot: peaks-over-threshold GPD fit"


# compact alias per ledger/NAMING.md
evtpotfit = evt_pot_fit
