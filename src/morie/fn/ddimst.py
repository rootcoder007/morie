# morie.fn -- function file (rootcoder007/morie)
"""One DDIM reverse step."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["ddim_step"]


def alpha_bar_cosine(t, T):
    """Nichol-Dhariwal cosine schedule for the cumulative alpha."""
    f = lambda u: math.cos((u / T + 0.008) / 1.008 * math.pi / 2.0) ** 2
    return f(t) / f(0.0)


def ddim_step(x_t, t, eps_theta, eta=0.0, T=1000, alpha_bar_t=None,
              alpha_bar_prev=None):
    """
    One DDIM reverse step

    Formula: non-Markovian deterministic reverse

    x_{t-1} = sqrt(a_{t-1}) x0_hat
              + sqrt(1 - a_{t-1} - sigma^2) eps_theta
              + sigma z,
    with x0_hat = (x_t - sqrt(1 - a_t) eps_theta) / sqrt(a_t) and
    sigma = eta sqrt((1 - a_{t-1})/(1 - a_t)) sqrt(1 - a_t/a_{t-1}).
    At eta = 0 the update is fully deterministic (that is DDIM); at
    eta = 1 it reproduces the DDPM ancestral step.  The noise term z is
    taken as zero here so the map is a function of its arguments alone.

    Parameters
    ----------
    x_t : array-like
        Current latent.
    t : int
        Current timestep, at least 1.
    eps_theta : array-like
        Predicted noise, same shape as x_t.
    eta : float
        Stochasticity in [0, 1].
    T : int
        Total number of steps for the cosine schedule.
    alpha_bar_t, alpha_bar_prev : float or None
        Override the schedule with explicit cumulative alphas.

    Returns
    -------
    result : dict
        Keys: estimate (mean of x_{t-1}), x_prev, x0_pred, sigma,
        alpha_bar_t, alpha_bar_prev, n.

    References
    ----------
    Song, Meng & Ermon (2021), Denoising Diffusion Implicit Models,
    ICLR 2021.
    """
    x = core.vec(x_t)
    e = core.vec(eps_theta)
    n = len(x)
    if n == 0:
        raise ValueError("empty input: x_t has no entries")
    if len(e) != n:
        raise ValueError("x_t and eps_theta must have the same length")
    t = int(t)
    if t < 1:
        raise ValueError("t must be at least 1")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    at = alpha_bar_cosine(t, T) if alpha_bar_t is None else float(alpha_bar_t)
    ap = alpha_bar_cosine(t - 1, T) if alpha_bar_prev is None \
        else float(alpha_bar_prev)
    if not (0.0 < at <= 1.0 and 0.0 < ap <= 1.0):
        raise ValueError("cumulative alphas must lie in (0, 1]")
    sigma = eta * math.sqrt((1.0 - ap) / (1.0 - at)) * \
        math.sqrt(max(1.0 - at / ap, 0.0))
    x0 = [(x[i] - math.sqrt(1.0 - at) * e[i]) / math.sqrt(at) for i in range(n)]
    c = math.sqrt(max(1.0 - ap - sigma * sigma, 0.0))
    xp = [math.sqrt(ap) * x0[i] + c * e[i] for i in range(n)]
    return RichResult(payload={
        "estimate": sum(xp) / n,
        "x_prev": xp,
        "x0_pred": x0,
        "sigma": sigma,
        "alpha_bar_t": at,
        "alpha_bar_prev": ap,
        "n": n,
        "method": "DDIM reverse step",
    })


def cheatsheet():
    return "ddimst: DDIM reverse step"


# compact alias per ledger/NAMING.md
ddimstep = ddim_step
