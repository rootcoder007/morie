# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Parametric bootstrap: simulate from the fitted model, not the data.

Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and their
Application*, Cambridge University Press, chapter 2 (the parametric
simulation is their equation for F_hat = F(. | theta_hat)).

Where the nonparametric bootstrap resamples the empirical distribution,
the parametric bootstrap draws fresh samples from the fitted member of
the assumed family:

    x*_b ~ F( . | theta_hat ),   theta*_b = T(x*_b),   b = 1, ..., B.

The trade is stark and worth naming: if the family is right this is
strictly more efficient -- the replicates are continuous, the tails are
extrapolated rather than truncated at the sample maximum, and B can
exceed the number of distinct nonparametric resamples -- and if the
family is wrong the bootstrap is confidently wrong, with no diagnostic
inside the procedure that would say so.

``rvs_fn(theta, n, g)`` supplies the simulator and receives the shared
Lehmer stream ``g`` so that both language arms draw the same numbers;
the default is the normal family with ``theta = (mu, sigma)``.

Anchor: for the normal family and the sample mean the conditional
variance of the replicates is exactly sigma_hat^2 / n, a closed form
that never enters the resampling loop; and sigma = 0 makes every
replicate exactly mu, which no amount of simulation noise can blur.
"""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_parametric"]


def normal_rvs(theta, n, g):
    """Default simulator: n iid N(theta[0], theta[1]^2) draws."""
    mu = float(theta[0])
    sd = float(theta[1])
    return [mu + sd * g.norm() for _ in range(int(n))]


def boot_parametric(theta_hat, rvs_fn=None, stat=None, B=200, n=None, seed=1, alpha=0.05):
    """Parametric bootstrap replicates.

    Parameters
    ----------
    theta_hat : array-like
        The fitted parameter, passed straight to ``rvs_fn``.
    rvs_fn : callable, optional
        ``rvs_fn(theta, n, g)`` returning a simulated sample.  Defaults
        to the normal family with ``theta = (mu, sigma)``.
    stat : callable, optional
        Statistic of a sample.  Defaults to the mean.
    B : int
        Replicates.
    n : int
        Simulated sample size.  Required.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    RichResult
        ``theta_b``, ``estimate`` (mean of replicates), ``se``,
        ``lo``/``hi``, ``var_closed`` (sigma^2/n for the default family
        and statistic, NaN otherwise), ``n``, ``B``.
    """
    from . import _tail1core as C

    th = core.vec(theta_hat)
    if n is None:
        raise ValueError("boot_parametric: n (the simulated sample size) is required")
    n = int(n)
    if n < 1:
        raise ValueError("boot_parametric: n must be at least 1")
    if int(B) < 2:
        raise ValueError("boot_parametric: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_parametric: alpha must lie strictly between 0 and 1")
    if rvs_fn is None:
        if len(th) < 2:
            raise ValueError("boot_parametric: the default normal family needs theta = (mu, sigma)")
        if th[1] < 0.0:
            raise ValueError("boot_parametric: sigma must be non-negative")
    f = core.mean if stat is None else stat
    r = normal_rvs if rvs_fn is None else rvs_fn
    g = C.Lcg(seed)
    reps = [float(f(r(th, n, g))) for _ in range(int(B))]
    default = rvs_fn is None and stat is None
    return RichResult(
        title="Parametric bootstrap",
        summary_lines=[("n", n), ("B", int(B)), ("estimate", core.mean(reps))],
        payload={
            "theta_b": reps,
            "estimate": core.mean(reps),
            "se": core.sd(reps, 1),
            "lo": core.quantile7(reps, a / 2.0),
            "hi": core.quantile7(reps, 1.0 - a / 2.0),
            "var_closed": (th[1] * th[1] / n) if default else float("nan"),
            "n": n,
            "B": int(B),
            "method": "Davison and Hinkley (1997) Bootstrap Methods and their Application, ch. 2",
        },
    )


def cheatsheet():
    return "btparm: simulate from F(.|theta_hat); efficient if the family is right, confidently wrong if not"
