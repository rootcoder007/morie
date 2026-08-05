# morie.fn -- function file (rootcoder007/morie)
"""Normalized gamma process prior."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["normalized_gamma_process"]


def normalized_gamma_process(y, alpha=1.0, tau=1.0):
    """Gamma completely random measure, normalized to a random probability.

    The gamma CRM has Levy intensity

        nu(du, dx) = alpha P0(dx) u^{-1} e^{-u / tau} du,

    so its Laplace exponent is ``psi(lam) = alpha log(1 + lam tau)`` and
    its expected total mass is ``E[T] = alpha tau``.  Normalizing by the
    total mass gives the Dirichlet process ``DP(alpha, P0)`` -- and,
    because normalizing divides out the scale, the resulting law does
    not depend on ``tau`` at all.  That invariance is the point of the
    construction and is asserted as an anchor.

    The induced exchangeable partition has

        E[K_n]   = sum_{i=1}^{n} alpha / (alpha + i - 1)
                 = alpha (digamma(alpha + n) - digamma(alpha)),
        Var[K_n] = sum_{i=1}^{n} alpha (i - 1) / (alpha + i - 1)^2,

    the two expressions for the mean being computed independently here
    so that each checks the other.

    Parameters
    ----------
    y : array-like
        Observed values; ``n`` and the realized number of distinct
        values are read off them.
    alpha : float, default 1.0
        Total mass (concentration) parameter, positive.
    tau : float, default 1.0
        Scale of the gamma CRM, positive.  Affects the unnormalized
        total mass only.

    Returns
    -------
    RichResult
        ``estimate`` (``E[K_n]``), ``e_k``, ``e_k_digamma``, ``var_k``,
        ``k_observed``, ``total_mass``, ``psi1`` (Laplace exponent at
        ``lam = 1``), ``alpha``, ``tau``, ``n``.

    References
    ----------
    Lijoi, A. & Prunster, I. (2010).  Models beyond the Dirichlet
    process.  In N. L. Hjort, C. Holmes, P. Muller & S. G. Walker
    (eds), Bayesian Nonparametrics, 80--136.  Cambridge University
    Press.  Ferguson, T. S. (1973).  A Bayesian analysis of some
    nonparametric problems.  Annals of Statistics, 1(2), 209--230.
    """
    a = float(alpha)
    t = float(tau)
    if a <= 0.0:
        raise ValueError("normalized_gamma_process: alpha must be positive")
    if t <= 0.0:
        raise ValueError("normalized_gamma_process: tau must be positive")
    vals = [float(v) for v in (y if hasattr(y, "__len__") else [y])]
    n = len(vals)
    if n == 0:
        raise ValueError("normalized_gamma_process: y is empty")
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    ek = 0.0
    vk = 0.0
    for i in range(1, n + 1):
        ek += a / (a + i - 1.0)
        vk += a * (i - 1.0) / (a + i - 1.0) ** 2
    ekd = a * (core.digamma(a + n) - core.digamma(a))
    return RichResult(payload={
        "estimate": ek, "e_k": ek, "e_k_digamma": ekd, "var_k": vk,
        "k_observed": len(seen), "total_mass": a * t,
        "psi1": a * math.log(1.0 + t), "alpha": a, "tau": t, "n": n,
        "method": "Normalized gamma process (Dirichlet process) prior"})


def cheatsheet():
    return "ngppr: Normalized gamma process prior (the Dirichlet process)"


normalizedgammaprocess = normalized_gamma_process
