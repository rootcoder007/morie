"""Simulate random utility shocks."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rng import random_uniform


def simulate_utility_shocks(n=100, sigma=1.0, dist="gumbel", seed=0) -> DescriptiveResult:
    r"""Draw the random-utility error components of a discrete-choice model.

    In ``U_ij = V_ij + e_ij`` the logit model has i.i.d. type-I extreme-value
    (Gumbel) errors and the probit model normal ones (McFadden 1974; Train
    2009, ch. 3 and 5). With Philox uniforms ``u`` (stream 0 of ``seed``):
    ``dist="gumbel"`` gives ``e = -sigma log(-log u)`` (location 0, scale
    ``sigma``, mean ``sigma * 0.5772...``), ``dist="normal"`` gives ``e =
    sigma Phi^{-1}(u)``. ``value`` is the list of draws.

    References
    ----------
    McFadden, D. (1974). Conditional logit analysis of qualitative choice
    behavior. In P. Zarembka (ed.), *Frontiers in Econometrics*. Academic
    Press, 105-142.
    Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
    Cambridge University Press.

    Examples
    --------
    >>> r = simulate_utility_shocks(4, dist="gumbel", seed=1)
    >>> len(r.value), r.extra["dist"]
    (4, 'gumbel')
    """
    from ._rrng_core import qnorm

    u = [float(v) for v in random_uniform(int(n), seed=seed, stream=0)]
    if dist == "gumbel":
        draws = [-sigma * math.log(-math.log(v)) for v in u]
    elif dist == "normal":
        draws = [sigma * qnorm(v) for v in u]
    else:
        raise ValueError("dist must be 'gumbel' or 'normal'")
    m = math.fsum(draws) / len(draws)
    return DescriptiveResult(
        name="simulate_utility_shocks",
        value=draws,
        extra={
            "shocks": draws,
            "n": int(n),
            "sigma": sigma,
            "dist": dist,
            "mean": m,
            "std": math.sqrt(math.fsum((d - m) ** 2 for d in draws) / len(draws)),
        },
    )


smutl = simulate_utility_shocks


def cheatsheet() -> str:
    return "simulate_utility_shocks(n, sigma, dist='gumbel') -> Gumbel (logit) or normal (probit) utility errors."
