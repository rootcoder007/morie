# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metropolis-Hastings sampler."""

from . import _array_core as np

from ._richresult import RichResult
from .wsmnpb import _lcg_uniforms
from .wsmgib import _norm_inv

__all__ = ["wasserman_mcmc_metropolis"]


def wasserman_mcmc_metropolis(target, proposal, x0, n, seed=13):
    """
    Random-walk Metropolis sampler.

    Formula: accept x' with probability
    alpha = min(1, p(x') q(x|x') / (p(x) q(x'|x))); for the
    symmetric normal random walk used here q cancels, so
    alpha = min(1, p(x')/p(x)). ``target`` is an (unnormalised)
    density; ``proposal`` is the random-walk step sd (> 0). Driven
    entirely by the shared LCG (one uniform for the step via
    inversion, one for the accept decision), so chains reproduce
    across languages.

    Parameters
    ----------
    target : callable
        Unnormalised density p(x) >= 0.
    proposal : float
        Random-walk standard deviation, > 0.
    x0 : float
        Starting point with target(x0) > 0.
    n : int
        Iterations, >= 1.
    seed : int
        LCG seed.

    Returns
    -------
    result : dict
        Keys: estimate (chain mean), samples, acceptance_rate,
        n, method.

    References
    ----------
    Wasserman (2004), Ch 24, section 24.3; Metropolis et al (1953).

    Examples
    --------
    Standard normal target: chain mean near 0, healthy acceptance.

    >>> import math
    >>> p = lambda x: math.exp(-0.5 * x * x)
    >>> out = wasserman_mcmc_metropolis(p, 1.0, 0.0, 5000)
    >>> abs(out["estimate"]) < 0.15
    True
    >>> 0.5 < out["acceptance_rate"] < 0.9
    True
    >>> wasserman_mcmc_metropolis(p, 0.0, 0.0, 100)
    Traceback (most recent call last):
        ...
    ValueError: the proposal sd must be positive; got 0.0.
    """
    step = float(proposal)
    if step <= 0:
        raise ValueError(f"the proposal sd must be positive; got {step}.")
    n = int(n)
    if n < 1:
        raise ValueError(f"the sampler needs n >= 1; got {n}.")
    x = float(x0)
    px = float(target(x))
    if px <= 0:
        raise ValueError("the chain must start where the target is positive.")
    u = _lcg_uniforms(2 * n, seed)
    samples = []
    accepted = 0
    for t in range(n):
        prop = x + step * _norm_inv(u[2 * t])
        pp = float(target(prop))
        if pp < 0:
            raise ValueError("a density cannot be negative.")
        if u[2 * t + 1] < min(1.0, pp / px if px > 0 else 0.0):
            x, px = prop, pp
            accepted += 1
        samples.append(x)
    arr = np.asarray(samples)
    return RichResult(payload={
        "estimate": float(np.mean(arr)),
        "samples": [float(v) for v in arr],
        "acceptance_rate": accepted / n, "n": n,
        "method": "random-walk Metropolis, LCG-driven, symmetric q cancels"})


def cheatsheet():
    return "wsmmcm: alpha = min(1, p(x')/p(x)); normal RW steps via LCG inversion"
