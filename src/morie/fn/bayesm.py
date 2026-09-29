# morie.fn -- slice s03 (rootcoder007/morie)
"""Differentially private release of a posterior sample.

Source consulted: Wang, Y.-X., Fienberg, S. E. and Smola, A. (2015).
Privacy for free: posterior sampling and stochastic gradient Monte
Carlo.  *ICML* 37, 2493-2502; and Dimitrakakis, C., Nelson, B.,
Mitrokotsa, A. and Rubinstein, B. (2014).  Robust and private Bayesian
inference.  *ALT*, 291-305.  Their result is that releasing a *single*
draw from the posterior is already differentially private when the
log-likelihood is bounded: if

    sup_(theta, x, x') | log p(x | theta) - log p(x' | theta) | <= B

then one posterior sample is 2B-differentially private, and rescaling
the likelihood by 1 / (2B / epsilon) -- i.e. tempering the posterior --
buys any target epsilon.  Neither was retrievable here as a full text;
the bound and the tempering are quoted in their standard published form.

So the privacy here is not bought with added noise: it comes from the
posterior's own randomness.  The function reports the *temperature* that
achieves the requested epsilon and releases ONE draw, chosen uniformly
from the supplied sample of the tempered posterior.  Releasing the
posterior mean instead -- what this function used to do -- is a
deterministic summary and carries none of the guarantee.  The Laplace mechanism (Dwork et al. 2006) is
returned alongside for comparison, because it is the alternative a user
would otherwise reach for.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["dp_bayesian_mechanism"]


def dp_bayesian_mechanism(y, posterior_sample=None, epsilon=1.0, B=1.0, sensitivity=None, seed=0):
    """Release one posterior draw with its privacy temperature.

    Parameters
    ----------
    y : array-like
        The data; only its length enters (the default Laplace sensitivity).
    posterior_sample : array-like
        Draws of the released quantity from the posterior TEMPERED to the
        requested budget, i.e. with the likelihood raised to
        ``1 / temperature = epsilon / (2 B)``. Required: the mechanism
        releases one of these draws, not a summary of them.
    epsilon : float
        The privacy budget.
    B : float
        The bound on the log-likelihood ratio.
    sensitivity : float, optional
        L1 sensitivity, for the Laplace comparison; defaults to 1/n.
    seed : int
        Philox seed choosing which draw is released.

    Returns
    -------
    RichResult with payload
        estimate / released : the single released draw (the private output)
        draw_index          : which draw was released
        temperature         : 2B / epsilon, the factor the likelihood is tempered by
        eps_free            : 2B, the epsilon an untempered single draw gives
        posterior_mean, posterior_sd : summaries of the supplied draws
                              (NOT private -- their release is not covered)
        laplace_scale       : sensitivity / epsilon, for comparison

    Examples
    --------
    >>> r = dp_bayesian_mechanism([0.2, 0.4], [1.0, 2.0, 3.0, 4.0], epsilon=0.5, seed=3)
    >>> r["temperature"], r["released"]
    (4.0, 4.0)
    """
    v = k.vec(y)
    n = len(v)
    if posterior_sample is None:
        raise ValueError(
            "posterior_sample is required: the mechanism releases one draw from "
            "the tempered posterior, and the data themselves are not a posterior"
        )
    post = k.vec(posterior_sample)
    S = len(post)
    if S == 0:
        raise ValueError("posterior_sample is empty")
    e = float(epsilon)
    b = float(B)
    temp = (2.0 * b) / e if e > 0.0 else float("inf")
    u = float(random_uniform(1, seed=seed, stream=0)[0])
    j = min(int(u * S), S - 1)
    released = float(post[j])
    m = k.mean(post)
    sd = k.sd(post, 1) if S > 1 else 0.0
    sens = float(sensitivity) if sensitivity is not None else (1.0 / n if n else float("nan"))
    return RichResult(
        title="Differentially private posterior release",
        summary_lines=[("epsilon", e), ("temperature", temp), ("released", released)],
        payload={
            "estimate": released,
            "released": released,
            "draw_index": j,
            "posterior_mean": m,
            "posterior_sd": sd,
            "temperature": temp,
            "eps_free": 2.0 * b,
            "laplace_scale": sens / e if e > 0.0 else float("inf"),
            "n": n,
            "method": "One draw from the posterior tempered by 2B/epsilon is epsilon-DP (Dimitrakakis et al. 2014; Wang et al. 2015)",
        },
    )


def cheatsheet():
    return "bayesm: DP Bayesian release of posterior"
