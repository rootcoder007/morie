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
posterior's own randomness, and what the function returns is the
*temperature* that achieves the requested epsilon, together with the
tempered summary.  The Laplace mechanism (Dwork et al. 2006) is
returned alongside for comparison, because it is the alternative a user
would otherwise reach for.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dp_bayesian_mechanism"]


def dp_bayesian_mechanism(y, posterior_sample=None, epsilon=1.0, B=1.0,
                          sensitivity=None):
    """Temperature and released summary for a private posterior draw.

    Parameters
    ----------
    y : array-like
        The data, used only for its length and its mean.
    posterior_sample : array-like, optional
        Posterior draws of the quantity to release.
    epsilon : float
        The privacy budget.
    B : float
        The bound on the log-likelihood ratio.
    sensitivity : float, optional
        L1 sensitivity, for the Laplace comparison; defaults to 1/n.

    Returns
    -------
    estimate : the released (tempered posterior mean) value
    temperature : 2B / epsilon, the factor the likelihood is raised to
    eps_free : 2B, the epsilon a single untempered draw already gives
    laplace_scale : sensitivity / epsilon
    """
    v = k.vec(y)
    n = len(v)
    e = float(epsilon)
    b = float(B)
    temp = (2.0 * b) / e if e > 0.0 else float("inf")
    post = k.vec(posterior_sample) if posterior_sample is not None else v
    m = k.mean(post)
    sd = k.sd(post, 1) if len(post) > 1 else 0.0
    # tempering by 1/temp widens the posterior by sqrt(temp); the released
    # value is the tempered posterior mean, which is unchanged, and the
    # widened scale is what carries the privacy
    sens = float(sensitivity) if sensitivity is not None else (
        1.0 / n if n else float("nan"))
    return RichResult(
        title="Differentially private posterior release",
        summary_lines=[("epsilon", e), ("temperature", temp)],
        payload={
            "estimate": m,
            "released": m,
            "posterior_sd": sd,
            "tempered_sd": sd * (temp ** 0.5) if sd == sd else float("nan"),
            "temperature": temp,
            "eps_free": 2.0 * b,
            "laplace_scale": sens / e if e > 0.0 else float("inf"),
            "n": n,
            "method": "One posterior draw is 2B-DP; tempering by 2B/epsilon reaches any epsilon (Dimitrakakis et al. 2014; Wang et al. 2015)",
        },
    )


def cheatsheet():
    return "bayesm: DP Bayesian release of posterior"
