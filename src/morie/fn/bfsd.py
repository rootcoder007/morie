# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Savage-Dickey Bayes factor."""

import numpy as np

from ._richresult import RichResult

__all__ = ["savage_dickey_ratio"]


def savage_dickey_ratio(samples, prior, theta0=0.0, bandwidth=None):
    """
    Savage-Dickey density ratio BF_01 for a point null, with an
    ARBITRARY prior.

    Distinct from morie.fn.bayes_factor_savage_dickey (module bfsdm),
    which assumes a Normal prior, requires scipy, and returns
    Wagenmakers evidence categories. This one takes the prior as a
    callable or as its density value at theta0, uses a stdlib-only
    Gaussian KDE, and adds nothing interpretive. Use bfsdm for the
    standard psychology-style report; use this when the prior is not
    Normal or scipy is unavailable.

    Formula: BF_01 = p(theta_0 | y) / p(theta_0), the posterior
    density at the null over the prior density at the null (valid
    for nested models with matched priors). The posterior density is
    estimated from MCMC ``samples`` by a Gaussian kernel density at
    theta_0 with Silverman's bandwidth 0.9 min(sd, IQR/1.34) n^{-1/5}
    unless one is given. ``prior`` is either a callable density or
    the prior density VALUE at theta_0.

    Parameters
    ----------
    samples : array-like
        Posterior draws (n >= 10).
    prior : callable or float
        Prior density function, or its value at theta0 (> 0).
    theta0 : float
        The null value (default 0).
    bandwidth : float, optional
        KDE bandwidth override (> 0).

    Returns
    -------
    result : dict
        Keys: estimate (BF_01), bf10, posterior_density_at_null,
        prior_density_at_null, bandwidth, n, method.

    References
    ----------
    Verdinelli, I., & Wasserman, L. (1995). Computing Bayes factors
        using a generalization of the Savage-Dickey density ratio.
        *Journal of the American Statistical Association*, 90(430),
        614-618.
    Dickey, J. M. (1971). The weighted likelihood ratio, linear
        hypotheses on normal location parameters. *The Annals of
        Mathematical Statistics*, 42(1), 204-223.

    Examples
    --------
    Posterior centred far from 0 gives BF_01 << 1 (evidence against
    the null); centred at 0 with matching prior gives BF_01 > 1:

    >>> import numpy as np
    >>> g = np.linspace(-3.0, 3.0, 2001)
    >>> draws = g / 3.0                            # posterior ~ Uniform(-1,1): denser at 0
    >>> flat = 1.0 / 6.0                           # Uniform(-3,3) prior
    >>> out = savage_dickey_ratio(draws, flat, 0.0)
    >>> 2.5 < out["estimate"] < 3.5                # ~ (1/2) / (1/6) = 3
    True
    >>> round(out["prior_density_at_null"], 12)
    0.166666666667
    >>> savage_dickey_ratio([0.0] * 5, flat)
    Traceback (most recent call last):
        ...
    ValueError: the Savage-Dickey KDE needs at least 10 posterior draws.
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = samples.size
    if n < 10:
        raise ValueError("the Savage-Dickey KDE needs at least 10 posterior draws.")
    theta0 = float(theta0)
    p0 = float(prior(theta0)) if callable(prior) else float(prior)
    if p0 <= 0:
        raise ValueError(f"the prior density at theta0 must be positive; got {p0}.")
    if bandwidth is None:
        sd = float(np.std(samples, ddof=1))
        q75, q25 = np.percentile(samples, [75, 25])
        spread = min(sd, (q75 - q25) / 1.34) if q75 > q25 else sd
        if spread <= 0:
            raise ValueError("degenerate posterior draws; KDE bandwidth is zero.")
        bandwidth = 0.9 * spread * n ** (-0.2)
    bandwidth = float(bandwidth)
    if bandwidth <= 0:
        raise ValueError(f"the KDE bandwidth must be positive; got {bandwidth}.")
    z = (theta0 - samples) / bandwidth
    post0 = float(np.mean(np.exp(-0.5 * z ** 2)) / (bandwidth * np.sqrt(2.0 * np.pi)))
    bf01 = post0 / p0
    return RichResult(payload={
        "estimate": float(bf01), "bf10": float(1.0 / bf01) if bf01 > 0 else float("inf"),
        "posterior_density_at_null": post0, "prior_density_at_null": p0,
        "bandwidth": bandwidth, "n": int(n),
        "method": "Savage-Dickey BF01 = KDE posterior(theta0) / prior(theta0)"})


def cheatsheet():
    return "bfsd: BF01 = p(theta0|y)/p(theta0); Gaussian KDE, Silverman bandwidth"
