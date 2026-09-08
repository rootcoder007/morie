# morie.fn -- function file (rootcoder007/morie)
"""Beta-binomial conjugate updating."""

import math

from ._richresult import RichResult

__all__ = ["betabinom", "beta_binomial"]


def betabinom(y, n, alpha=1.0, beta=1.0, m=None):
    """Posterior and predictive quantities of the Beta-Binomial model.

    With p ~ Beta(alpha, beta) and y | p ~ Binomial(n, p), the Beta family
    is conjugate: the posterior is again Beta, with the successes and
    failures simply added to the prior counts,

        p | y ~ Beta(alpha + y, beta + n - y),

    so the posterior mean is a weighted average of the prior mean and the
    sample proportion.  Marginalising p out of the sampling model gives
    the beta-binomial prior predictive,

        p(y) = C(n, y) B(alpha + y, beta + n - y) / B(alpha, beta),

    and repeating the same integral with the posterior parameters gives
    the posterior predictive for m further trials.

    Parameters
    ----------
    y : int
        Observed successes.
    n : int
        Trials, y <= n.
    alpha, beta : float
        Prior shape parameters, strictly positive.
    m : int or None
        Number of future trials for the posterior predictive; ``None``
        uses ``n``.

    Returns
    -------
    RichResult
        ``postalpha``, ``postbeta``, ``postmean``, ``postvar``,
        ``postmode``, ``priormean``, ``logmarglik``, ``predmean``,
        ``predvar``, ``m``.

    References
    ----------
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.
    and Rubin, D. B. (2013), Bayesian Data Analysis, 3rd edn, CRC Press,
    Sect. 2.4 (the beta-binomial conjugate pair and its posterior mean as
    a weighted average) and Sect. 2.5 / Appendix A (the beta-binomial
    predictive distribution).  Standard published form; the third edition
    was not in the local corpus and was not read for this implementation.
    """
    y = int(y)
    n = int(n)
    if n < 0 or y < 0 or y > n:
        raise ValueError("need 0 <= y <= n")
    a, b = float(alpha), float(beta)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("prior shapes must be strictly positive")
    pa, pb = a + y, b + n - y
    s = pa + pb
    pm = pa / s
    pv = pa * pb / (s * s * (s + 1.0))
    mode = float("nan")
    if pa > 1.0 and pb > 1.0:
        mode = (pa - 1.0) / (s - 2.0)

    def lbeta(x, z):
        return math.lgamma(x) + math.lgamma(z) - math.lgamma(x + z)

    lml = (math.lgamma(n + 1.0) - math.lgamma(y + 1.0)
           - math.lgamma(n - y + 1.0) + lbeta(pa, pb) - lbeta(a, b))
    mm = n if m is None else int(m)
    if mm < 0:
        raise ValueError("m must be non-negative")
    predmean = mm * pm
    predvar = mm * pm * (1.0 - pm) * (mm + s) / (s + 1.0)
    return RichResult(payload={
        "postalpha": pa, "postbeta": pb, "postmean": pm, "postvar": pv,
        "postmode": mode, "priormean": a / (a + b), "logmarglik": lml,
        "predmean": predmean, "predvar": predvar, "m": mm,
        "method": "Beta-Binomial conjugate updating (Gelman et al. BDA3 Sect. 2.4)"})


beta_binomial = betabinom


betabinomial = betabinom


def cheatsheet():
    return "betbnm: Beta-binomial conjugate updating."
