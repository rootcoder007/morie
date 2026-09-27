"""Gamma-conjugate posterior for an exponential or Poisson rate."""

from ._richresult import RichResult
from ._stats_core import gamma

__all__ = ["gamma_conjugate_posterior"]


def gamma_conjugate_posterior(x, k0, lambda0, likelihood="exponential", cred_level=0.95):
    r"""Posterior Gamma(k*, lambda*) (shape, rate) of a rate under a Gamma(k0, lambda0) prior.

    Hedderich, Sachs & Reynarowych (2023, Overview 30): for exponential data
    :math:`k^* = k_0 + n`, :math:`\lambda^* = \lambda_0 + \sum x_i`. For Poisson
    counts :math:`k^* = k_0 + \sum x_i`, :math:`\lambda^* = \lambda_0 + n` (the
    table prints :math:`\lambda_0 x` for that row; the likelihood
    :math:`\lambda^{\sum x} e^{-n\lambda}` gives :math:`\lambda_0 + n`).

    Parameters
    ----------
    x : sequence of float
        Exponential waiting times or Poisson counts.
    k0, lambda0 : float
        Prior shape and rate (> 0).
    likelihood : {"exponential", "poisson"}
    cred_level : float
        Level of the equal-tailed credible interval.

    Returns
    -------
    RichResult
        ``shape``, ``rate``, ``mean``, ``lower``, ``upper``.

    References
    ----------
    Gelman, A. et al. (2013). Bayesian Data Analysis (3rd ed.), Sec. 2.6.
    """
    xs = [float(v) for v in x]
    if not xs or k0 <= 0 or lambda0 <= 0 or min(xs) < 0:
        raise ValueError("need non-empty non-negative data and positive prior parameters")
    if likelihood == "exponential":
        k, lam = k0 + len(xs), lambda0 + sum(xs)
    elif likelihood == "poisson":
        k, lam = k0 + sum(xs), lambda0 + len(xs)
    else:
        raise ValueError("`likelihood` must be 'exponential' or 'poisson'")
    a = (1 - cred_level) / 2
    lo, hi = float(gamma.ppf(a, k, scale=1 / lam)), float(gamma.ppf(1 - a, k, scale=1 / lam))
    return RichResult(
        title=f"Gamma posterior ({likelihood})",
        summary_lines=[("shape", k), ("rate", lam), ("mean", k / lam)],
        payload={"shape": k, "rate": lam, "mean": k / lam, "lower": lo, "upper": hi},
    )


def cheatsheet():
    return "gmconj: exponential k0 + n, l0 + sum x; poisson k0 + sum x, l0 + n"
