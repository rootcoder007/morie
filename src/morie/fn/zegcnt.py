"""Marginal moments of Zeger's parameter-driven count model."""

import math

from ._richresult import RichResult

__all__ = ["zeger_count_moments"]


def zeger_count_moments(mean_s, mean_sh, sigma2, rho1):
    r"""Marginal mean, variance and correlation of a parameter-driven count model.

    Schabenberger & Gotway (2005, p. 294), after Zeger (1988): a latent
    second-order stationary :math:`Z_1(s)` with :math:`E[Z_1] = 1` and
    :math:`\mathrm{Cov}[Z_1(u), Z_1(u+h)] = \sigma^2\rho_1(h)`, and counts
    with :math:`E[Z_2(s)\mid Z_1] = \mathrm{Var}[Z_2(s)\mid Z_1] =
    \exp\{x(s)'\beta\}Z_1(s)`. With :math:`\mu(s) = \exp\{x(s)'\beta\}`,

    .. math::

        E[Z_2(s)] = \mu(s),\quad \mathrm{Var}[Z_2(s)] = \mu(s) + \sigma^2\mu(s)^2,

        \mathrm{Corr}[Z_2(s), Z_2(s+h)] = \rho_1(h)
        \Big/\Big[\{1 + 1/(\sigma^2\mu(s))\}\{1 + 1/(\sigma^2\mu(s+h))\}\Big]^{1/2},

    so conditioning adds overdispersion but caps the correlation below
    :math:`\rho_1(h)`: with :math:`\sigma^2 = \mu = 1` it is :math:`\rho_1(h)/2`.

    Parameters
    ----------
    mean_s, mean_sh : float
        Marginal means :math:`\mu(s)` and :math:`\mu(s+h)`, positive.
    sigma2 : float
        Latent variance :math:`\sigma^2`, positive.
    rho1 : float
        Latent correlation :math:`\rho_1(h)`, in [-1, 1].

    Returns
    -------
    RichResult
        ``mean_s``, ``mean_sh``, ``var_s``, ``var_sh``, ``cov``, ``corr``.

    References
    ----------
    Zeger, S. L. (1988). A regression model for time series of counts.
    Biometrika 75, 621-629. Schabenberger, O. & Gotway, C. A. (2005).
    Statistical Methods for Spatial Data Analysis. Chapman & Hall/CRC, p. 294.
    """
    ms, mh, s2, r = float(mean_s), float(mean_sh), float(sigma2), float(rho1)
    if not (ms > 0 and mh > 0 and s2 > 0) or not -1 <= r <= 1:
        raise ValueError("means and `sigma2` must be positive and `rho1` in [-1, 1]")
    vs, vh = ms + s2 * ms * ms, mh + s2 * mh * mh
    cov = s2 * r * ms * mh
    return RichResult(
        title="Zeger parameter-driven count model moments",
        summary_lines=[("corr", cov / math.sqrt(vs * vh)), ("rho1", r)],
        payload={"mean_s": ms, "mean_sh": mh, "var_s": vs, "var_sh": vh, "cov": cov, "corr": cov / math.sqrt(vs * vh)},
    )


def cheatsheet():
    return "zegcnt: Zeger (1988) count model; Corr = rho1 / sqrt((1+1/(s2 mu))(1+1/(s2 mu')))"
