"""Neyman-Scott cluster process."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_neyman_scott"]


def schabenberger_neyman_scott(r, rho=10.0, mu=5.0, sigma=0.1):
    r"""
    Neyman-Scott cluster process: second-order behaviour.

    Parents form a homogeneous Poisson process of intensity
    :math:`\rho`; each parent produces :math:`N` offspring, displaced
    independently and identically by a radially symmetric density
    :math:`f`. With a stationary parent process the resulting cluster
    process is stationary with intensity :math:`\lambda = \rho E[N]`, and
    the book gives the second-order intensity as

    .. math::

        \lambda_2(h) = \rho^2 E[N]^2 + \rho\,E[N(N-1)]\,f(h)

    For :math:`N \sim \mathrm{Poisson}(\mu)` (so
    :math:`E[N] = \mu`, :math:`E[N(N-1)] = \mu^2`) with Gaussian
    displacements of standard deviation :math:`\sigma` in the plane, this
    integrates to the K-function

    .. math::

        K(r) = \pi r^2 + \frac{1}{\rho}
               \left(1 - e^{-r^2/(4\sigma^2)}\right)

    The first term is the CSR contribution and the second is the excess
    from clustering: it is strictly positive, so a cluster process always
    sits ABOVE the Poisson K-function, and the excess vanishes as
    :math:`\rho \to \infty` (many, sparse clusters look Poisson).

    Parameters
    ----------
    r : array-like
        Distances at which to evaluate, non-negative.
    rho : float
        Parent intensity, > 0.
    mu : float
        Mean offspring per parent, > 0.
    sigma : float
        Offspring displacement standard deviation, > 0.

    Returns
    -------
    RichResult
        ``r``, ``k`` (K-function), ``k_csr`` (:math:`\pi r^2`),
        ``excess``, ``lambda`` (:math:`\rho\mu`).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.7.2, pp. 126-128.
    """
    if rho <= 0 or mu <= 0 or sigma <= 0:
        raise ValueError("`rho`, `mu` and `sigma` must all be > 0")
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(r < 0):
        raise ValueError("`r` must be non-negative")
    excess = (1.0 - np.exp(-(r**2) / (4.0 * sigma**2))) / rho
    return RichResult(
        title="Neyman-Scott cluster process",
        summary_lines=[("rho", rho), ("mu", mu), ("sigma", sigma),
                       ("lambda", rho * mu)],
        payload={"r": r, "k": np.pi * r**2 + excess, "k_csr": np.pi * r**2,
                 "excess": excess, "lambda": float(rho * mu),
                 "rho": float(rho), "mu": float(mu), "sigma": float(sigma)},
    )


def cheatsheet():
    return "spnscl: Neyman-Scott K(r) = pi r^2 + (1-exp(-r^2/4sigma^2))/rho."
