"""Thomas cluster process: Normal offspring displacements."""

from . import _array_core as np

from ._richresult import RichResult
from .spnscl import schabenberger_neyman_scott

__all__ = ["schabenberger_thomas_process"]


def schabenberger_thomas_process(r, rho=10.0, mu=5.0, sigma=0.1):
    r"""
    Thomas process: the Neyman-Scott process with Gaussian offspring.

    Poisson :math:`(\mu)` offspring per parent, displaced by an isotropic
    Gaussian of standard deviation :math:`\sigma`, parents Poisson
    :math:`(\rho)`. Its K-function is

    .. math::

        K(r) = \pi r^2 + \frac{1}{\rho}
               \left(1 - e^{-r^2/(4\sigma^2)}\right)

    This is the Gaussian special case of the general Neyman-Scott form,
    so it delegates rather than restating the algebra.

    Note this is the theoretical K-function, not a simulator. To generate
    a realisation see :func:`morie.fn.sgthm.thomas_process`.

    Parameters
    ----------
    r : array-like
        Distances, non-negative.
    rho : float
        Parent intensity, > 0.
    mu : float
        Mean offspring per parent, > 0.
    sigma : float
        Displacement standard deviation, > 0.

    Returns
    -------
    RichResult
        As :func:`schabenberger_neyman_scott`, plus ``k_function`` as an
        alias of ``k``.

    References
    ----------
    Thomas, M. (1949) A generalization of Poisson's binomial limit for
    use in ecology. Biometrika 36(1-2):18-25.
    doi:10.1093/biomet/36.1-2.18
    The cluster-process framework is Schabenberger & Gotway (2005)
    Sec. 3.7.2, pp. 126-128, which does not name this special case.
    """
    res = schabenberger_neyman_scott(r, rho, mu, sigma)
    payload = dict(res)
    payload["k_function"] = payload["k"]
    return RichResult(
        title="Thomas cluster process (K-function)",
        summary_lines=[("rho", rho), ("mu", mu), ("sigma", sigma)],
        payload=payload,
    )


def cheatsheet():
    return "spthom: Thomas K(r); Gaussian-offspring case of Neyman-Scott."
