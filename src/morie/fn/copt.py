# morie.fn -- function file (rootcoder007/morie)
"""Student t copula."""

from ._copula import copula_cdf, copula_tau
from ._richresult import RichResult

__all__ = ["t_copula"]


def t_copula(u, v, rho, nu=4.0):
    r"""Student t copula CDF and Kendall's tau.

    .. math:: C(u, v) = T_{\nu,\rho}\big(t_\nu^{-1}(u),
              t_\nu^{-1}(v)\big),

    evaluated by writing the bivariate t as a scale mixture of
    normals and integrating over the chi-square mixing variable. It
    shares the Gaussian copula's tau, :math:`\tau = \tfrac2\pi
    \arcsin\rho` (Czado Table 3.2), but unlike the Gaussian it has
    *symmetric tail dependence* for finite nu -- which is why the two
    can agree on tau and still differ sharply in the joint extremes.

    Parameters
    ----------
    u, v : array-like in [0, 1]
        Uniform margins.
    rho : float in (-1, 1)
        Correlation parameter.
    nu : float > 0, default 4.0
        Degrees of freedom; larger nu approaches the Gaussian copula.

    Returns
    -------
    RichResult
        keys: ``cdf``, ``tau``, ``rho``, ``nu``, ``family``,
        ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3, Table 3.2 p. 54.
    """
    cdf = copula_cdf("t", u, v, rho, nu)
    return RichResult(
        payload={
            "cdf": cdf,
            "tau": copula_tau("t", rho),
            "rho": float(rho),
            "nu": float(nu),
            "family": "t",
            "method": "Student t copula CDF (normal scale-mixture quadrature)",
        }
    )


def cheatsheet():
    return "copt: T_{nu,rho}(t^-1 u, t^-1 v); tau = (2/pi) arcsin rho, symmetric tails"


# compact alias per ledger/NAMING.md
tcopula = t_copula
