# morie.fn -- function file (rootcoder007/morie)
"""Kendall's tau implied by a copula family and parameter."""

from ._copula import FAMILIES, copula_tau, tau_to_theta
from ._richresult import RichResult

__all__ = ["kendalls_tau_copula"]


def kendalls_tau_copula(family, theta=None, nu=None):
    r"""Kendall's tau from a parametric copula.

    .. math:: \tau = 4 \int\!\!\int C(u, v)\, dC(u, v) - 1,

    reported from the closed forms of Czado (2019) Table 3.2 where
    they exist (Gaussian/t: :math:`\tfrac2\pi\arcsin\rho`; Gumbel:
    :math:`1 - 1/\delta`; Clayton: :math:`\delta/(\delta+2)`; Frank:
    the Debye-function expression; Joe: the digamma expression) and
    by numerical evaluation of the double integral otherwise. Also
    returns the inverse map, so a target tau can be turned back into
    a parameter.

    Parameters
    ----------
    family : str
        One of ``independence gaussian t clayton gumbel frank joe
        plackett``.
    theta : float
        The family's parameter (rho for gaussian/t).
    nu : float, optional
        Degrees of freedom for the t copula.

    Returns
    -------
    RichResult
        keys: ``tau``, ``family``, ``theta``, ``theta_roundtrip``
        (the parameter recovered from tau, a self-check), ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Theorem 3.9 (eq. 3.17) and Table 3.2, p. 54.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    tau = copula_tau(family, theta, nu)
    back = None
    if family != "independence" and -1 < tau < 1 and abs(tau) > 1e-9:
        try:
            back = tau_to_theta(family, tau)
        except ValueError:
            back = None
    return RichResult(
        payload={
            "tau": float(tau),
            "family": family,
            "theta": None if theta is None else float(theta),
            "theta_roundtrip": back,
            "method": "Kendall's tau from the copula parameter (Czado Table 3.2)",
        }
    )


def cheatsheet():
    return "taukcp: tau from family+theta via Czado Table 3.2; roundtrip parameter included"
