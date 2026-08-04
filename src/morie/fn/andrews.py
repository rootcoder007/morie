# morie.fn -- function file (rootcoder007/morie)
"""Andrews sine psi function."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['andrewspsi', 'andrews_sine']


def andrewspsi(r, c=1.339):
    """Andrews sine psi function.

    A redescending psi: beyond c pi the influence is not merely bounded but exactly zero, so a far-enough outlier contributes nothing at all. That is the property that makes it powerful and also the property that makes the objective non-convex, so the starting value of any IRLS loop using it matters. Note the leading c. psi is the derivative of rho(r) = c^2(1 - cos(r/c)), which gives c sin(r/c), not sin(r/c); dropping it breaks the identity psi(r)/r = w(r) against the IRLS weight in morie.fn.andrew, and rescales every M-estimate that uses it.


    Formula: rho(r) = c^2(1 - cos(r/c)), psi(r) = c sin(r/c) for |r| <= c pi, and 0 otherwise

    Parameters
    ----------
    r : array-like
        Scaled residuals.
    c : float
        Tuning constant; 1.339 gives 95% Gaussian efficiency.

    Returns
    -------
    RichResult
        ``psi``, ``rho``, ``psi_deriv``, ``rejected`` (count with |r| > c pi), ``c``, ``n``.

    References
    ----------
    Andrews, Bickel, Hampel, Huber, Rogers and Tukey (1972), Robust
    Estimates of Location, Princeton University Press; Andrews (1974),
    Technometrics 16:523-531.  Not held locally; the form and the
    tuning constant 1.339 are as implemented by statsmodels'
    AndrewWave norm, the reference implementation, whose source was read:
    rho = a^2(1 - cos(z/a)), psi = a sin(z/a), weights = sin(z/a)/(z/a),
    psi_deriv = cos(z/a), all zero (rho constant) beyond |z| > a pi.
    """
    r = C.vec(r)
    c = float(c)
    if c <= 0:
        raise ValueError("c must be positive")
    lim = c * math.pi
    psi = [c * math.sin(v / c) if abs(v) <= lim else 0.0 for v in r]
    rho = [c * c * (1.0 - math.cos(v / c)) if abs(v) <= lim else 2.0 * c * c
           for v in r]
    dpsi = [math.cos(v / c) if abs(v) <= lim else 0.0 for v in r]
    return RichResult(payload={
        "psi": psi, "rho": rho, "psi_deriv": dpsi,
        "rejected": sum(1 for v in r if abs(v) > lim),
        "c": c, "n": len(r), "method": "Andrews sine psi"})


andrews_sine = andrewspsi


def cheatsheet():
    return "andrews: Andrews sine psi function."
