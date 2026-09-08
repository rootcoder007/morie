# morie.fn -- function file (rootcoder007/morie)
"""Welsch (Leclerc) robust weight."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["welsch_weight"]


def welsch_weight(y, c=2.9846):
    """Gaussian-shaped weight that decays smoothly and never reaches zero.

    Unlike the biweight, the Welsch weight has no hard cutoff -- a
    residual ten times the tuning constant still counts, just by
    ``e^-100``.  In practice the difference is that the objective is
    smooth everywhere, so gradient methods behave, whereas the
    biweight's kink at ``c`` can stall them.

    Formula: ``w(r) = exp(-(r / c)^2)``, with
    ``rho(r) = (c^2 / 2)[1 - exp(-(r / c)^2)]`` and ``psi = r w(r)``.

    Parameters
    ----------
    y : array-like
        Scaled residuals.
    c : float, default 2.9846
        Tuning constant; 2.9846 gives 95 percent Gaussian efficiency.

    Returns
    -------
    RichResult
        ``estimate`` (total loss ``sum rho``), ``w``, ``rho``, ``psi``,
        ``n``.

    References
    ----------
    Dennis, J. E. & Welsch, R. E. (1978).  Techniques for nonlinear
    least squares and robust regression.  Communications in Statistics
    B 7:345-359, which is the published form of the weight Welsch
    proposed in 1977.
    """
    v = C.vec(y)
    c = float(c)
    w = [math.exp(-((t / c) ** 2)) for t in v]
    rho = [(c * c / 2.0) * (1.0 - w[i]) for i in range(len(v))]
    psi = [v[i] * w[i] for i in range(len(v))]
    return RichResult(payload={
        "estimate": sum(rho), "w": w, "rho": rho, "psi": psi, "n": len(v),
        "method": "Welsch robust weight"})


welschweight = welsch_weight


def cheatsheet():
    return "welshw: Welsch (Leclerc) robust weight."
