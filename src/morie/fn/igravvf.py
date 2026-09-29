# morie.fn -- function file (rootcoder007/morie)
"""Gravity model variance function."""

from __future__ import annotations

from . import _gravity as gv
from ._containers import SpatialResult


def igravvf(flows_hat, phi=1.0, power=1.0):
    r"""Gravity model variance function.

    The variance function ``Var(F) = phi mu^power`` of the pseudo-maximum
    likelihood family at the fitted flows ``mu``: ``power = 1`` is the
    Poisson (PPML) assumption, ``power = 2`` the gamma PML, ``power = 0``
    constant variance (NLS); ``phi`` is the dispersion (Santos Silva and
    Tenreyro 2006, sec. III; McCullagh and Nelder 1989, ch. 9).

    Parameters
    ----------
    flows_hat : array-like, shape (n,)
        Fitted mean flows.
    phi : float
        Dispersion.
    power : float
        Variance power.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean variance; ``local_values`` the variances.

    References
    ----------
    Santos Silva, J. M. C. and Tenreyro, S. (2006). The log of gravity. *Review of Economics and
    Statistics*, 88(4), 641-658.

    McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. Chapman and
    Hall, ch. 9.

    Examples
    --------
    >>> r = igravvf([2.0, 8.0], phi=1.5, power=2.0)
    >>> r.local_values, r.statistic
    ([6.0, 96.0], 51.0)
    """
    v = [float(phi) * m ** float(power) for m in gv.vec(flows_hat)]
    return SpatialResult(name="igravvf", statistic=gv.ssum(v) / len(v), local_values=v)


igravvf_fn = igravvf


def cheatsheet() -> str:
    return "igravvf(flows_hat, phi, power) -> phi mu^power"
