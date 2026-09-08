# morie.fn -- tail3 batch (rootcoder007/morie)
"""Top-of-atmosphere radiation balance.

Source consulted: Hartmann, D.L. (1994). *Global Physical Climatology*.
Academic Press, chapter 2.  The globally averaged net radiation at the top of
the atmosphere is the absorbed solar radiation minus the outgoing longwave
radiation,

    N = (1 - alpha) S / 4 - OLR

the factor of four being the ratio of the Earth's surface area to the disc
area intercepting the solar beam.  Setting N = 0 and applying the
Stefan-Boltzmann law gives the planetary equilibrium temperature

    T_e = ( (1 - alpha) S / (4 sigma) )^{1/4} .
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["toa_radiation_balance"]

SIGMA_SB = 5.670374419e-8


def toa_radiation_balance(S=1361.0, alpha=0.3, OLR=None):
    """Net top-of-atmosphere radiation and equilibrium temperature.

    Parameters
    ----------
    S : float or array-like
        Total solar irradiance in W/m^2.
    alpha : float or array-like
        Planetary albedo.
    OLR : float or array-like, optional
        Outgoing longwave radiation in W/m^2.  When omitted the balance is
        reported as zero and only the equilibrium temperature is meaningful.

    Returns
    -------
    RichResult
        estimate (net N), absorbed, olr, teq, n, method.

    References
    ----------
    Hartmann (1994), Global Physical Climatology, chapter 2.
    """
    Sv = np.atleast_1d(np.asarray(S, dtype=float)).ravel()
    av = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    n = int(max(Sv.size, av.size))
    absorbed = []
    for i in range(n):
        s = float(Sv[i % int(Sv.size)])
        a = float(av[i % int(av.size)])
        absorbed.append((1.0 - a) * s / 4.0)
    if OLR is None:
        ov = [absorbed[i] for i in range(n)]
    else:
        o = np.atleast_1d(np.asarray(OLR, dtype=float)).ravel()
        ov = [float(o[i % int(o.size)]) for i in range(n)]
    net = [absorbed[i] - ov[i] for i in range(n)]
    teq = [float((absorbed[i] / SIGMA_SB) ** 0.25) for i in range(n)]
    return RichResult(
        payload={
            "estimate": float(np.mean(np.asarray(net, dtype=float))),
            "net": np.asarray(net, dtype=float),
            "absorbed": float(np.mean(np.asarray(absorbed, dtype=float))),
            "olr": float(np.mean(np.asarray(ov, dtype=float))),
            "teq": float(np.mean(np.asarray(teq, dtype=float))),
            "n": n,
            "method": "Top-of-atmosphere radiation balance (Hartmann 1994)",
        }
    )


# CANONICAL TEST
# >>> # S = 1361, alpha = 0.30 -> absorbed 238.175, T_e about 254.6 K
# >>> r = toa_radiation_balance(1361.0, 0.30, 238.175)
# >>> assert abs(r["absorbed"] - 238.175) < 1e-9
# >>> assert abs(r["estimate"]) < 1e-9
# >>> assert abs(r["teq"] - 254.6) < 0.2


def cheatsheet():
    return "recapTOA(S, alpha, OLR): TOA net radiation + equilibrium temp."
