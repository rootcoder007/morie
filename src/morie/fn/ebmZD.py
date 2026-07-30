# morie.fn -- function file (rootcoder007/morie)
"""Zero-dimensional zonal energy-balance model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["zonal_ebm"]


def zonal_ebm(S, albedo=0.3, A=203.3, B=2.09, k=3.8, n_zones=9, max_iter=500,
              tol=1e-8, ice_albedo=0.62, ice_threshold=-10.0, start=15.0):
    r"""Budyko-Sellers zonal energy balance with ice-albedo feedback.

    Each latitude band balances absorbed shortwave against outgoing longwave
    and meridional transport:

    .. math::
        Q S_i (1 - \alpha_i) = A + B T_i + k\,(T_i - \bar T),

    with the linearised outgoing longwave :math:`A + BT` and transport
    proportional to the departure from the global mean.

    The model exists to show that this system has **multiple stable
    equilibria**. Albedo depends on temperature -- ice forms below a threshold
    and reflects far more -- so cooling begets cooling. Solved iteratively from
    a warm start it settles on the present climate; from a cold start, on a
    globally ice-covered state that is equally stable under identical forcing.
    That bistability, not the temperature values, is the result worth having,
    and it is what the doctest demonstrates.

    Linearising outgoing longwave as :math:`A + BT` is an approximation to
    Stefan-Boltzmann valid over a narrow range; it is not reliable for the
    snowball state it predicts. The model establishes that bistability exists,
    not the properties of the far equilibrium.

    Parameters
    ----------
    S : float or array-like
        Solar distribution factor per zone, or a scalar multiplier on the
        default insolation profile.
    albedo : float
        Ice-free albedo.
    A, B : float
        Outgoing-longwave intercept (W/m^2) and slope (W/m^2/K).
    k : float
        Meridional transport coefficient.
    n_zones : int
        Latitude bands.
    max_iter, tol
        Iteration controls.
    start : float
        Initial temperature (C). The equilibrium reached depends on it -- that
        is the bistability, not a numerical artifact.
    ice_albedo : float
        Albedo of ice-covered zones.
    ice_threshold : float
        Temperature (C) below which a zone becomes ice-covered.

    Returns
    -------
    RichResult
        ``temperature``, ``global_mean``, ``ice_fraction``, ``albedo``,
        ``latitude``, ``converged``, ``snowball``.

    References
    ----------
    Budyko, M. I. (1969). The effect of solar radiation variations on the
        climate of the Earth. *Tellus*, 21(5), 611-619.
    Sellers, W. D. (1969). A global climatic model based on the energy balance
        of the earth-atmosphere system. *Journal of Applied Meteorology*,
        8(3), 392-400.

    Examples
    --------
    From a warm start the model settles on a temperate climate with partial
    ice cover.

    >>> import numpy as np
    >>> warm = zonal_ebm(1.0, start=20.0)
    >>> bool(warm["global_mean"] > 0 and not warm["snowball"])
    True

    From a cold start, under identical forcing, it settles on a globally
    ice-covered state. Two stable equilibria for one forcing -- the point of
    the model.

    >>> cold = zonal_ebm(1.0, start=-40.0)
    >>> bool(cold["snowball"] and cold["global_mean"] < warm["global_mean"])
    True

    Equatorial zones are warmer than polar ones, as the insolation profile
    requires.

    >>> bool(warm["temperature"][len(warm["temperature"]) // 2]
    ...      > warm["temperature"][0])
    True

    Reducing the solar constant enough tips the warm branch into snowball.

    >>> bool(zonal_ebm(0.7, start=20.0)["snowball"])
    True
    """
    n = int(n_zones)
    if n < 2:
        raise ValueError("n_zones must be at least 2")
    # Equal-area bands: uniform in sin(latitude), so a plain mean over zones
    # IS the area-weighted global mean. Uniform-in-degrees bands would
    # over-weight the poles and cool the global mean by tens of degrees.
    edges = np.linspace(-1.0, 1.0, n + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    lat = np.degrees(np.arcsin(x))
    # Second-Legendre insolation profile, mean 1 over equal-area bands.
    prof = 1.0 - 0.482 * (1.5 * x**2 - 0.5)
    Sarr = (np.atleast_1d(np.asarray(S, dtype=float)).ravel()
            if np.size(S) > 1 else float(S) * prof)
    if Sarr.size != n:
        raise ValueError(f"S must be a scalar or have {n} entries")
    Q = 1361.0 / 4.0

    return _solve(Sarr, Q, albedo, A, B, k, lat, max_iter, tol,
                  ice_albedo, ice_threshold, start)


def _solve(Sarr, Q, albedo, A, B, k, lat, max_iter, tol, ice_albedo,
           ice_threshold, start):
    n = Sarr.size
    T = np.full(n, float(start))
    conv = False
    for _ in range(int(max_iter)):
        al = np.where(T < ice_threshold, ice_albedo, albedo)
        absorbed = Q * Sarr * (1.0 - al)
        Tbar = float(T.mean())
        # Solve A + B T + k (T - Tbar) = absorbed for T, holding Tbar fixed.
        T_new = (absorbed - A + k * Tbar) / (B + k)
        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            conv = True
            break
        T = 0.5 * T + 0.5 * T_new
    al = np.where(T < ice_threshold, ice_albedo, albedo)
    ice = float(np.mean(T < ice_threshold))
    from ._richresult import RichResult as _RR

    return _RR(
        title="Zonal energy-balance model",
        summary_lines=[("zones", int(n)), ("global mean", float(T.mean())),
                       ("ice fraction", ice)],
        warnings=["the linearised A + BT outgoing longwave is valid over a "
                  "narrow temperature range and is not reliable for the "
                  "snowball state the model predicts"],
        payload={
            "temperature": T, "global_mean": float(T.mean()),
            "ice_fraction": ice, "albedo": al, "latitude": lat,
            "converged": conv, "snowball": bool(ice > 0.95),
            "method": "zonal_ebm",
        },
    )


def cheatsheet():
    return "ebmZD: ice-albedo feedback gives TWO stable equilibria for one forcing; start decides which"
