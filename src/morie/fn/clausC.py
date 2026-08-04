# morie.fn -- tail3 batch (rootcoder007/morie)
"""Clausius-Clapeyron scaling of saturation vapour pressure.

Source consulted: Held, I.M. & Soden, B.J. (2006). Robust responses of the
hydrological cycle to global warming.  *Journal of Climate* 19(21),
5686-5699, whose robust responses all follow from the Clausius-Clapeyron
increase in lower-tropospheric water vapour.  For water vapour over a plane
liquid surface the relation is

    d e_s / dT = L_v e_s / (R_v T^2)

so the fractional rate is (1/e_s) d e_s/dT = L_v / (R_v T^2), about 7 percent
per kelvin near 288 K, the figure Held and Soden work with.  Integrating with
a constant latent heat gives the saturation vapour pressure itself,

    e_s(T) = e_s0 exp( (L_v / R_v) (1/T_0 - 1/T) ),
    e_s0 = 611.2 Pa at T_0 = 273.15 K.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["clausius_clapeyron"]

LV = 2.501e6
RV = 461.5
ES0 = 611.2
T0 = 273.15


def clausius_clapeyron(T, Lv=LV, Rv=RV):
    """Saturation vapour pressure, its derivative and the fractional rate.

    Parameters
    ----------
    T : float or array-like
        Absolute temperature in kelvin.
    Lv : float
        Latent heat of vaporisation, J/kg.
    Rv : float
        Gas constant for water vapour, J/(kg K).

    Returns
    -------
    RichResult
        estimate (fractional rate per K), rate_percent_per_K, es, des_dt, T,
        n, method.

    References
    ----------
    Held & Soden (2006), Journal of Climate 19(21), 5686-5699.
    """
    Tv = np.atleast_1d(np.asarray(T, dtype=float)).ravel()
    n = int(Tv.size)
    es = []
    der = []
    rate = []
    for i in range(n):
        t = float(Tv[i])
        e = ES0 * float(np.exp((float(Lv) / float(Rv)) * (1.0 / T0 - 1.0 / t)))
        d = float(Lv) * e / (float(Rv) * t * t)
        es.append(e)
        der.append(d)
        rate.append(float(Lv) / (float(Rv) * t * t))
    ra = np.asarray(rate, dtype=float)
    return RichResult(
        payload={
            "estimate": float(np.mean(ra)),
            "rate": ra,
            "rate_percent_per_K": float(100.0 * float(np.mean(ra))),
            "es": np.asarray(es, dtype=float),
            "des_dt": np.asarray(der, dtype=float),
            "T": Tv,
            "n": n,
            "method": "Clausius-Clapeyron scaling (Held & Soden 2006)",
        }
    )


# CANONICAL TEST
# >>> # about 7 percent per kelvin near 288 K
# >>> r = clausius_clapeyron(288.0)
# >>> assert 0.06 < r["estimate"] < 0.08
# >>> # e_s at the reference temperature is the reference pressure
# >>> r0 = clausius_clapeyron(273.15)
# >>> assert abs(float(r0["es"][0]) - 611.2) < 1e-9


def cheatsheet():
    return "clausC(T): Clausius-Clapeyron e_s, de_s/dT and fractional rate."
