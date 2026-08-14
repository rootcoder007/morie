"""Standardized Precipitation Index (McKee, Doesken & Kleist 1993)."""

import math

from ._stats_core import gamma as _gamma_dist
from ._richresult import RichResult

__all__ = ["droSPI", "standardized_precipitation_index"]

# Abramowitz-Stegun rational approximation constants as printed in
# Edwards & McKee (1997), Eqs. 3.17-3.18.
_C0, _C1, _C2 = 2.515517, 0.802853, 0.010328
_D1, _D2, _D3 = 1.432788, 0.189269, 0.001308


def _as_z(h):
    # Edwards & McKee Eqs. 3.14-3.16: equiprobability transform of the
    # cumulative probability H to the standard normal Z.
    if h <= 0.0 or h >= 1.0:
        raise ValueError("cumulative probability out of (0, 1)")
    if h <= 0.5:
        t = math.sqrt(math.log(1.0 / (h * h)))
        sign = -1.0
    else:
        t = math.sqrt(math.log(1.0 / ((1.0 - h) ** 2)))
        sign = 1.0
    num = _C0 + _C1 * t + _C2 * t * t
    den = 1.0 + _D1 * t + _D2 * t * t + _D3 * t ** 3
    return sign * (t - num / den)


def _fit_thom(xs):
    # Thom (1958/1966) approximate ML estimates, Edwards & McKee
    # Eqs. 3.6-3.8: A = ln(xbar) - mean(ln x);
    # alpha = (1 + sqrt(1 + 4A/3)) / (4A); beta = xbar / alpha.
    pos = [v for v in xs if v > 0]
    n = len(xs)
    m = n - len(pos)
    q = m / float(n)                        # Thom: q = m/n
    if len(pos) < 3:
        raise ValueError("need at least three positive totals to fit")
    xbar = sum(pos) / len(pos)
    a_stat = math.log(xbar) - sum(math.log(v) for v in pos) / len(pos)
    if a_stat <= 0:
        raise ValueError("degenerate sample (all values equal)")
    alpha = (1.0 + math.sqrt(1.0 + 4.0 * a_stat / 3.0)) / (4.0 * a_stat)
    beta = xbar / alpha
    return q, alpha, beta


def droSPI(precip, scale=3, by_month=True):
    """
    Standardized Precipitation Index for a monthly series.

    McKee, Doesken & Kleist (1993) as operationalized by Edwards &
    McKee (1997), Sec. 3.2: (i) form running `scale`-month totals;
    (ii) fit a gamma distribution to the totals -- separately per
    calendar month when ``by_month`` -- by Thom's approximate
    maximum-likelihood estimates (Eqs. 3.6-3.8; Thom 1958 derives
    the estimator, calling the ML estimates jointly sufficient);
    (iii) form the mixed cumulative probability H(x) =
    q + (1 - q) G(x) with q = m/n the zero-precipitation fraction
    (Eq. 3.12); (iv) transform H to the standard normal Z by the
    Abramowitz-Stegun rational approximation with the printed
    constants c0..d3 (Eqs. 3.14-3.18).  Over the fitting period the
    SPI is standard normal (their Fig. 3.3: P(SPI < -1) = 0.1587),
    with SPI > 0 exactly when precipitation exceeds the fitted
    median.  WMO-No. 1090 (2012) is the operational user guide and
    drought-classification reference.

    Sources
    -------
    McKee, T. B., Doesken, N. J. & Kleist, J. (1993). The
    relationship of drought frequency and duration to time scales.
    *8th Conf. on Applied Climatology*, 179-184 (local copy
    fetched-wave3/THE RELATIONSHIP OF DROUGHT FREQUENCY AND
    DURATION TO TIME SCALES.pdf).
    Edwards, D. C. & McKee, T. B. (1997). *Characteristics of 20th
    Century Drought in the United States at Multiple Time Scales*.
    Climatology Report 97-2, Colorado State University, Eqs.
    3.6-3.18 (local copy fetched-wave3/Characteristics of 20th
    century drought...pdf).
    Thom, H. C. S. (1958). A note on the gamma distribution.
    *Monthly Weather Review*, 86(4), 117-122 (local copy
    fetched-wave3/A note on the gamma distribution.pdf).
    WMO (2012). *Standardized Precipitation Index User Guide*
    (WMO-No. 1090), Geneva (local copy fetched-wave3/Standardized
    Precipitation Index User Guide 1090.pdf).

    Parameters
    ----------
    precip : sequence of float
        Monthly precipitation totals (>= 0), January-first ordering
        assumed for ``by_month``.
    scale : int
        Accumulation time scale in months (1, 3, 6, 12, ...).
    by_month : bool
        Fit gamma parameters separately per calendar month
        (operational standard); False pools all totals.

    Returns
    -------
    RichResult
        Keys: spi (None for the first scale-1 entries), totals,
        params ({month or "pooled": (q, alpha, beta)}), scale.
    """
    x = [float(v) for v in precip]
    if any(v < 0 for v in x):
        raise ValueError("precipitation must be non-negative")
    n = len(x)
    scale = int(scale)
    if scale < 1 or n < scale + 5:
        raise ValueError("series too short for this scale")
    totals = [None] * (scale - 1) + [
        sum(x[i - scale + 1:i + 1]) for i in range(scale - 1, n)]
    groups = {}
    for i, tv in enumerate(totals):
        if tv is None:
            continue
        key = (i % 12) if by_month else "pooled"
        groups.setdefault(key, []).append(tv)
    params = {}
    for key, vals in groups.items():
        params[key] = _fit_thom(vals)
    spi = [None] * n
    for i, tv in enumerate(totals):
        if tv is None:
            continue
        key = (i % 12) if by_month else "pooled"
        q, alpha, beta = params[key]
        g = float(_gamma_dist.cdf(tv, alpha, scale=beta)) if tv > 0 else 0.0
        h = q + (1.0 - q) * g
        h = min(max(h, 1e-9), 1.0 - 1e-9)
        spi[i] = _as_z(h)
    return RichResult(payload={
        "spi": spi,
        "totals": totals,
        "params": {str(k): list(v) for k, v in params.items()},
        "scale": scale,
        "by_month": bool(by_month),
        "method": "SPI (McKee 1993; Edwards-McKee 1997 Eqs. 3.6-3.18)",
    })


# long descriptive alias (stub-era name)
standardized_precipitation_index = droSPI


def cheatsheet():
    return "droSPI: gamma-fit totals (Thom MLE), H=q+(1-q)G, A-S normal transform"

# public names resolved by fn/_lazy_map.json
spi = droSPI
