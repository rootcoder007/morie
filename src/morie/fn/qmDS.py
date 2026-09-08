# morie.fn -- wave3 coordinator batch (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empirical quantile-mapping bias correction (QUANT)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["qmDS", "quantile_mapping"]


def _ecdf_val(sorted_x, v):
    # inverse of the type-7 interpolated quantile function: the
    # piecewise-linear CDF through (x_(j), j/(n-1)), clamped to [0, 1].
    # Self-consistent with _quantile_interp, so mapping a sample onto
    # itself is the identity and a pure shift is recovered exactly --
    # the percentile-table + linear-interpolation procedure of
    # Boe et al. (2007) as adopted by Gudmundsson et al. (2012).
    n = len(sorted_x)
    if n == 1:
        return 0.5
    if v <= sorted_x[0]:
        return 0.0
    if v >= sorted_x[-1]:
        return 1.0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sorted_x[mid] <= v:
            lo = mid
        else:
            hi = mid
    x0, x1 = sorted_x[lo], sorted_x[lo + 1]
    g = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
    return (lo + g) / (n - 1)


def _quantile_interp(sorted_x, p):
    # inverse empirical CDF with linear interpolation between order
    # statistics (type-7 convention, h = (n-1) p)
    n = len(sorted_x)
    if n == 1:
        return sorted_x[0]
    h = (n - 1) * p
    j = int(h)
    if j >= n - 1:
        return sorted_x[n - 1]
    g = h - j
    return sorted_x[j] * (1.0 - g) + sorted_x[j + 1] * g


def quantile_mapping(x_mod, obs, mod):
    """
    Empirical quantile-mapping bias correction.

    The corrected value of a modelled datum x is
    x' = F_obs^{-1}(F_mod(x)) (Gudmundsson et al. 2012, Eq. 2), with
    both distribution functions estimated EMPIRICALLY from calibration
    samples (their Sec. 2.3.1, QUANT, following Boe et al. 2007):
    F_mod is the empirical CDF of the modelled calibration sample and
    F_obs^{-1} the linearly-interpolated empirical quantile function of
    the observed sample. Values beyond the calibration range receive
    the correction of the highest (lowest) trained quantile, per the
    same section.

    Parameters
    ----------
    x_mod : array-like
        Modelled values to correct.
    obs : array-like
        Observed calibration sample (defines F_obs^{-1}).
    mod : array-like
        Modelled calibration sample (defines F_mod).

    Returns
    -------
    result : RichResult
        Keys: estimate (corrected values), probs (F_mod(x)), n_obs,
        n_mod, method.

    References
    ----------
    Gudmundsson, L., Bremnes, J. B., Haugen, J. E. and
    Engen-Skaugen, T. (2012), "Technical Note: Downscaling RCM
    precipitation to the station scale using statistical
    transformations", Hydrology and Earth System Sciences 16,
    3383-3390, doi:10.5194/hess-16-3383-2012, Eq. 2 and Sec. 2.3.1
    (QUANT). Local source: fetched-wave3/gudmundsson-2012-quantile-
    mapping-hess16-3383.pdf. Antecedents: Wood, A. W. et al. (2004)
    Climatic Change 62, 189-216; Boe, J. et al. (2007) Int J
    Climatology 27, 1643-1655 (procedure adopted by the source);
    Maraun, D. (2013) J Climate 26, 2137-2143 (stub's second lead).
    """
    xm = [float(v) for v in np.asarray(x_mod, dtype=float).ravel()]
    ob = sorted(float(v) for v in np.asarray(obs, dtype=float).ravel())
    md = sorted(float(v) for v in np.asarray(mod, dtype=float).ravel())
    if not ob or not md:
        raise ValueError("qmDS: calibration samples must be non-empty")
    probs = []
    out = []
    n = len(md)
    for v in xm:
        p = _ecdf_val(md, v)
        probs.append(p)
        out.append(_quantile_interp(ob, p))
    return RichResult(payload={
        "estimate": out,
        "probs": probs,
        "n_obs": len(ob),
        "n_mod": n,
        "method": "empirical quantile mapping x' = F_obs^-1(F_mod(x)) (Gudmundsson 2012 Eq. 2, QUANT)",
    })


qmDS = quantile_mapping


def cheatsheet():
    return "qmDS(x_mod, obs, mod) -> F_obs^-1(F_mod(x)) empirical quantile mapping (Gudmundsson 2012)"
