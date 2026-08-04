"""Fit a parametric semivariogram model to data by weighted least squares."""

from __future__ import annotations

import math

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["variogram_fit"]

_NGRID = 200


def variogram_fit(coords, values, model="exponential", n_bins=15,
                  max_dist=None):
    r"""Fit :math:`\gamma(h) = c_0 + c\,\rho(h/a)` to an empirical semivariogram.

    The empirical semivariogram is Matheron's method-of-moments estimator

    .. math::
        \hat\gamma(h) = \frac{1}{2|N(h)|}\sum_{N(h)} \{Z(s_i) - Z(s_j)\}^2 ,

    binned into ``n_bins`` lag classes. The model is then

    .. math::
        \gamma(h; c_0, c, a) = c_0 + c\,\{1 - R(h; a)\}

    with :math:`R` the exponential :math:`e^{-h/a}`, gaussian
    :math:`e^{-(h/a)^2}` or spherical
    :math:`1 - 1.5(h/a) + 0.5(h/a)^3` (zero beyond ``a``) correlogram, in
    the range-PARAMETER convention of :mod:`morie.fn.expvar`.

    The previous body was a placeholder: it averaged ``coords`` and never
    looked at ``values`` or ``model``.

    Fitting is deterministic by construction. For a fixed :math:`a` the
    model is linear in :math:`(c_0, c)`, so those two come from a
    closed-form weighted least squares solve with the pair counts
    :math:`|N(h)|` as weights; :math:`a` is profiled over a fixed
    logarithmic grid of ``200`` values between one twentieth and twice
    the largest binned lag. There is no iteration, no starting value and
    no random restart, so the Python and R arms return the same numbers
    bit for bit.

    Parameters
    ----------
    coords : array-like
        Locations, shape ``(n, d)``.
    values : array-like
        Observations, length ``n``.
    model : {'exponential', 'gaussian', 'spherical'}
        Correlogram family.
    n_bins : int, default 15
        Number of lag classes.
    max_dist : float, optional
        Largest lag used; defaults to half the maximum inter-point
        distance, the usual rule of thumb.

    Returns
    -------
    RichResult
        ``c0``, ``c``, ``a``, ``sill``, ``model``, ``lag``, ``gamma_hat``,
        ``counts``, ``fitted``, ``wss`` (the weighted residual sum of
        squares at the optimum), ``n_bins_used``, ``method``.

    Notes
    -----
    Weights are the pair counts, not Cressie's
    :math:`|N(h)|/\gamma(h;\theta)^2`, which would make the objective
    depend on the parameters and require iteration. The weighting is
    stated rather than hidden because it changes the answer.

    Non-negativity is imposed by projection: if the unconstrained solve
    returns a negative nugget or partial sill, that component is fixed at
    zero and the other refitted.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data*, rev. edn.
    Wiley, sec. 2.4 (variogram estimation) and 2.6.2 (least squares
    fitting of variogram models).

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, eq. (4.1) for the
    method-of-moments estimator and eqs. (4.10)-(4.13) for the models.
    """
    lag, gam, cnt = empirical_semivariogram(coords, values, n_bins, max_dist)
    lag = [float(v) for v in lag]
    gam = [float(v) for v in gam]
    cnt = [int(v) for v in cnt]
    use = [i for i in range(len(lag)) if cnt[i] > 0 and lag[i] == lag[i]]
    if len(use) < 3:
        raise ValueError("fewer than three non-empty lag classes; "
                         "cannot fit three parameters")
    hs = [lag[i] for i in use]
    gs = [gam[i] for i in use]
    ws = [float(cnt[i]) for i in use]

    hmax = max(hs)
    lo = math.log(hmax / 20.0)
    hi = math.log(2.0 * hmax)
    best = None
    for k in range(_NGRID):
        a = math.exp(lo + (hi - lo) * k / (_NGRID - 1))
        x = [1.0 - _rho(h / a, model) for h in hs]
        c0, c, wss = _wls2(x, gs, ws)
        if best is None or wss < best[3] - 1e-15:
            best = (a, c0, c, wss)
    a, c0, c, wss = best

    fitted = [c0 + c * (1.0 - _rho(h / a, model)) for h in hs]
    return RichResult(
        payload={
            "c0": c0,
            "c": c,
            "a": a,
            "sill": c0 + c,
            "model": model,
            "lag": hs,
            "gamma_hat": gs,
            "counts": [int(w) for w in ws],
            "fitted": fitted,
            "wss": wss,
            "n_bins_used": len(use),
            "method": "Weighted least squares variogram fit, |N(h)| weights, "
                      "range profiled on a fixed grid",
        }
    )


def _rho(u, model):
    """Correlogram on the range-PARAMETER scale; u = h / a."""
    if model == "exponential":
        return math.exp(-u)
    if model == "gaussian":
        return math.exp(-(u * u))
    if model == "spherical":
        if u >= 1.0:
            return 0.0
        return 1.0 - 1.5 * u + 0.5 * u ** 3
    raise ValueError("unknown model %r; expected exponential, gaussian or "
                     "spherical" % (model,))


def _wls2(x, y, w):
    """Weighted least squares of y on [1, x], projected onto c0 >= 0, c >= 0."""
    sw = sum(w)
    sx = sum(w[i] * x[i] for i in range(len(x)))
    sxx = sum(w[i] * x[i] * x[i] for i in range(len(x)))
    sy = sum(w[i] * y[i] for i in range(len(x)))
    sxy = sum(w[i] * x[i] * y[i] for i in range(len(x)))
    det = sw * sxx - sx * sx
    if abs(det) > 1e-300:
        c0 = (sxx * sy - sx * sxy) / det
        c = (sw * sxy - sx * sy) / det
    else:
        c0, c = -1.0, -1.0
    if c0 < 0.0 or c < 0.0:
        if c0 < 0.0 and c < 0.0:
            c0, c = 0.0, 0.0
        elif c0 < 0.0:
            c0 = 0.0
            c = sxy / sxx if sxx > 0.0 else 0.0
            if c < 0.0:
                c = 0.0
        else:
            c = 0.0
            c0 = sy / sw
            if c0 < 0.0:
                c0 = 0.0
    wss = sum(w[i] * (y[i] - c0 - c * x[i]) ** 2 for i in range(len(x)))
    return c0, c, wss


def cheatsheet():
    return ("krigsv: fit c0, c, a of gamma(h) = c0 + c(1 - rho(h/a)) by "
            "weighted least squares on the binned empirical semivariogram")
