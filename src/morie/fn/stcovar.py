# morie.fn -- function file (rootcoder007/morie)
"""Space-time covariance models: gstat's separable, product-sum, metric and sum-metric variograms, and the
Gneiting, Cressie-Huang, Iaco-Cesare, periodic and linear-combination covariance families."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["st_model_variogram", "st_covariance_family", "st_linear_combination"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _g(model, h):
    """gstat variogramLine of one structure (nugget, psill, model, range); 0 at distance 0."""
    if h == 0:
        return 0.0
    psill, kind, a = float(model["psill"]), model["model"], float(model["range"])
    nug = float(model.get("nugget", 0.0))
    r = h / a
    if kind == "Exp":
        f = 1 - math.exp(-r)
    elif kind == "Gau":
        f = 1 - math.exp(-r * r)
    elif kind == "Sph":
        f = 1.0 if r >= 1 else 1.5 * r - 0.5 * r**3
    elif kind == "Lin":
        f = r
    else:
        raise ValueError("marginal model must be Exp, Gau, Sph or Lin")
    return nug + psill * f


def _sill(model):
    return float(model["psill"]) + float(model.get("nugget", 0.0))


def st_model_variogram(
    h,
    u,
    model: str,
    *,
    space=None,
    time=None,
    joint=None,
    sill: float | None = None,
    k: float | None = None,
    stani: float | None = None,
):
    r"""Space-time variograms in the parametrisation of ``gstat::vgmST`` / ``variogramSurface``.

    Marginals are dicts ``{"psill", "model" ("Exp", "Gau", "Sph", "Lin"),
    "range", "nugget"}`` evaluated as ``gstat::variogramLine`` (0 at lag 0):

    - ``separable``: ``sill (g_s + g_t - g_s g_t)`` with unit-sill marginals;
    - ``productSum`` (De Iaco et al. 2001): ``(k C_t + 1) g_s + (k C_s + 1)
      g_t - k g_s g_t``, ``C_s``, ``C_t`` the marginal sills;
    - ``metric``: ``g_j(sqrt(h^2 + (stani u)^2))``;
    - ``sumMetric``: ``g_s(h) + g_t(u) + g_j(sqrt(h^2 + (stani u)^2))``.

    References
    ----------
    Graeler, B., Pebesma, E. and Heuvelink, G. (2016). Spatio-temporal
    interpolation using gstat. *The R Journal*, 8(1), 204-218.
    De Iaco, S., Myers, D. E. and Posa, D. (2001). Space-time analysis using
    a general product-sum model. *Statistics and Probability Letters*,
    52(1), 21-28.

    Examples
    --------
    >>> s = {"psill": 2.0, "model": "Exp", "range": 100.0}
    >>> t = {"psill": 3.0, "model": "Sph", "range": 5.0}
    >>> [round(v, 6) for v in st_model_variogram([0, 50, 120], [1, 3, 10], "productSum", space=s, time=t, k=0.1)]
    [1.0656, 3.687244, 4.997612]
    """
    H, U = _vec(h), _vec(u)
    out = []
    for a, b in zip(H, U):
        if model == "separable":
            gs, gt = _g(space, a), _g(time, b)
            out.append(sill * (gs + gt - gs * gt))
        elif model == "productSum":
            gs, gt = _g(space, a), _g(time, b)
            out.append((k * _sill(time) + 1) * gs + (k * _sill(space) + 1) * gt - k * gs * gt)
        elif model == "metric":
            out.append(_g(joint, math.hypot(a, stani * b)))
        elif model == "sumMetric":
            out.append(_g(space, a) + _g(time, b) + _g(joint, math.hypot(a, stani * b)))
        else:
            raise ValueError("model must be separable, productSum, metric or sumMetric")
    return out


def _cov(family, h, u, p):
    s2 = float(p.get("sigma2", 1.0))
    if family == "gneiting":
        psi = p["a"] * abs(u) ** (2 * p["alpha"]) + 1
        return s2 / psi ** p["tau"] * math.exp(-p["c"] * h ** (2 * p["gamma"]) / psi ** (p["beta"] * p["gamma"]))
    if family == "cressie_huang":
        q = p["a"] ** 2 * u * u + 1
        return s2 / q ** (p.get("d", 2) / 2) * math.exp(-(p["b"] ** 2) * h * h / q)
    if family == "iaco_cesare":
        return s2 * (1 + (h / p["a"]) ** p["alpha"] + (abs(u) / p["b"]) ** p["beta"]) ** (-p["delta"])
    if family == "periodic":
        return (
            s2
            * math.exp(-h / p["range"])
            * math.exp(-abs(u) / p.get("tau", math.inf))
            * math.cos(2 * math.pi * u / p["period"])
        )
    if family == "separable_exp":
        return s2 * math.exp(-h / p["range_s"] - abs(u) / p["range_t"])
    if family == "porcu":
        d = p.get("sep", 0.5)
        if not (
            0 < p["power_s"] <= 2 and 0 < p["power_t"] <= 2 and p["scale_s"] > 0 and p["scale_t"] > 0 and 0 <= d <= 1
        ):
            raise ValueError("porcu needs power_s, power_t in (0, 2], scale_s, scale_t > 0 and sep in [0, 1]")
        a1 = 1 + (h / p["scale_s"]) ** p["power_s"]
        a2 = 1 + (abs(u) / p["scale_t"]) ** p["power_t"]
        if d == 0:
            # the sep -> 0 limit of the quasi-arithmetic mean is the geometric mean; GeoModels/CompRandFld
            # instead return the product 1 / (a1 a2) at sep = 0 exactly (discontinuous in sep)
            return s2 / (a1 * a2) if p.get("method") == "GeoModels" else s2 / math.sqrt(a1 * a2)
        return s2 * (0.5 * a1**d + 0.5 * a2**d) ** (-1 / d)
    raise ValueError("unknown covariance family")


def st_covariance_family(h, u, family: str, **params):
    r"""Nonseparable space-time covariance families ``C(h, u)`` (``h`` spatial distance, ``u`` time lag).

    - ``gneiting`` (Gneiting 2002, eq. 14): ``sigma2 / psi^tau exp(-c
      h^{2 gamma} / psi^{beta gamma})``, ``psi = a |u|^{2 alpha} + 1``,
      ``alpha, gamma`` in ``(0, 1]``, ``beta`` in ``[0, 1]`` (0 = separable),
      ``tau >= d/2``;
    - ``cressie_huang`` (Cressie and Huang 1999, example 1): ``sigma2 / (a^2
      u^2 + 1)^{d/2} exp(-b^2 h^2 / (a^2 u^2 + 1))``;
    - ``iaco_cesare`` (De Iaco, Myers and Posa 2002): ``sigma2 (1 + (h/a)^alpha
      + (|u|/b)^beta)^{-delta}``, ``alpha, beta`` in ``(0, 2]``, ``delta > 0``
      (a gamma mixture of products of stable covariances, hence valid);
    - ``periodic``: ``sigma2 exp(-h/range) exp(-|u|/tau) cos(2 pi u /
      period)``, a product of valid covariances with a periodic temporal
      factor (``tau`` infinite by default);
    - ``separable_exp``: ``sigma2 exp(-h/range_s - |u|/range_t)``;
    - ``porcu`` (Porcu, Gregori and Mateu 2006; the quasi-arithmetic-mean
      class, eq. 4 of Bevilacqua et al. 2010 with unit exponents): ``sigma2
      (0.5 (1 + (h/scale_s)^power_s)^sep + 0.5 (1 + (|u|/scale_t)^power_t)^sep)^{-1/sep}``,
      ``power_s, power_t`` in ``(0, 2]``, ``sep`` in ``[0, 1]`` (default 0.5);
      at ``sep = 0`` the continuous limit, the geometric mean, gives the
      separable ``sigma2 ((1 + (h/scale_s)^power_s)(1 + (|u|/scale_t)^power_t))^{-1/2}``;
      ``method="GeoModels"`` reproduces GeoModels/CompRandFld, whose
      ``sep = 0`` branch returns the square of that product instead.

    References
    ----------
    Gneiting, T. (2002). Nonseparable, stationary covariance functions for
    space-time data. *Journal of the American Statistical Association*,
    97(458), 590-600.
    Cressie, N. and Huang, H.-C. (1999). Classes of nonseparable,
    spatio-temporal stationary covariance functions. *Journal of the
    American Statistical Association*, 94(448), 1330-1340.
    De Iaco, S., Myers, D. E. and Posa, D. (2002). Nonseparable space-time
    covariance models: some parametric families. *Mathematical Geology*,
    34(1), 23-42.
    Porcu, E., Gregori, P. and Mateu, J. (2006). Nonseparable stationary
    anisotropic space-time covariance functions. *Stochastic Environmental
    Research and Risk Assessment*, 21(2), 113-122.
    Bevilacqua, M., Mateu, J., Porcu, E., Zhang, H. and Zini, A. (2010).
    Weighted composite likelihood-based tests for space-time separability of
    covariance functions. *Statistics and Computing*, 20(3), 283-293.

    Examples
    --------
    >>> round(st_covariance_family([1.0], [2.0], "gneiting", a=1.0, alpha=0.5, beta=1.0, gamma=0.5, c=1.0, tau=1.0)[0], 6)
    0.187128
    >>> round(st_covariance_family([1.0], [2.0], "porcu", power_s=1.0, power_t=1.0, scale_s=1.0, scale_t=1.0, sep=1.0)[0], 6)
    0.4
    """
    return [_cov(family, a, b, params) for a, b in zip(_vec(h), _vec(u))]


def st_linear_combination(h, u, components, weights) -> RichResult:
    r"""Nonnegative linear combination ``sum_k w_k C_k(h, u)`` of :func:`st_covariance_family` families (a valid covariance).

    ``components`` is a list of ``(family, params)`` pairs.

    Examples
    --------
    >>> r = st_linear_combination([0.0], [0.0], [("separable_exp", {"range_s": 1.0, "range_t": 1.0}),
    ...                                      ("cressie_huang", {"a": 1.0, "b": 1.0})], [0.3, 0.7])
    >>> r.covariance
    [1.0]
    """
    w = _vec(weights)
    if any(v < 0 for v in w):
        raise ValueError("weights must be nonnegative")
    parts = [st_covariance_family(h, u, f, **p) for f, p in components]
    cov = [ssum(wk * c[i] for wk, c in zip(w, parts)) for i in range(len(parts[0]))]
    return RichResult(payload={"covariance": cov, "components": parts})


def cheatsheet() -> str:
    return "st_model_variogram / st_covariance_family / st_linear_combination -> space-time covariance models."
