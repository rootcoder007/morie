# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Generalized Pareto distribution.

Pickands (1975), "Statistical inference using extreme order
statistics", Annals of Statistics 3(1):119-131,
doi:10.1214/aos/1176343003.  The distribution of exceedances over a
high threshold converges to

    F(x) = 1 - (1 + xi x / sigma)^(-1/xi),   xi != 0,
    F(x) = 1 - exp(-x / sigma),              xi = 0,

on x >= 0 for xi >= 0 and on 0 <= x <= -sigma/xi for xi < 0.  The
xi -> 0 limit is the exponential, the mean is sigma/(1 - xi) for
xi < 1 and infinite otherwise, and the quantile function inverts in
closed form; all three are what the tests check.

The wave2 audit pointed this module at ``morie.fn.dtgpd`` as a
duplicate.  dtgpd is itself still an unimplemented placeholder (it
returns the mean of its input), so there is nothing to delegate to and
the distribution is implemented here.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gpd_distribution"]

_EPS = 1e-12


def _cdf(x, sigma, xi):
    if x <= 0:
        return 0.0
    if abs(xi) < _EPS:
        return 1.0 - math.exp(-x / sigma)
    z = 1.0 + xi * x / sigma
    if z <= 0:
        return 1.0
    return 1.0 - z ** (-1.0 / xi)


def _pdf(x, sigma, xi):
    if x < 0:
        return 0.0
    if abs(xi) < _EPS:
        return math.exp(-x / sigma) / sigma
    z = 1.0 + xi * x / sigma
    if z <= 0:
        return 0.0
    return z ** (-1.0 / xi - 1.0) / sigma


def _q(p, sigma, xi):
    if abs(xi) < _EPS:
        return -sigma * math.log(1.0 - p)
    return sigma * ((1.0 - p) ** (-xi) - 1.0) / xi


def gpd_distribution(sigma, xi, x=None, p=None):
    """CDF, density, quantiles and moments of the GPD."""
    s = float(sigma)
    k = float(xi)
    if s <= 0:
        raise ValueError("gpd_distribution: sigma must be positive")
    xs = [0.5, 1.0, 2.0, 4.0] if x is None else core.vec(x)
    ps = [0.5, 0.9, 0.95, 0.99] if p is None else core.vec(p)
    for v in ps:
        if not 0 < v < 1:
            raise ValueError("gpd_distribution: probabilities must lie in (0, 1)")
    upper = float("inf") if k >= 0 else -s / k
    cdf = [_cdf(v, s, k) for v in xs]
    pdf = [_pdf(v, s, k) for v in xs]
    quant = [_q(v, s, k) for v in ps]
    mean = s / (1.0 - k) if k < 1 else float("inf")
    var = s * s / ((1.0 - k) ** 2 * (1.0 - 2.0 * k)) if k < 0.5 else float("inf")
    return RichResult(
        title="Generalized Pareto distribution",
        summary_lines=[("sigma", s), ("xi", k)],
        payload={
            "estimate": mean,
            "cdf": cdf,
            "pdf": pdf,
            "quantile": quant,
            "mean": mean,
            "variance": var,
            "upper_endpoint": upper,
            "sigma": s,
            "xi": k,
            "n": len(xs),
            "method": "F(x) = 1 - (1 + xi x/sigma)^(-1/xi), Pickands (1975)",
        },
    )


def cheatsheet():
    return "gpdD: generalized Pareto distribution"
