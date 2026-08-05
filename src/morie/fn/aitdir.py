# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Dirichlet density on the simplex.

The Dirichlet density is the standard one,

    f(x | alpha) = Gamma(sum_i alpha_i) / prod_i Gamma(alpha_i)
                   * prod_i x_i^(alpha_i - 1),

on the open (D-1)-simplex, x_i > 0, sum_i x_i = 1.  The stub cites Wilks
(1962), Mathematical Statistics, Wiley, which gives the Dirichlet as the
multivariate generalisation of the beta; that text was not retrievable
here, so the density is written in its standard published form and is
pinned by two independent identities rather than by a page number:

  * alpha = (1, ..., 1) makes the density constant and equal to
    Gamma(D) = (D-1)!, the reciprocal volume of the unit simplex;
  * D = 2 collapses to the Beta(alpha_1, alpha_2) density in x_1.

Both are checked as anchors.  The value is computed on the log scale and
exponentiated once, so the normalising constant never overflows.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["dirichlet_density"]

_SIMPLEX_TOL = 1e-8


def dirichlet_density(x, alpha):
    """Density of a composition under a Dirichlet law.

    Parameters
    ----------
    x : array-like
        A point of the open simplex; must be strictly positive and sum
        to one to within 1e-8.
    alpha : array-like
        Strictly positive concentration parameters, same length as x.

    Returns
    -------
    f : the density
    log_f : its logarithm
    log_const : log Gamma(sum alpha) - sum log Gamma(alpha_i)
    """
    xx = [float(v) for v in k.vec(x)]
    aa = [float(v) for v in k.vec(alpha)]
    D = len(xx)
    if D < 2:
        raise ValueError("dirichlet_density: a composition needs at least 2 parts")
    if len(aa) != D:
        raise ValueError("dirichlet_density: x and alpha have different lengths")
    for v in aa:
        if not (v > 0.0):
            raise ValueError("dirichlet_density: alpha must be strictly positive")
    s = 0.0
    for v in xx:
        if not (v > 0.0):
            raise ValueError("dirichlet_density: x must be strictly inside the simplex")
        s += v
    if abs(s - 1.0) > _SIMPLEX_TOL:
        raise ValueError("dirichlet_density: x does not sum to one")
    a0 = 0.0
    for v in aa:
        a0 += v
    lc = k.lgamma(a0)
    for v in aa:
        lc -= k.lgamma(v)
    lf = lc
    for i in range(D):
        lf += (aa[i] - 1.0) * math.log(xx[i])
    return RichResult(
        title="Dirichlet density",
        summary_lines=[("D", D), ("f", math.exp(lf))],
        payload={
            "f": math.exp(lf),
            "estimate": math.exp(lf),
            "log_f": lf,
            "log_const": lc,
            "alpha0": a0,
            "D": D,
            "method": "f(x|alpha) = Gamma(sum alpha)/prod Gamma(alpha_i) prod x_i^(alpha_i-1)",
        },
    )


def cheatsheet():
    return "aitdir: Dirichlet density on the simplex"


# compact alias per ledger/NAMING.md
dirichletdensity = dirichlet_density
