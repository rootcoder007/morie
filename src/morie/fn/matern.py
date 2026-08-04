# morie.fn -- slice s03 (rootcoder007/morie)
"""The Matern cluster process: Poisson centres, satellites in a disc.

Matern, B. (1960), *Spatial Variation*, Meddelanden fran Statens
Skogsforskningsinstitut 49(5), 1-144.  Section 3.6, "Models of randomly
located points", pp. 46-47, rendered at 150 dpi with pdftoppm and read as
images.

Matern's own set-up, p. 46: centres follow a Poisson process of intensity
lambda; each centre carries a cluster of satellites whose count has mean
m and variance tau^2; f(x - y) is the density of a satellite's position
about a centre at y; and gamma is the autoconvolution

    gamma(u) = Integral f(u + y) f(y) dy.

With Z(S) the number of satellites in S and mu the volume,

    E[Z(S)] = lambda m mu(S)                                  (3.6.1)
    Cov[Z(S1), Z(S2)] = lambda m mu(S1 ^ S2)
        + lambda (m^2 + tau^2 - m) Int Int gamma(u - y) du dy  (3.6.2)

Taking the cluster size Poisson, so tau^2 = m, the bracket collapses to
m^2, and comparing (3.6.2) with the general second-moment form gives the
pair correlation and Ripley's K directly:

    rho_2(u) = (lambda m)^2 + lambda m^2 gamma(u)
    g(u)     = 1 + gamma(u) / lambda
    K(t)     = pi t^2 + H(t) / lambda,  H(t) = Int_{|u| <= t} gamma(u) du.

This module is the planar case n = 2 with satellites uniform on the disc
of radius r about their centre, which is the model that carries Matern's
name today.  There f = 1 / (pi r^2) on the disc, so gamma(u) is the area
common to two discs of radius r whose centres are u apart, divided by
(pi r^2)^2.  That common area is Matern's V_n(A, A; v) of eq. (3.4.4)
p. 38; at n = 2 it is the circular lens

    V_2(r, r; v) = 2 r^2 arccos(v / 2r) - (v / 2) sqrt(4 r^2 - v^2),

zero for v >= 2r, and pi r^2 at v = 0.

H(t) integrates in closed form.  Substituting v = 2 r s and S = t / 2r,

    H(t) = (16 / pi) [ (S^2/2) arccos S + (1/8) arcsin S
                       - (1/8) S sqrt(1 - S^2)
                       - (1/4) S^3 sqrt(1 - S^2) ],   S = min(t / 2r, 1),

and at S = 1 the bracket is pi/16, so H = 1 exactly.  That is forced:
gamma is the autoconvolution of a probability density, so it integrates
to 1 over the whole plane, and two discs of radius r more than 2r apart
cannot overlap.  Hence for every t >= 2r

    K(t) = pi t^2 + 1 / lambda

exactly, which is the anchor this module is checked against.  It also
says what the process is: K exceeds the Poisson pi t^2 by 1/lambda, one
whole excess neighbour per parent, all of it accumulated inside 2r.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["matern_cluster"]


def _lens_area(r, v):
    """Area common to two discs of radius r whose centres are v apart.

    Matern (1960) V_n(A, A; v) of eq. (3.4.4) p. 38, at n = 2.
    """
    if v >= 2.0 * r:
        return 0.0
    if v <= 0.0:
        return math.pi * r * r
    q = v / (2.0 * r)
    if q > 1.0:
        q = 1.0
    return 2.0 * r * r * math.acos(q) - 0.5 * v * math.sqrt(4.0 * r * r - v * v)


def _H(t, r):
    """Integral of gamma over the disc of radius t; 1 for t >= 2r."""
    if t <= 0.0:
        return 0.0
    S = t / (2.0 * r)
    if S >= 1.0:
        return 1.0
    w = math.sqrt(1.0 - S * S)
    br = (
        0.5 * S * S * math.acos(S)
        + 0.125 * math.asin(S)
        - 0.125 * S * w
        - 0.25 * S * S * S * w
    )
    return 16.0 / math.pi * br


def matern_cluster(lambda_p, mu, r, t=None):
    """Second-order properties of the Matern cluster process.

    Parameters
    ----------
    lambda_p : float
        Intensity lambda of the Poisson process of centres, positive.
    mu : float
        Mean number m of satellites per centre, positive.
    r : float
        Cluster radius; satellites are uniform on the disc of radius r
        about their centre.  Positive.
    t : array-like, optional
        Lags at which g and K are reported.  The default is the
        deterministic grid r * (0.25, 0.5, 1, 1.5, 2, 3), which straddles
        2r so that the K(t) = pi t^2 + 1/lambda plateau is exercised.

    Returns
    -------
    estimate  : the intensity lambda m of eq. (3.6.1)
    intensity : same
    t, g, K   : the lags and the pair correlation and K function there
    gamma     : gamma(t), the autoconvolution of the satellite density
    H         : H(t), the integral of gamma over the disc of radius t
    Kpois     : pi t^2, for comparison
    """
    lam = float(lambda_p)
    m = float(mu)
    rr = float(r)
    for nm, val in (("lambda_p", lam), ("mu", m), ("r", rr)):
        if val != val or not (val > 0.0):
            raise ValueError("matern_cluster: %s must be positive" % nm)
    if t is None:
        tv = [0.25 * rr, 0.5 * rr, 1.0 * rr, 1.5 * rr, 2.0 * rr, 3.0 * rr]
    else:
        tv = core.vec(t)
        if len(tv) == 0:
            raise ValueError("matern_cluster: no lags supplied")
        for x in tv:
            if x != x or x < 0.0:
                raise ValueError("matern_cluster: every lag t must be non-negative")
    norm = (math.pi * rr * rr) ** 2
    gam = [_lens_area(rr, x) / norm for x in tv]
    g = [1.0 + gi / lam for gi in gam]
    H = [_H(x, rr) for x in tv]
    K = [math.pi * x * x + H[i] / lam for i, x in enumerate(tv)]
    return RichResult(
        payload={
            "estimate": lam * m,
            "intensity": lam * m,
            "t": tv,
            "g": g,
            "K": K,
            "gamma": gam,
            "H": H,
            "Kpois": [math.pi * x * x for x in tv],
            "lambda_p": lam,
            "mu": m,
            "r": rr,
            "n": len(tv),
            "method": "Matern (1960) cluster process, eqs. (3.6.1)-(3.6.2) p. 46",
        }
    )


def cheatsheet():
    return "matern: Matern cluster process"


# compact alias per ledger/NAMING.md
materncluster = matern_cluster
