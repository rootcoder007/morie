# morie.fn -- slice s03 (rootcoder007/morie)
"""Matern's hard-core processes: no two points closer than R.

Matern, B. (1960), *Spatial Variation*, Meddelanden fran Statens
Skogsforskningsinstitut 49(5), 1-144.  Pages 47 and 48, rendered at
150 dpi with pdftoppm and read as images, give the two models by which
Matern obtains sub-normal dispersion, described there as "two very simple
models, in which no pair of random points is allowed to have a mutual
distance below a certain bound".

MODEL I, p. 47.  Realise a Poisson process of intensity lambda and then
"exclude every event such that the distance to its nearest neighbour is
less than a given positive number R"; if two points are closer than R,
BOTH go.  The retention probability and the pair retention function are

    E[Z(S)] = alpha lambda mu(S)                            (3.6.5) p. 48
    alpha   = exp(-lambda C_n R^n)                          (3.6.6) p. 48
    k(v)    = 0                          for 0 < v < R      (3.6.4) p. 47
              exp[-lambda U(R, R; v)]    for R <= v

MODEL II, p. 48.  The same primary process runs in time over 0 < t < 1
and a point survives if no other primary event fell within R of it
earlier.  Then

    alpha = [1 - exp(-lambda gamma)] / (lambda gamma)        (3.6.8) p. 48
    k(v)  = 0                                for 0 < v < R   (3.6.9) p. 48
            [2 U (1 - e^-(lambda gamma)) - 2 gamma (1 - e^-(lambda U))]
              / [lambda^2 gamma U (U - gamma)]  for R <= v

with gamma the volume C_n R^n of the sphere of radius R.

U is Matern's U(a, b; v) of eq. (3.4.7) p. 38, the volume of the UNION of
the two spheres,

    U(a, b; v) = C_n a^n + C_n b^n - V_n(a, b; v),

V_n being the volume they share, eq. (3.4.4) p. 38.  This module is the
planar case n = 2, where C_2 = pi and V_2 is the circular lens; that lens
is written once, in :mod:`morie.fn.matern`, and imported here.

Two consequences are used as anchors, because both are exact and neither
runs through the code being tested.  At v >= 2R the two discs are
disjoint, so U = 2 gamma, and

    Model I:  k = exp(-2 lambda gamma) = alpha^2
    Model II: k = [1 - 2 e^-x + e^-2x] / x^2 = [(1 - e^-x)/x]^2 = alpha^2
              with x = lambda gamma,

that is, both models decorrelate to independence exactly, and the Model
II collapse in particular only comes out if (3.6.9) has been transcribed
right.  As R -> 0 both alphas tend to 1 and the intensity to lambda, the
Poisson limit.  Model II always retains more than Model I, since
(1 - e^-x)/x > e^-x for every x > 0.

The Gibbs reading.  Independently of Matern's thinnings, the hard-core
point pattern has the unnormalised density lambda^n on configurations
that satisfy the constraint and 0 on those that do not; that is what the
``density`` and ``log_density`` elements report for the supplied
coordinates, and ``retained`` is Model I's own accept/reject decision
per point.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from .matern import _lens_area

from ._richresult import RichResult

__all__ = ["hardcore_process"]


def _pairs(coords):
    """Coerce coords to a list of (x, y).

    Accepts a sequence of length-2 sequences, or a flat sequence of even
    length read in x, y order.
    """
    try:
        n = len(coords)
    except TypeError:
        raise ValueError("hardcore_process: coords must be a sequence")
    if n == 0:
        raise ValueError("hardcore_process: no coordinates supplied")
    first = coords[0]
    if hasattr(first, "__len__") and not isinstance(first, str):
        out = []
        for p in coords:
            if len(p) != 2:
                raise ValueError("hardcore_process: every coordinate must have two components")
            out.append((float(p[0]), float(p[1])))
        return out
    flat = core.vec(coords)
    if len(flat) % 2 != 0:
        raise ValueError("hardcore_process: flat coords must have even length")
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


def _U(r, v):
    """Matern eq. (3.4.7) p. 38 at n = 2: area of the union of two discs."""
    return 2.0 * math.pi * r * r - _lens_area(r, v)


def _k_model1(r, v, lam):
    """Equation (3.6.4) p. 47."""
    if v < r:
        return 0.0
    return math.exp(-lam * _U(r, v))


def _k_model2(r, v, lam):
    """Equation (3.6.9) p. 48."""
    if v < r:
        return 0.0
    gam = math.pi * r * r
    U = _U(r, v)
    num = 2.0 * U * (1.0 - math.exp(-lam * gam)) - 2.0 * gam * (1.0 - math.exp(-lam * U))
    den = lam * lam * gam * U * (U - gam)
    return num / den


def hardcore_process(coords, r, lam, model=2):
    """Matern hard-core process, Matern (1960) section 3.6 pp. 47-48.

    Parameters
    ----------
    coords : array-like
        Planar coordinates, as (x, y) pairs or a flat x, y, x, y, ...
        sequence.
    r : float
        The hard-core distance R; positive.
    lam : float
        Intensity lambda of the underlying Poisson process; positive.
    model : {1, 2}, optional
        Which of Matern's models the reported ``alpha``, ``intensity``
        and ``k`` refer to.  Both models are always computed; this only
        selects which one the headline elements carry.  Default 2.

    Returns
    -------
    estimate     : the thinned intensity alpha lambda of eq. (3.6.5)
    alpha_I, alpha_II : eqs. (3.6.6) and (3.6.8)
    intensity_I, intensity_II : alpha lambda for each
    feasible     : whether every pairwise distance is at least r
    min_dist     : the smallest pairwise distance
    density, log_density : lambda^n on a feasible configuration, else 0
    retained     : Model I's per-point 1/0 decision, deleting both members
                   of every pair closer than r
    n_retained   : how many survived
    d            : the sorted pairwise distances
    k_I, k_II    : eqs. (3.6.4) and (3.6.9) at those distances
    """
    pts = _pairs(coords)
    rr = float(r)
    lm = float(lam)
    if rr != rr or not (rr > 0.0):
        raise ValueError("hardcore_process: the hard-core distance r must be positive")
    if lm != lm or not (lm > 0.0):
        raise ValueError("hardcore_process: the intensity lam must be positive")
    n = len(pts)

    d = []
    close = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            v = math.sqrt(dx * dx + dy * dy)
            d.append(v)
            if v < rr:
                close[i] = True
                close[j] = True
    d.sort()
    min_dist = d[0] if d else float("inf")
    feasible = not any(close)
    retained = [0 if c else 1 for c in close]
    n_retained = sum(retained)

    gam = math.pi * rr * rr
    x = lm * gam
    alpha_I = math.exp(-x)                                  # (3.6.6)
    alpha_II = (1.0 - math.exp(-x)) / x                     # (3.6.8)
    k_I = [_k_model1(rr, v, lm) for v in d]
    k_II = [_k_model2(rr, v, lm) for v in d]

    if model not in (1, 2):
        raise ValueError("hardcore_process: model must be 1 or 2")
    alpha = alpha_I if model == 1 else alpha_II

    return RichResult(
        payload={
            "estimate": alpha * lm,
            "alpha": alpha,
            "alpha_I": alpha_I,
            "alpha_II": alpha_II,
            "intensity_I": alpha_I * lm,
            "intensity_II": alpha_II * lm,
            "feasible": bool(feasible),
            "min_dist": min_dist,
            "density": (lm ** n) if feasible else 0.0,
            "log_density": (n * math.log(lm)) if feasible else float("-inf"),
            "retained": retained,
            "n_retained": n_retained,
            "d": d,
            "k_I": k_I,
            "k_II": k_II,
            "gamma": gam,
            "r": rr,
            "lam": lm,
            "model": model,
            "n": n,
            "method": "Matern (1960) hard-core models I and II, eqs. (3.6.4)-(3.6.9) pp. 47-48",
        }
    )


def cheatsheet():
    return "hcoreg: Hard-core process -- minimum allowed distance"
