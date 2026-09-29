"""Cokriging of a primary variable from co-located primary and secondary data (redirect to lmc_cokriging)."""

from __future__ import annotations

import math

from ._richresult import RichResult
from .lmckrige import lmc_cokriging

__all__ = ["cokriging"]


def _points(v, d=None):
    v = v.tolist() if hasattr(v, "tolist") else list(v)
    if v and isinstance(v[0], (list, tuple)):
        return [[float(t) for t in p] for p in v]
    f = [float(t) for t in v]
    d = 1 if d is None else d
    if len(f) % d:
        raise ValueError("target/coords dim mismatch")
    return [f[i : i + d] for i in range(0, len(f), d)]


def cokriging(
    x,
    y,
    coords,
    target,
    sill_p: float = 1.0,
    range_p: float = 1.0,
    sill_s: float = 1.0,
    range_s: float = 1.0,
    cross_sill: float = 0.5,
    cross_range: float = 1.0,
    nugget: float = 0.0,
    means=(0.0, 0.0),
):
    r"""Cokriging of the primary variable ``Z1`` from co-located (isotopic) ``Z1, Z2`` samples.

    Covariances are exponential: ``C11(h) = (sill_p - nugget) e^{-h/range_p}``,
    ``C22(h) = (sill_s - nugget) e^{-h/range_s}`` (each plus ``nugget`` at
    ``h = 0``) and ``C12(h) = cross_sill e^{-h/cross_range}``. The model is
    written as a sum of coregionalization structures (nugget, the two direct
    exponentials and the cross exponential) and solved by the general
    heterotopic cokriging of :func:`morie.fn.lmckrige.lmc_cokriging`: simple
    cokriging with known ``means`` (default zero means, the former behaviour
    of this function) or ordinary cokriging with ``means=None``. The
    parameters must give a valid (positive definite) cross-covariance.

    Parameters
    ----------
    x, y : primary and secondary values at ``coords``.
    coords : ``(n, d)`` sample coordinates (or ``(n,)`` for ``d = 1``).
    target : one point of dimension ``d``, or a list of points.
    sill_p, range_p, sill_s, range_s : direct covariance parameters.
    cross_sill, cross_range : cross-covariance parameters.
    nugget : nugget of both direct covariances.
    means : ``(m1, m2)`` for simple cokriging, ``None`` for ordinary cokriging.

    Returns
    -------
    RichResult
        ``estimate``, ``se`` (scalars for one target), ``n``, ``method``.

    References
    ----------
    Wackernagel, H. (2003). *Multivariate Geostatistics*, 3rd edn. Springer, ch. 24-25.

    Schabenberger, O. and Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman and Hall/CRC, section 5.8.

    Examples
    --------
    >>> r = cokriging([1.0, 2.0, 1.5, 0.5], [0.8, 2.2, 1.1, 0.7], [(0, 0), (2, 0), (1, 1), (0, 2)], (1, 0))
    >>> round(r.estimate, 10), round(r.se, 10)
    (1.1730729181, 0.8439211982)
    """
    xv = [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    P = _points(coords)
    n = len(xv)
    if len(yv) != n or len(P) != n:
        raise ValueError("x, y, and coords must have matching n")
    d = len(P[0])
    T = _points(target, d)
    if any(len(t) != d for t in T):
        raise ValueError("target/coords dim mismatch")
    lmc = [
        {"model": "Nug", "B": [[nugget, 0.0], [0.0, nugget]]},
        {"model": "Exp", "range": range_p, "B": [[sill_p - nugget, 0.0], [0.0, 0.0]]},
        {"model": "Exp", "range": range_s, "B": [[0.0, 0.0], [0.0, sill_s - nugget]]},
        {"model": "Exp", "range": cross_range, "B": [[0.0, cross_sill], [cross_sill, 0.0]]},
    ]
    r = lmc_cokriging(xv + yv, P + P, [0] * n + [1] * n, T, lmc, target=0, means=None if means is None else list(means))
    ests = r.prediction
    ses = [math.sqrt(max(v, 0.0)) for v in r.variance]
    kind = "Ordinary" if means is None else "Simple"
    return RichResult(
        payload={
            "estimate": ests[0] if len(T) == 1 else ests,
            "se": ses[0] if len(T) == 1 else ses,
            "n": n,
            "method": kind + " cokriging (exponential direct and cross covariances)",
        }
    )


def cheatsheet():
    return "cokrg: cokriging(x, y, coords, target, ...) -> simple/ordinary cokriging via lmc_cokriging"
