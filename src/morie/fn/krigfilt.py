# morie.fn -- function file (rootcoder007/morie)
"""Kriging filters and relatives: kriging of the error-free signal (measurement-error filtering),
multi-threshold indicator kriging of the local ccdf with order-relation correction, kriging
efficiency and slope of regression, and collocated co-kriging under the Markov model."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .krgsys import krige, kriging_covariance
from .stgeo import _krige

__all__ = ["filtered_krige", "indicator_ccdf", "kriging_efficiency", "collocated_cokriging"]


def _rows(A):
    return [tuple(float(v) for v in r) for r in (A.tolist() if hasattr(A, "tolist") else A)]


def filtered_krige(z, coords, new_coords, model, error_variance, *, X=None, X0=None, beta=None) -> RichResult:
    r"""Kriging of the error-free signal ``S`` from ``Z = S + e``, ``e`` white noise with variance ``error_variance`` (as gstat's ``Err`` component).

    The data covariance is ``C_S(h) + error_variance 1(i = j)``; the
    covariances with the target and the target variance are those of the
    signal ``C_S`` (a nugget in ``model`` is part of the signal and not
    filtered), so predictions at data locations smooth instead of
    interpolating. Ordinary (default), simple (``beta``) or universal
    (``X``/``X0``) kriging.

    References
    ----------
    Cressie, N. (1993). *Statistics for Spatial Data*, rev. edn. Wiley,
    section 3.2.1.
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat
    package. *Computers and Geosciences*, 30(7), 683-691.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> r = filtered_krige([1.0, 3.0, 2.0], [(0, 0), (2, 0), (0, 2)], [(0, 0)], m, 0.0)
    >>> round(r.prediction[0], 12), abs(round(r.variance[0], 12))
    (1.0, 0.0)
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    n = len(P)
    C = [
        [kriging_covariance(math.dist(P[i], P[j]), model) + (error_variance if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]
    c0 = [[kriging_covariance(math.dist(P[i], q), model) for i in range(n)] for q in Q]
    C00 = [kriging_covariance(0.0, model)] * len(Q)
    Xm = [[float(v) for v in r] for r in X] if X is not None else None
    X0m = [[float(v) for v in r] for r in X0] if X0 is not None else None
    pr, va, W, b = _krige(C, c0, C00, zv, Xm, X0m, beta)
    return RichResult(payload={"prediction": pr, "variance": va, "weights": W, "beta": b})


def indicator_ccdf(z, coords, new_coords, thresholds, models, *, zmin=None, zmax=None) -> RichResult:
    r"""Indicator kriging of the local conditional distribution at several thresholds, with order-relation correction.

    For each threshold ``t_k`` the indicators ``1(z <= t_k)`` are ordinarily
    kriged with ``models[k]`` (one model for all thresholds is median
    indicator kriging). The raw estimates are clipped to ``[0, 1]`` and made
    monotone by averaging the upward and downward corrections (GSLIB
    ``ik3d``; Deutsch and Journel 1998). The E-type mean and variance treat
    the value as uniform within each class, the tails bounded by ``zmin``
    and ``zmax`` (default the data range).

    References
    ----------
    Journel, A. G. (1983). Nonparametric estimation of spatial
    distributions. *Mathematical Geology*, 15(3), 445-468.
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide*, 2nd edn. Oxford University Press.

    Examples
    --------
    >>> m = {"model": "Sph", "psill": 0.25, "range": 3.0}
    >>> r = indicator_ccdf([1.0, 3.0, 2.0, 5.0], [(0, 0), (2, 0), (0, 2), (2, 2)], [(0, 0)], [1.5, 2.5, 4.0], m)
    >>> [round(v, 12) for v in r.ccdf[0]]
    [1.0, 1.0, 1.0]
    """
    zv = [float(v) for v in z]
    T = [float(t) for t in thresholds]
    if any(b <= a for a, b in zip(T, T[1:])):
        raise ValueError("thresholds must increase")
    ms = (
        list(models)
        if isinstance(models, (list, tuple)) and len(models) == len(T) and isinstance(models[0], dict)
        else [models] * len(T)
    )
    lo = min(zv) if zmin is None else float(zmin)
    hi = max(zv) if zmax is None else float(zmax)
    raw = []
    for t, m in zip(T, ms):
        raw.append(krige([1.0 if v <= t else 0.0 for v in zv], coords, new_coords, m)["prediction"])
    K, M = len(T), len(raw[0])
    ccdf, mean, var = [], [], []
    cuts = [lo] + T + [hi]
    for q in range(M):
        F = [min(1.0, max(0.0, raw[k][q])) for k in range(K)]
        up, dn = F[:], F[:]
        for k in range(1, K):
            up[k] = max(up[k - 1], up[k])
        for k in range(K - 2, -1, -1):
            dn[k] = min(dn[k + 1], dn[k])
        G = [(a + b) / 2 for a, b in zip(up, dn)]
        ccdf.append(G)
        p = [G[0]] + [G[k] - G[k - 1] for k in range(1, K)] + [1 - G[-1]]
        mid = [(cuts[k] + cuts[k + 1]) / 2 for k in range(K + 1)]
        wid = [cuts[k + 1] - cuts[k] for k in range(K + 1)]
        mu = ssum(a * b for a, b in zip(p, mid))
        mean.append(mu)
        var.append(ssum(a * (b * b + w * w / 12) for a, b, w in zip(p, mid, wid)) - mu * mu)
    return RichResult(
        payload={
            "raw": [[raw[k][q] for k in range(K)] for q in range(M)],
            "ccdf": ccdf,
            "etype": mean,
            "conditional_variance": var,
            "thresholds": T,
        }
    )


def kriging_efficiency(z, coords, new_coords, model, *, block=None) -> RichResult:
    r"""Kriging efficiency ``KE = (BV - KV) / BV`` and slope of regression of ordinary (block) kriging (Krige 1996).

    ``BV`` is the variance of the true block (``C(0)`` for points, the
    block-block covariance average with :func:`~morie.fn.krgsys.krige`'s
    ``block`` discretisation), ``KV`` the kriging variance and ``mu`` the
    Lagrange multiplier; slope ``(BV - KV + |mu|) / (BV - KV + 2 |mu|)``
    (1 is conditionally unbiased).

    References
    ----------
    Krige, D. G. (1996). A practical analysis of the effects of spatial
    structure and of data available and accessed, on conditional biases in
    ordinary kriging. In *Geostatistics Wollongong '96*, 799-810. Kluwer.
    Vann, J., Jackson, S. and Bertoli, O. (2003). Quantitative kriging
    neighbourhood analysis for the mining geologist. *AusIMM 5th Mining
    Geology Conference*, 215-223.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> r = kriging_efficiency([1.0, 3.0, 2.0], [(0, 0), (2, 0), (0, 2)], [(0, 0)], m)
    >>> round(r.efficiency[0], 12), round(r.slope[0], 12)
    (1.0, 1.0)
    """
    r = krige(z, coords, new_coords, model, block=block)
    if block is None:
        bv = [kriging_covariance(0.0, model)] * len(r["prediction"])
    else:
        offs = block["offsets"] if isinstance(block, dict) else list(block)
        bw = [float(v) for v in block["weights"]] if isinstance(block, dict) else [1.0 / len(offs)] * len(offs)
        comps = model if isinstance(model, (list, tuple)) else [model]
        sig = [c for c in comps if c.get("model") != "Nug"]
        bbv = ssum(
            wa * wb * ssum(kriging_covariance(math.dist(a, b), {**c, "nugget": 0.0}) for c in sig)
            for a, wa in zip(offs, bw)
            for b, wb in zip(offs, bw)
        )
        bv = [bbv] * len(r["prediction"])
    mu = [abs(float(v[0])) for v in r["lagrange"]]
    ke = [(b - v) / b for b, v in zip(bv, r["variance"])]
    slope = [(b - v + m) / (b - v + 2 * m) for b, v, m in zip(bv, r["variance"], mu)]
    return RichResult(
        payload={
            "efficiency": ke,
            "slope": slope,
            "block_variance": bv,
            "kriging_variance": r["variance"],
            "lagrange": mu,
        }
    )


def collocated_cokriging(
    z, coords, y0, new_coords, model, rho: float, *, mean_z: float = 0.0, mean_y: float = 0.0, var_y: float = 1.0
) -> RichResult:
    r"""Collocated simple co-kriging of ``Z`` with a secondary variable ``Y`` known at the targets (Markov model 1).

    Under Xu et al.'s Markov model ``C_ZY(h) = rho sqrt(C_Z(0) var_y) C_Z(h) /
    C_Z(0)``, only the collocated secondary value enters: solve ``[[C, c],
    [c', var_y]] [lambda; nu] = [c_0; C_ZY(0)]`` with ``c_i = C_ZY(|s_i -
    s_0|)``; the estimate is ``mean_z + lambda'(z - mean_z) + nu (y_0 -
    mean_y)`` and the variance ``C_Z(0) - lambda'c_0 - nu C_ZY(0)``.
    ``rho = 0`` is simple kriging.

    References
    ----------
    Xu, W., Tran, T. T., Srivastava, R. M. and Journel, A. G. (1992).
    Integrating seismic data in reservoir modeling: the collocated cokriging
    alternative. *SPE Annual Technical Conference*, SPE 24742.
    Almeida, A. S. and Journel, A. G. (1994). Joint simulation of multiple
    variables with a Markov-type coregionalization model. *Mathematical
    Geology*, 26(5), 565-588.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> r = collocated_cokriging([1.0, -1.0], [(0, 0), (3, 0)], [0.5], [(1, 0)], m, 0.0)
    >>> round(r.nu[0], 12)
    0.0
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    n = len(P)
    c00 = kriging_covariance(0.0, model)
    s = rho * math.sqrt(c00 * var_y) / c00
    C = [[kriging_covariance(math.dist(P[i], P[j]), model) for j in range(n)] for i in range(n)]
    pred, var, lam_all, nu_all = [], [], [], []
    for q, yq in zip(Q, y0):
        c0 = [kriging_covariance(math.dist(P[i], q), model) for i in range(n)]
        cc = [s * v for v in c0]
        A = [C[i] + [cc[i]] for i in range(n)] + [cc + [float(var_y)]]
        b = c0 + [s * c00]
        Ai = [[float(v) for v in r] for r in inverse(A)]
        w = [ssum(Ai[i][j] * b[j] for j in range(n + 1)) for i in range(n + 1)]
        lam, nu = w[:n], w[n]
        pred.append(mean_z + ssum(lam[i] * (zv[i] - mean_z) for i in range(n)) + nu * (float(yq) - mean_y))
        var.append(c00 - ssum(lam[i] * c0[i] for i in range(n)) - nu * s * c00)
        lam_all.append(lam)
        nu_all.append(nu)
    return RichResult(payload={"prediction": pred, "variance": var, "weights": lam_all, "nu": nu_all})


def cheatsheet() -> str:
    return (
        "filtered_krige / indicator_ccdf / kriging_efficiency / collocated_cokriging -> kriging filters, "
        "indicator ccdf, efficiency and collocated co-kriging."
    )
