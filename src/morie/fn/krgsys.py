# morie.fn -- function file (rootcoder007/morie)
"""Simple, ordinary, universal and block kriging with cross-validation and back-transforms (gstat conventions)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import normal_quantile
from .stkrig import _comp

__all__ = [
    "kriging_covariance",
    "kriging_system",
    "block_discretize",
    "krige",
    "krige_cv",
    "krige_lognormal",
    "factorial_krige",
    "kriging_quantile",
    "kriging_exceedance",
]


def _comps(model):
    return list(model) if isinstance(model, (list, tuple)) else [model]


def kriging_covariance(h: float, model) -> float:
    r"""Covariance of a gstat ``vgm`` model (one component dict or a list of nested ones) at distance ``h``.

    Each component is ``psill rho(h / range)`` plus its ``nugget`` at
    ``h = 0``; ``model`` is ``Exp``, ``Gau``, ``Sph``, ``Mat`` (with
    ``kappa``) or ``Nug`` (as :func:`morie.fn.stkrig.st_covariance`).

    Examples
    --------
    >>> round(kriging_covariance(1.0, [{"model": "Nug", "psill": 0.5}, {"model": "Exp", "psill": 2.0, "range": 2.0}]), 6)
    1.213061
    """
    return sum(_comp(abs(float(h)), c) for c in _comps(model))


def _gauss_legendre(n):
    """Nodes and weights of n-point Gauss-Legendre quadrature on [-1, 1] (Newton on P_n)."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0) if n > 1 else 1.0
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0) if n > 1 else 1.0
        xs.append(x)
        ws.append(2.0 / ((1.0 - x * x) * dp * dp) if n > 1 else 2.0)
    return xs, ws


def block_discretize(block, n: int = 4, method: str = "gauss") -> dict:
    r"""Discretisation of a rectangular block centred at the origin.

    ``method="gauss"`` (gstat's default for ``krige(block = c(bx, by))``):
    the ``n``-point Gauss-Legendre nodes ``b_j x_i / 2`` in each dimension
    with product weights ``prod w_i / 2`` (gstat stores these weights in
    single precision, so its block kriging differs in the eighth digit);
    ``method="regular"``: the ``n^d``
    cell centres ``b_j ((i + 1/2)/n - 1/2)`` with equal weights.

    :return: dict with ``offsets`` and ``weights`` (summing to one), accepted
        by :func:`krige` as ``block``.

    Examples
    --------
    >>> d = block_discretize((2.0, 2.0), 2, "regular")
    >>> d["offsets"], d["weights"]
    ([(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)], [0.25, 0.25, 0.25, 0.25])
    >>> [round(v, 6) for v in block_discretize((0.4, 0.6))["offsets"][0]]
    [0.172227, 0.258341]
    """
    b = [float(v) for v in block]
    if len(b) not in (1, 2, 3) or n < 1 or method not in ("gauss", "regular"):
        raise ValueError("block must have 1-3 sides, n >= 1 and method gauss or regular")
    if method == "gauss":
        xs, ws = _gauss_legendre(int(n))
        axes = [[(bj * x / 2.0, w / 2.0) for x, w in zip(xs, ws)] for bj in b]
    else:
        axes = [[(bj * ((i + 0.5) / n - 0.5), 1.0 / n) for i in range(n)] for bj in b]
    pts = [((), 1.0)]
    for axis in axes:
        pts = [(p + (v,), w * u) for p, w in pts for v, u in axis]
    return {"offsets": [p for p, _ in pts], "weights": [w for _, w in pts]}


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _rows(a):
    A = np.asarray(a, dtype=float).tolist()
    return [tuple(float(v) for v in (r if isinstance(r, list) else [r])) for r in A]


def _krige1(zv, P, X, q, x0, model, beta, block, blue):
    """Prediction, variance, weights and Lagrange multipliers at one target from the given neighbours."""
    n = len(zv)
    C = [[kriging_covariance(_dist(P[i], P[j]), model) for j in range(n)] for i in range(n)]
    Ci = [[float(v) for v in r] for r in inverse(C)]
    if block is None:
        c0 = [kriging_covariance(_dist(P[i], q), model) for i in range(n)]
        c00 = kriging_covariance(0.0, model)
    else:
        offs, bw = block
        pts = [tuple(a + b for a, b in zip(q, o)) for o in offs]
        K = len(pts)
        c0 = [ssum(bw[s] * kriging_covariance(_dist(P[i], pts[s]), model) for s in range(K)) for i in range(n)]
        # gstat: the nugget does not enter the block-block average
        sill0 = ssum(float(c.get("psill", 1.0)) for c in _comps(model) if c.get("model", "Exp") != "Nug")
        c00 = ssum(
            bw[s] * bw[t] * (sill0 if s == t else kriging_covariance(_dist(pts[s], pts[t]), model))
            for s in range(K)
            for t in range(K)
        )
    Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
    if beta is not None:
        b = [float(v) for v in beta]
        m = [ssum(X[i][a] * b[a] for a in range(len(b))) for i in range(n)]
        m0 = ssum(x0[a] * b[a] for a in range(len(b)))
        pred = m0 + ssum(Cic0[i] * (zv[i] - m[i]) for i in range(n))
        return pred, c00 - ssum(c0[i] * Cic0[i] for i in range(n)), Cic0, []
    p = len(X[0])
    CiX = [[ssum(Ci[i][j] * X[j][a] for j in range(n)) for a in range(p)] for i in range(n)]
    A = [[ssum(X[i][a] * CiX[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
    Ai = [[float(v) for v in r] for r in inverse(A)]
    Ciz = [ssum(Ci[i][j] * zv[j] for j in range(n)) for i in range(n)]
    bhat = [ssum(Ai[a][c] * ssum(X[i][c] * Ciz[i] for i in range(n)) for c in range(p)) for a in range(p)]
    if blue:
        v = ssum(x0[a] * Ai[a][c] * x0[c] for a in range(p) for c in range(p))
        return ssum(x0[a] * bhat[a] for a in range(p)), v, [], bhat
    r = [x0[a] - ssum(X[i][a] * Cic0[i] for i in range(n)) for a in range(p)]
    Air = [ssum(Ai[a][c] * r[c] for c in range(p)) for a in range(p)]
    lam = [Cic0[i] + ssum(CiX[i][a] * Air[a] for a in range(p)) for i in range(n)]
    pred = ssum(lam[i] * zv[i] for i in range(n))
    var = c00 - ssum(c0[i] * Cic0[i] for i in range(n)) + ssum(r[a] * Air[a] for a in range(p))
    return pred, var, lam, [-v for v in Air]


def kriging_system(coords, target, model, *, X=None, x0=None) -> RichResult:
    r"""Augmented universal kriging system ``[[C, X], [X', 0]] [lambda; mu] = [c_0; x_0]``.

    ``C`` is the data covariance matrix (nugget on the diagonal), ``c_0``
    the covariances with ``target`` and ``X`` the trend design (default a
    column of ones: ordinary kriging).  Returns the matrix, the right-hand
    side and its solution (weights ``lambda`` and multipliers ``mu``), as
    the system :func:`krige` solves (Cressie 1993, section 3.4).

    Examples
    --------
    >>> r = kriging_system([(0, 0), (1, 0)], (0.5, 0), {"model": "Exp", "psill": 1.0, "range": 1.0})
    >>> [round(v, 6) for v in r.weights], [round(v, 6) for v in r.lagrange]
    ([0.5, 0.5], [-0.077409])
    """
    P = _rows(coords)
    q = tuple(float(v) for v in target)
    n = len(P)
    Xm = [[1.0]] * n if X is None else [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    xv = [1.0] if X is None else [float(v) for v in x0]
    p = len(Xm[0])
    A = [[kriging_covariance(_dist(P[i], P[j]), model) for j in range(n)] + list(Xm[i]) for i in range(n)]
    A += [[Xm[i][a] for i in range(n)] + [0.0] * p for a in range(p)]
    b = [kriging_covariance(_dist(P[i], q), model) for i in range(n)] + xv
    Ai = [[float(v) for v in r] for r in inverse(A)]
    sol = [ssum(Ai[i][j] * b[j] for j in range(n + p)) for i in range(n + p)]
    return RichResult(payload={"matrix": A, "rhs": b, "weights": sol[:n], "lagrange": sol[n:]})


def _neighbours(P, q, nmax, maxdist):
    d = [_dist(p, q) for p in P]
    idx = sorted(range(len(P)), key=lambda i: d[i])
    if maxdist is not None:
        idx = [i for i in idx if d[i] <= maxdist]
    if nmax is not None:
        idx = idx[: int(nmax)]
    return idx


def krige(
    z, coords, new_coords, model, *, X=None, X0=None, beta=None, block=None, nmax=None, maxdist=None, blue: bool = False
) -> RichResult:
    r"""Kriging prediction and variance, as ``gstat::krige``.

    With the covariance matrix ``C`` of the observations (nugget on the
    diagonal), ``c_0`` their covariances with the target and ``C_00`` the
    target variance (nugget included, so kriging interpolates):

    - simple kriging (``beta`` given, mean ``X beta``): prediction
      ``x_0' beta + c_0' C^{-1} (z - X beta)``, variance ``C_00 - c_0' C^{-1} c_0``;
    - universal kriging (default; ``X`` defaults to a column of ones, i.e.
      ordinary kriging): with ``A = X' C^{-1} X``, GLS ``beta_hat`` and
      ``r = x_0 - X' C^{-1} c_0``, prediction ``x_0' beta_hat + c_0' C^{-1}
      (z - X beta_hat)``, variance ``C_00 - c_0' C^{-1} c_0 + r' A^{-1} r``;
      weights ``lambda = C^{-1}(c_0 + X A^{-1} r)`` and Lagrange multipliers
      ``mu = -A^{-1} r`` of ``C lambda + X mu = c_0``, ``X' lambda = x_0``;
    - ``blue``: the GLS trend ``x_0' beta_hat`` with variance ``x_0' A^{-1} x_0``;
    - ``block``: block kriging over the discretisation ``block`` (see
      :func:`block_discretize`) around each target: ``c_0`` and ``C_00``
      are weighted point-block and block-block covariances, the nugget
      left out of the block-block average as in gstat;
    - ``nmax`` / ``maxdist``: local neighbourhood of the ``nmax`` nearest
      observations within ``maxdist`` (``nan`` when empty).

    Coordinates may be of any dimension (Cressie 1993; Wackernagel 2003;
    Pebesma 2004).

    :param z: Observations (n,).
    :param coords: (n, d) locations.
    :param new_coords: (m, d) targets (block centres with ``block``).
    :param model: Covariance model (:func:`kriging_covariance`).
    :param X: (n, p) trend design; default ordinary kriging.
    :param X0: (m, p) trend design at the targets (required with ``X``).
    :param beta: Known trend coefficients (simple kriging; scalar mean without ``X``).
    :param block: Block discretisation: a list of offsets (equal weights)
        or the dict of :func:`block_discretize`.
    :param nmax: Maximum number of nearest neighbours.
    :param maxdist: Maximum neighbour distance.
    :param blue: Return the GLS trend instead of the prediction.
    :return: :class:`RichResult` with ``prediction``, ``variance``,
        ``weights``, ``lagrange`` (per target) and, with a global
        neighbourhood and unknown trend, the GLS ``beta`` and the
        ``trend_residuals`` ``z - X beta_hat``.

    References
    ----------
    Cressie, N. (1993). *Statistics for Spatial Data*, rev. edn. Wiley,
    New York, sections 3.2-3.4.
    Wackernagel, H. (2003). *Multivariate Geostatistics*, 3rd edn.
    Springer, Berlin, chapters 11-12.
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat
    package. *Computers and Geosciences*, 30(7), 683-691.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0, "nugget": 0.1}
    >>> r = krige([1.0, 3.0, 2.0], [(0, 0), (2, 0), (0, 2)], [(1, 1)], m)
    >>> round(r.prediction[0], 6), round(r.variance[0], 6)
    (2.060225, 0.696387)
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P, Q = _rows(coords), _rows(new_coords)
    n, m = len(zv), len(Q)
    if len(P) != n or (Q and len(Q[0]) != len(P[0])):
        raise ValueError("coords must match z and new_coords must have the same dimension")
    if X is None:
        Xm, X0m = [[1.0]] * n, [[1.0]] * m
        if beta is not None and not isinstance(beta, (list, tuple)):
            beta = [float(beta)]
    else:
        if X0 is None:
            raise ValueError("X0 is required with X")
        Xm = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
        X0m = [[float(v) for v in r] for r in np.asarray(X0, dtype=float).tolist()]
        if len(Xm) != n or len(X0m) != m:
            raise ValueError("X must have n rows and X0 m rows")
    blk = None
    if block is not None:
        if isinstance(block, dict):
            offs, bw = block["offsets"], [float(v) for v in block["weights"]]
        else:
            offs = list(block)
            bw = [1.0 / len(offs)] * len(offs)
        blk = ([tuple(float(v) for v in o) for o in offs], bw)
    pred, var, W, L = [], [], [], []
    for k in range(m):
        idx = _neighbours(P, Q[k], nmax, maxdist) if (nmax is not None or maxdist is not None) else list(range(n))
        if not idx:
            pred.append(float("nan"))
            var.append(float("nan"))
            W.append([])
            L.append([])
            continue
        pk, vk, lam, mu = _krige1(
            [zv[i] for i in idx], [P[i] for i in idx], [Xm[i] for i in idx], Q[k], X0m[k], model, beta, blk, blue
        )
        pred.append(pk)
        var.append(vk)
        W.append(lam)
        L.append(mu)
    out = {"prediction": pred, "variance": var, "weights": W, "lagrange": L}
    if beta is None and nmax is None and maxdist is None and n > 0:
        bh = _krige1(zv, P, Xm, P[0], Xm[0], model, None, None, True)[3]
        out["beta"] = bh
        out["trend_residuals"] = [zv[i] - ssum(Xm[i][a] * bh[a] for a in range(len(bh))) for i in range(n)]
    return RichResult(payload=out)


def krige_cv(z, coords, model, *, X=None, beta=None, folds=None, nmax=None, maxdist=None) -> RichResult:
    r"""Cross-validation of kriging, as ``gstat::krige.cv``.

    Each fold (leave-one-out when ``folds`` is omitted, else the integer
    fold labels) is predicted by :func:`krige` from the other folds.
    Returns the predictions, variances, observed values, residuals
    (observed minus predicted), z-scores (residual over the kriging
    standard error), the RMSE, MAE, mean error and mean squared z-score.

    Examples
    --------
    >>> m = {"model": "Sph", "psill": 1.0, "range": 3.0}
    >>> r = krige_cv([1.0, 2.0, 1.5, 1.2, 0.7], [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)], m)
    >>> round(r.rmse, 6)
    0.64189
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = _rows(coords)
    n = len(zv)
    Xm = None if X is None else [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    lab = list(range(n)) if folds is None else [int(v) for v in folds]
    if len(lab) != n or len(P) != n:
        raise ValueError("coords and folds must match z")
    pred, var = [0.0] * n, [0.0] * n
    for f in sorted(set(lab)):
        out = [i for i in range(n) if lab[i] == f]
        keep = [i for i in range(n) if lab[i] != f]
        r = krige(
            [zv[i] for i in keep],
            [P[i] for i in keep],
            [P[i] for i in out],
            model,
            X=None if Xm is None else [Xm[i] for i in keep],
            X0=None if Xm is None else [Xm[i] for i in out],
            beta=beta,
            nmax=nmax,
            maxdist=maxdist,
        )
        for t, i in enumerate(out):
            pred[i], var[i] = r["prediction"][t], r["variance"][t]
    res = [zv[i] - pred[i] for i in range(n)]
    zs = [res[i] / math.sqrt(var[i]) for i in range(n)]
    return RichResult(
        payload={
            "prediction": pred,
            "variance": var,
            "observed": zv,
            "residual": res,
            "zscore": zs,
            "fold": lab,
            "rmse": math.sqrt(ssum(v * v for v in res) / n),
            "mae": ssum(abs(v) for v in res) / n,
            "me": ssum(res) / n,
            "msz": ssum(v * v for v in zs) / n,
        }
    )


def krige_lognormal(z, coords, new_coords, model, *, beta: float | None = None) -> RichResult:
    r"""Lognormal kriging with the unbiased back-transform.

    ``Y = log Z`` is kriged with ``model`` (the covariance of ``Y``); simple
    kriging (known mean ``beta`` of ``Y``) back-transforms as
    ``exp(Y_hat + sigma^2/2)``, ordinary kriging as ``exp(Y_hat + sigma^2/2
    - m)`` with ``m = -mu`` the Lagrange multiplier of the variogram-form
    system (Journel 1980; Cressie 1993, eq. 3.2.40).  Returns the
    back-transformed prediction with the log-scale prediction and variance.

    References
    ----------
    Journel, A. G. (1980). The lognormal approach to predicting local
    distributions of selective mining unit grades. *Mathematical Geology*,
    12(4), 285-303.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 0.3, "range": 2.0}
    >>> r = krige_lognormal([1.0, 3.0, 2.0], [(0, 0), (2, 0), (0, 2)], [(1, 1)], m)
    >>> round(r.prediction[0], 6)
    2.027853
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    if any(v <= 0 for v in zv):
        raise ValueError("lognormal kriging needs positive z")
    r = krige([math.log(v) for v in zv], coords, new_coords, model, beta=beta)
    back = []
    for k, (y, s2) in enumerate(zip(r["prediction"], r["variance"])):
        adj = 0.0 if beta is not None else r["lagrange"][k][0]
        back.append(math.exp(y + 0.5 * s2 + adj))
    return RichResult(payload={"prediction": back, "log_prediction": r["prediction"], "log_variance": r["variance"]})


def factorial_krige(z, coords, new_coords, model, component: int) -> RichResult:
    r"""Factorial kriging of one component of a nested covariance model.

    With ``model`` a list of components and ``C_k`` the one selected,
    ``Y_k(s_0)`` is estimated by ``lambda' z`` under ``sum lambda = 0``
    (the unknown mean is filtered): ``C lambda + mu 1 = c_{k,0}``,
    ``1' lambda = 0``, with variance ``C_k(0) - lambda' c_{k,0}``
    (Matheron 1982; Wackernagel 2003, chapter 14; Goovaerts 1997, 5.6).

    References
    ----------
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
    Oxford University Press, New York.

    Examples
    --------
    >>> m = [{"model": "Exp", "psill": 1.0, "range": 1.0}, {"model": "Sph", "psill": 0.5, "range": 4.0}]
    >>> r = factorial_krige([1.0, 3.0, 2.0, 2.5], [(0, 0), (2, 0), (0, 2), (2, 2)], [(0.5, 0.2)], m, 1)
    >>> round(r.prediction[0], 6)
    -0.175133
    """
    comps = _comps(model)
    if not 0 <= component < len(comps):
        raise ValueError("component out of range")
    ck = comps[component]
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P, Q = _rows(coords), _rows(new_coords)
    n = len(zv)
    C = [[kriging_covariance(_dist(P[i], P[j]), comps) for j in range(n)] for i in range(n)]
    Ci = [[float(v) for v in r] for r in inverse(C)]
    one = [ssum(r) for r in Ci]
    s1 = ssum(one)
    pred, var = [], []
    for q in Q:
        c0 = [_comp(_dist(P[i], q), ck) for i in range(n)]
        Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
        mu = ssum(Cic0) / s1
        lam = [Cic0[i] - mu * one[i] for i in range(n)]
        pred.append(ssum(lam[i] * zv[i] for i in range(n)))
        var.append(_comp(0.0, ck) - ssum(lam[i] * c0[i] for i in range(n)))
    return RichResult(payload={"prediction": pred, "variance": var})


def kriging_quantile(prediction, variance, p: float):
    r"""Gaussian quantile map ``Z_hat + z_p sigma`` of kriging predictions.

    Examples
    --------
    >>> [round(v, 6) for v in kriging_quantile([1.0, 2.0], [0.25, 1.0], 0.975)]
    [1.979982, 3.959964]
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    zp = float(normal_quantile(p))
    return [float(m) + zp * math.sqrt(float(v)) for m, v in zip(prediction, variance)]


def kriging_exceedance(prediction, variance, threshold: float):
    r"""Probability map ``P(Z > t) = 1 - Phi((t - Z_hat)/sigma)`` under a Gaussian kriging error.

    Examples
    --------
    >>> [round(v, 6) for v in kriging_exceedance([1.0, 2.0], [0.25, 1.0], 1.5)]
    [0.158655, 0.691462]
    """
    return [
        0.5 * math.erfc((float(threshold) - float(m)) / math.sqrt(2.0 * float(v))) for m, v in zip(prediction, variance)
    ]


def cheatsheet() -> str:
    return "krige / krige_cv / krige_lognormal / factorial_krige -> gstat-style kriging core."
