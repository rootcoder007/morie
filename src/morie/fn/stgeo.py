# morie.fn -- function file (rootcoder007/morie)
"""Space-time geostatistics beyond global ordinary kriging: fitting ``vgmST`` models to an empirical
space-time variogram (with information criteria), universal kriging with drift, local-neighbourhood
kriging, block kriging, leave-h-out validation, kriging-system diagnostics, smoothness of a model,
grid prediction and Gaussian simulation."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_normal
from .lbfgsb import lbfgsb_minimize
from .stcovar import st_model_variogram
from .stkrig import st_covariance

__all__ = [
    "st_fit",
    "st_universal_kriging",
    "st_local_kriging",
    "st_block_kriging",
    "st_leave_h_out",
    "st_kriging_diagnostics",
    "st_smoothness",
    "st_predict_grid",
    "st_simulate",
]

_POS = math.sqrt(2.220446049250313e-16)


def _pts(c):
    return [(float(a), float(b)) for a, b in (c.tolist() if hasattr(c, "tolist") else c)]


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _unpack(kind, p, fixed):
    """gstat ``insertPar``: parameter vector -> (model kwargs, marginal dicts)."""
    sm, tm, jm = fixed.get("space_model", "Exp"), fixed.get("time_model", "Exp"), fixed.get("joint_model", "Exp")
    if kind == "separable":
        return {
            "space": {"psill": 1 - p[1], "model": sm, "range": p[0], "nugget": p[1]},
            "time": {"psill": 1 - p[3], "model": tm, "range": p[2], "nugget": p[3]},
            "sill": p[4],
        }
    if kind == "productSum":
        return {
            "space": {"psill": p[0], "model": sm, "range": p[1], "nugget": p[2]},
            "time": {"psill": p[3], "model": tm, "range": p[4], "nugget": p[5]},
            "k": p[6],
        }
    if kind == "metric":
        return {"joint": {"psill": p[0], "model": jm, "range": p[1], "nugget": p[2]}, "stani": p[3]}
    if kind == "sumMetric":
        return {
            "space": {"psill": p[0], "model": sm, "range": p[1], "nugget": p[2]},
            "time": {"psill": p[3], "model": tm, "range": p[4], "nugget": p[5]},
            "joint": {"psill": p[6], "model": jm, "range": p[7], "nugget": p[8]},
            "stani": p[9],
        }
    raise ValueError("model must be separable, productSum, metric or sumMetric")


def st_fit(
    dist,
    timelag,
    gamma,
    np_,
    model: str,
    start,
    *,
    fit_method: int = 6,
    stani: float | None = None,
    space_model: str = "Exp",
    time_model: str = "Exp",
    joint_model: str = "Exp",
    lower=None,
    upper=None,
) -> RichResult:
    r"""Fit a ``vgmST`` space-time variogram to an empirical one by weighted least squares, as ``gstat::fit.StVariogram``.

    Minimises the mean of ``w (gamma - gamma_model)^2`` over the non-empty
    bins (``np > 0``) with L-BFGS-B (projected gradient ``1e-12``) from
    ``start`` in gstat's ``extractPar`` order: ``separable`` (range.s,
    nugget.s, range.t, nugget.t, sill; relative nuggets), ``productSum``
    (sill.s, range.s, nugget.s, sill.t, range.t, nugget.t, k), ``metric``
    (sill, range, nugget, anis), ``sumMetric`` (sill.s, range.s, nugget.s,
    sill.t, range.t, nugget.t, sill.st, range.st, nugget.st, anis). Weights
    ``fit_method`` 1 ``np``, 2 ``np / gamma_model^2``, 6 unit, 7 ``np / (h^2 +
    (stani u)^2)``; default bounds as gstat (ranges at least 5 % of the
    smallest positive distance). Also returns ``mse`` (unweighted, gstat's
    ``MSE`` attribute) and least-squares information criteria ``aic = n
    log(SSE_w / n) + 2 p``, ``bic`` (``n`` bins, ``p`` parameters).

    References
    ----------
    Graeler, B., Pebesma, E. and Heuvelink, G. (2016). Spatio-temporal
    interpolation using gstat. *The R Journal*, 8(1), 204-218.

    Examples
    --------
    >>> h = [1, 2, 3, 1, 2, 3]
    >>> u = [0, 0, 0, 1, 1, 1]
    >>> g = st_model_variogram(h, u, "metric", joint={"psill": 2.0, "model": "Exp", "range": 2.0}, stani=1.5)
    >>> r = st_fit(h, u, g, [5] * 6, "metric", [1.0, 1.0, 0.1, 1.0])
    >>> [round(v, 6) for v in r.par]
    [2.0, 2.0, 0.0, 1.5]
    """
    H, U, G, N = _vec(dist), _vec(timelag), _vec(gamma), _vec(np_)
    keep = [i for i in range(len(G)) if N[i] > 0 and G[i] == G[i]]
    H, U, G, N = [H[i] for i in keep], [U[i] for i in keep], [G[i] for i in keep], [N[i] for i in keep]
    fixed = {"space_model": space_model, "time_model": time_model, "joint_model": joint_model}
    min_s = min(h for h in H if h > 0) * 0.05
    min_t = min(h for h, u in zip(H, U) if u > 0) * 0.05
    lo = {
        "separable": [min_s, 0, min_t, 0, 0],
        "productSum": [0, min_s, 0, 0, min_t, 0, _POS],
        "metric": [0, _POS, 0, _POS],
        "sumMetric": [0, min_s, 0, 0, min_t, 0, 0, _POS, 0, _POS],
    }[model]
    hi = [math.inf, 1, math.inf, 1, math.inf] if model == "separable" else [math.inf] * len(lo)
    lo = list(lower) if lower is not None else lo
    hi = list(upper) if upper is not None else hi

    def gm(p):
        return st_model_variogram(H, U, model, **_unpack(model, p, fixed))

    def weights(p, gmod):
        if fit_method == 1:
            return N
        if fit_method == 2:
            return [n / g**2 for n, g in zip(N, gmod)]
        if fit_method == 6:
            return [1.0] * len(N)
        if fit_method == 7:
            a = stani if stani is not None else _unpack(model, p, fixed).get("stani", 1.0)
            return [n / (h * h + (a * u) ** 2) for n, h, u in zip(N, H, U)]
        raise ValueError("fit_method must be 1, 2, 6 or 7")

    def obj(p):
        gmod = gm(p)
        return ssum(w * (g - m) ** 2 for w, g, m in zip(weights(p, gmod), G, gmod)) / len(G)

    res = lbfgsb_minimize(obj, [float(v) for v in start], lower=lo, upper=hi, pgtol=1e-12, factr=0.0, max_iter=2000)
    par = [float(v) for v in res.x]
    gmod = gm(par)
    sse = obj(par) * len(G)
    n, k = len(G), len(par)
    return RichResult(
        payload={
            "par": par,
            "model": _unpack(model, par, fixed),
            "objective": obj(par),
            "mse": ssum((g - m) ** 2 for g, m in zip(G, gmod)) / n,
            "aic": n * math.log(sse / n) + 2 * k,
            "bic": n * math.log(sse / n) + k * math.log(n),
            "converged": res.converged,
            "fitted": gmod,
        }
    )


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _krige(C, c0s, C00s, zv, X=None, X0=None, beta=None):
    """Simple (``beta``), ordinary (``X`` None) or universal kriging; returns predictions, variances, weights."""
    n = len(zv)
    Ci = [[float(v) for v in r] for r in inverse(C)]
    if beta is not None:
        X = None
    elif X is None:
        X = [[1.0] for _ in range(n)]
        X0 = [[1.0] for _ in c0s]
    preds, vars_, wts = [], [], []
    if X is not None:
        p = len(X[0])
        CiX = [[ssum(Ci[i][j] * X[j][a] for j in range(n)) for a in range(p)] for i in range(n)]
        A = [[ssum(X[i][a] * CiX[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        Ai = [[float(v) for v in r] for r in inverse(A)]
        Ciz = [ssum(Ci[i][j] * zv[j] for j in range(n)) for i in range(n)]
        bgls = [ssum(Ai[a][b] * ssum(X[i][b] * Ciz[i] for i in range(n)) for b in range(p)) for a in range(p)]
    for k, (c0, C00) in enumerate(zip(c0s, C00s)):
        Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
        if X is None:
            w = Cic0
            pred = beta + ssum(w[i] * (zv[i] - beta) for i in range(n))
            var = C00 - ssum(c0[i] * Cic0[i] for i in range(n))
        else:
            d = [X0[k][a] - ssum(X[i][a] * Cic0[i] for i in range(n)) for a in range(p)]
            lam = [ssum(Ai[a][b] * d[b] for b in range(p)) for a in range(p)]
            w = [Cic0[i] + ssum(CiX[i][a] * lam[a] for a in range(p)) for i in range(n)]
            pred = ssum(w[i] * zv[i] for i in range(n))
            var = C00 - ssum(c0[i] * Cic0[i] for i in range(n)) + ssum(d[a] * lam[a] for a in range(p))
        preds.append(pred)
        vars_.append(var)
        wts.append(w)
    return preds, vars_, wts, (bgls if X is not None else None)


def st_universal_kriging(z, X, coords, times, X0, new_coords, new_times, model: dict) -> RichResult:
    r"""Space-time universal kriging with drift ``Z(s, t) = x(s, t)'beta + e(s, t)``, as ``gstat::krigeST(z ~ x, ...)``.

    With the space-time covariance ``C`` of :func:`~morie.fn.stkrig.st_covariance`
    the prediction is ``x_0'beta_GLS + c_0'C^-1 (z - X beta_GLS)`` and the
    variance ``C_00 - c_0'C^-1 c_0 + d'(X'C^-1 X)^-1 d``, ``d = x_0 - X'C^-1
    c_0``. ``X`` and ``X0`` include the intercept column.

    References
    ----------
    Graeler, B., Pebesma, E. and Heuvelink, G. (2016). Spatio-temporal
    interpolation using gstat. *The R Journal*, 8(1), 204-218.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> r = st_universal_kriging([1.0, 2.0, 1.5, 1.2], [[1, 0], [1, 1], [1, 0], [1, 1]], [(0, 0), (1, 0), (0, 1), (1, 1)],
    ...                          [0, 0, 1, 1], [[1, 0]], [(0, 0)], [0], m)
    >>> round(r.prediction[0], 10), round(r.variance[0], 12)
    (1.0, 0.0)
    """
    zv = _vec(z)
    P, T, Q, S = _pts(coords), _vec(times), _pts(new_coords), _vec(new_times)
    C = [[st_covariance(_dist(P[i], P[j]), T[i] - T[j], model) for j in range(len(P))] for i in range(len(P))]
    c0s = [[st_covariance(_dist(P[i], Q[k]), T[i] - S[k], model) for i in range(len(P))] for k in range(len(Q))]
    C00 = [st_covariance(0.0, 0.0, model) for _ in Q]
    pr, va, _, b = _krige(C, c0s, C00, zv, [_vec(r) for r in X], [_vec(r) for r in X0])
    return RichResult(payload={"prediction": pr, "variance": va, "beta": b})


def st_local_kriging(
    z,
    coords,
    times,
    new_coords,
    new_times,
    model: dict,
    *,
    nmax: int,
    stani: float,
    buffer: float = 2.0,
    beta: float | None = None,
) -> RichResult:
    r"""Space-time kriging in a local neighbourhood, as ``gstat::krigeST(nmax = , stAni = )``.

    For each target the ``ceiling(buffer nmax)`` observations nearest in the
    metric ``sqrt(dx^2 + dy^2 + (stani dt)^2)`` are candidates, of which the
    ``nmax`` with the largest covariance to the target are kept (ties by
    index) for ordinary (or simple, ``beta``) kriging.

    References
    ----------
    Graeler, B., Pebesma, E. and Heuvelink, G. (2016). Spatio-temporal
    interpolation using gstat. *The R Journal*, 8(1), 204-218.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> r = st_local_kriging([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], [(0.1, 0)], [0],
    ...                      m, nmax=2, stani=1.0)
    >>> r.neighbours
    [[0, 1]]
    """
    zv = _vec(z)
    P, T, Q, S = _pts(coords), _vec(times), _pts(new_coords), _vec(new_times)
    n = len(zv)
    m = min(n, int(math.ceil(buffer * nmax)))
    pr, va, nb = [], [], []
    for q, s in zip(Q, S):
        d = [math.sqrt((P[i][0] - q[0]) ** 2 + (P[i][1] - q[1]) ** 2 + (stani * (T[i] - s)) ** 2) for i in range(n)]
        cand = sorted(sorted(range(n), key=lambda i: (d[i], i))[:m])
        if m > nmax:
            cv = [st_covariance(_dist(P[i], q), T[i] - s, model) for i in cand]
            keep = sorted(cand[j] for j in sorted(range(len(cand)), key=lambda j: (-cv[j], j))[:nmax])
        else:
            keep = cand
        C = [[st_covariance(_dist(P[i], P[j]), T[i] - T[j], model) for j in keep] for i in keep]
        c0 = [[st_covariance(_dist(P[i], q), T[i] - s, model) for i in keep]]
        p, v, _, _ = _krige(C, c0, [st_covariance(0.0, 0.0, model)], [zv[i] for i in keep], beta=beta)
        pr.append(p[0])
        va.append(v[0])
        nb.append(keep)
    return RichResult(payload={"prediction": pr, "variance": va, "neighbours": nb})


def st_block_kriging(
    z,
    coords,
    times,
    new_coords,
    new_times,
    model: dict,
    *,
    block: float,
    duration: float,
    n_space: int = 4,
    n_time: int = 3,
    beta: float | None = None,
) -> RichResult:
    r"""Space-time block kriging of the mean over a square spatial block (side ``block``) and time window (``duration``).

    The block is discretised into ``n_space x n_space`` cell centres and
    ``n_time`` equally spaced instants; point-to-block covariances are block
    averages and the block variance ``C_BB`` the average over all pairs of
    discretisation points (Journel and Huijbregts 1978). Ordinary kriging
    unless ``beta``; ``n_space = n_time = 1`` is point kriging.

    References
    ----------
    Journel, A. G. and Huijbregts, C. J. (1978). *Mining Geostatistics*.
    Academic Press.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> a = st_block_kriging([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], [(0.5, 0.5)], [0.5], m,
    ...                      block=1.0, duration=1.0)
    >>> b = st_block_kriging([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], [(0.5, 0.5)], [0.5], m,
    ...                      block=1.0, duration=1.0, n_space=1, n_time=1)
    >>> a.variance[0] < b.variance[0]
    True
    """
    zv = _vec(z)
    P, T, Q, S = _pts(coords), _vec(times), _pts(new_coords), _vec(new_times)
    n = len(zv)
    offs = [(-0.5 + (i + 0.5) / n_space) * block for i in range(n_space)]
    toffs = [(-0.5 + (i + 0.5) / n_time) * duration for i in range(n_time)]
    C = [[st_covariance(_dist(P[i], P[j]), T[i] - T[j], model) for j in range(n)] for i in range(n)]
    c0s, C00s = [], []
    for q, s in zip(Q, S):
        pts = [((q[0] + a, q[1] + b), s + c) for a in offs for b in offs for c in toffs]
        c0s.append([ssum(st_covariance(_dist(P[i], p), T[i] - t, model) for p, t in pts) / len(pts) for i in range(n)])
        C00s.append(
            ssum(st_covariance(_dist(p1, p2), t1 - t2, model) for p1, t1 in pts for p2, t2 in pts) / len(pts) ** 2
        )
    pr, va, _, _ = _krige(C, c0s, C00s, zv, beta=beta)
    return RichResult(payload={"prediction": pr, "variance": va, "block_variance": C00s})


def st_leave_h_out(z, coords, times, model: dict, *, h: float, tau: float, beta: float | None = None) -> RichResult:
    r"""Leave-h-out (buffered) cross-validation of space-time kriging.

    Each observation is predicted from the observations farther than ``h``
    in space or ``tau`` in time (the target and its space-time buffer are
    removed), which counters the optimism of leave-one-out validation under
    autocorrelation (Le Rest et al. 2014). ``h = tau = 0`` is leave-one-out.

    References
    ----------
    Le Rest, K., Pinaud, D., Monestiez, P., Chadoeuf, J. and Bretagnolle, V.
    (2014). Spatial leave-one-out cross-validation for variable selection in
    the presence of spatial autocorrelation. *Global Ecology and
    Biogeography*, 23(7), 811-820.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> round(st_leave_h_out([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], m, h=0, tau=0).rmse, 6)
    0.593837
    """
    zv = _vec(z)
    P, T = _pts(coords), _vec(times)
    n = len(zv)
    pred, var, used = [], [], []
    for i in range(n):
        keep = [j for j in range(n) if _dist(P[i], P[j]) > h or abs(T[i] - T[j]) > tau]
        C = [[st_covariance(_dist(P[a], P[b]), T[a] - T[b], model) for b in keep] for a in keep]
        c0 = [[st_covariance(_dist(P[a], P[i]), T[a] - T[i], model) for a in keep]]
        p, v, _, _ = _krige(C, c0, [st_covariance(0.0, 0.0, model)], [zv[a] for a in keep], beta=beta)
        pred.append(p[0])
        var.append(v[0])
        used.append(len(keep))
    res = [zv[i] - pred[i] for i in range(n)]
    return RichResult(
        payload={
            "prediction": pred,
            "variance": var,
            "residuals": res,
            "n_used": used,
            "rmse": math.sqrt(ssum(v * v for v in res) / n),
            "msdr": ssum(res[i] ** 2 / var[i] for i in range(n)) / n,
        }
    )


def st_kriging_diagnostics(coords, times, new_coords, new_times, model: dict) -> RichResult:
    r"""Diagnostics of the ordinary space-time kriging system: conditioning, weights and relative variance.

    Returns the 2-norm condition number of the covariance matrix (ratio of
    extreme eigenvalues), and per target the kriging weights, their sum
    (1), the share and total of negative weights, the Lagrange multiplier
    and the kriging variance relative to the sill ``C(0, 0)``.

    References
    ----------
    Isaaks, E. H. and Srivastava, R. M. (1989). *An Introduction to Applied
    Geostatistics*. Oxford University Press.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> d = st_kriging_diagnostics([(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], [(0, 0)], [0], m)
    >>> [round(w, 10) for w in d.weights[0]], round(d.relative_variance[0], 12)
    ([1.0, 0.0, 0.0, 0.0], 0.0)
    """
    P, T, Q, S = _pts(coords), _vec(times), _pts(new_coords), _vec(new_times)
    n = len(P)
    C = [[st_covariance(_dist(P[i], P[j]), T[i] - T[j], model) for j in range(n)] for i in range(n)]
    ev = [float(v) for v in np.linalg.eigh(np.asarray(C, dtype=float))[0].tolist()]
    Ci = [[float(v) for v in r] for r in inverse(C)]
    s1 = ssum(ssum(r) for r in Ci)
    sill = st_covariance(0.0, 0.0, model)
    W, lag, rel, neg, negsum = [], [], [], [], []
    for q, s in zip(Q, S):
        c0 = [st_covariance(_dist(P[i], q), T[i] - s, model) for i in range(n)]
        Cic0 = [ssum(Ci[i][j] * c0[j] for j in range(n)) for i in range(n)]
        mu = (1 - ssum(Cic0)) / s1
        w = [Cic0[i] + mu * ssum(Ci[i]) for i in range(n)]
        var = sill - ssum(c0[i] * Cic0[i] for i in range(n)) + (1 - ssum(Cic0)) ** 2 / s1
        W.append(w)
        lag.append(mu)
        rel.append(var / sill)
        neg.append(sum(1 for v in w if v < 0) / n)
        negsum.append(ssum(v for v in w if v < 0))
    return RichResult(
        payload={
            "condition_number": max(ev) / min(ev),
            "eigenvalues": ev,
            "weights": W,
            "weight_sums": [ssum(w) for w in W],
            "negative_share": neg,
            "negative_total": negsum,
            "lagrange": lag,
            "relative_variance": rel,
        }
    )


def _order(c):
    if c is None:
        return math.inf
    kind = c.get("model", "Exp")
    if kind == "Gau":
        return math.inf
    if kind == "Mat":
        k = float(c.get("kappa", 0.5))
        return math.ceil(k) - 1
    if kind in ("Exp", "Sph", "Lin"):
        return 0
    return math.inf


def st_smoothness(model: dict) -> RichResult:
    r"""Mean-square continuity and differentiability of a ``vgmST`` model in space and in time.

    A stationary field is mean-square differentiable ``m`` times along a
    direction iff its covariance is ``2m`` times differentiable at the origin
    (Stein 1999): Gaussian components are infinitely smooth, Matern with
    ``kappa`` ``ceil(kappa) - 1`` times, exponential, spherical and linear
    not at all (order 0), and a nugget makes the field mean-square
    discontinuous (order -1). A process built from several components (or
    the metric joint component, which acts in both directions) has the
    minimum order.

    References
    ----------
    Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for
    Kriging*. Springer.

    Examples
    --------
    >>> st_smoothness({"type": "separable", "space": {"model": "Gau"}, "time": {"model": "Mat", "kappa": 2.5}}).space
    inf
    """
    typ = model.get("type")
    sp = [model.get("space")] if typ in ("separable", "productSum", "sumMetric") else []
    tm = [model.get("time")] if typ in ("separable", "productSum", "sumMetric") else []
    if typ in ("metric", "sumMetric"):
        sp.append(model.get("joint"))
        tm.append(model.get("joint"))

    def order(cs):
        o = min(_order(c) for c in cs)
        return -1 if any(float(c.get("nugget", 0.0)) > 0 for c in cs if c) else o

    osp, otm = order(sp), order(tm)
    return RichResult(payload={"space": osp, "time": otm, "continuous_space": osp >= 0, "continuous_time": otm >= 0})


def st_predict_grid(z, coords, times, model: dict, xs, ys, new_times, *, beta: float | None = None) -> RichResult:
    r"""Space-time kriging on a regular grid for each of ``new_times`` (animation frames).

    ``frames[t][r][c]`` predicts at ``(xs[c], ys[r], new_times[t])`` (and
    ``variance`` likewise) by global ordinary (or simple) space-time kriging.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> g = st_predict_grid([1.0, 2.0, 1.5, 1.2], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], m, [0, 1], [0], [0])
    >>> [round(v, 10) for v in g.frames[0][0]]
    [1.0, 2.0]
    """
    zv = _vec(z)
    P, T = _pts(coords), _vec(times)
    X, Y, S = _vec(xs), _vec(ys), _vec(new_times)
    n = len(P)
    C = [[st_covariance(_dist(P[i], P[j]), T[i] - T[j], model) for j in range(n)] for i in range(n)]
    targets = [((x, y), s) for s in S for y in Y for x in X]
    c0s = [[st_covariance(_dist(P[i], q), T[i] - s, model) for i in range(n)] for q, s in targets]
    pr, va, _, _ = _krige(C, c0s, [st_covariance(0.0, 0.0, model)] * len(targets), zv, beta=beta)
    nx, ny = len(X), len(Y)
    frames = [[pr[t * nx * ny + r * nx : t * nx * ny + (r + 1) * nx] for r in range(ny)] for t in range(len(S))]
    varf = [[va[t * nx * ny + r * nx : t * nx * ny + (r + 1) * nx] for r in range(ny)] for t in range(len(S))]
    return RichResult(payload={"frames": frames, "variance": varf})


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 0.0)) if i == j else (s / L[j][j] if L[j][j] > 0 else 0.0)
    return L


def st_simulate(
    coords,
    times,
    model: dict,
    *,
    nsim: int = 1,
    seed: int = 1,
    mean: float = 0.0,
    z=None,
    data_coords=None,
    data_times=None,
) -> RichResult:
    r"""Gaussian space-time random field simulation by Cholesky factorisation, unconditional or conditioned on data.

    Unconditional: ``mean + L e`` with ``C = L L'`` over the targets and
    ``e`` Philox normals (stream ``r`` for realisation ``r``). Conditional on
    ``z`` observed at ``data_coords``/``data_times``: jointly simulate at
    data and targets and correct by simple kriging (known ``mean``) of the
    residuals, ``Z_c = Z_s + SK(z - Z_s(data))`` (Journel and Huijbregts
    1978), which honours the data and has the conditional covariance.

    References
    ----------
    Journel, A. G. and Huijbregts, C. J. (1978). *Mining Geostatistics*.
    Academic Press.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> s = st_simulate([(0, 0), (1, 0)], [0, 0], m, z=[2.0], data_coords=[(0, 0)], data_times=[0])
    >>> round(s.realisations[0][0], 10)
    2.0
    """
    Q, S = _pts(coords), _vec(times)
    cond = z is not None
    D, DT = (_pts(data_coords), _vec(data_times)) if cond else ([], [])
    allp = D + Q
    allt = DT + S
    N, nd = len(allp), len(D)
    C = [[st_covariance(_dist(allp[i], allp[j]), allt[i] - allt[j], model) for j in range(N)] for i in range(N)]
    L = _chol(C)
    out = []
    if cond:
        zv = _vec(z)
        Cd = [row[:nd] for row in C[:nd]]
        Cdi = [[float(v) for v in r] for r in inverse(Cd)]
        lam = [[ssum(C[nd + k][a] * Cdi[a][b] for a in range(nd)) for b in range(nd)] for k in range(len(Q))]
    for r in range(nsim):
        e = [float(v) for v in random_normal(N, seed=seed, stream=r)]
        x = [mean + ssum(L[i][k] * e[k] for k in range(i + 1)) for i in range(N)]
        if cond:
            res = [zv[a] - x[a] for a in range(nd)]
            out.append([x[nd + k] + ssum(lam[k][b] * res[b] for b in range(nd)) for k in range(len(Q))])
        else:
            out.append(x)
    return RichResult(payload={"realisations": out, "cholesky": L})


def cheatsheet() -> str:
    return (
        "st_fit / st_universal_kriging / st_local_kriging / st_block_kriging / st_leave_h_out / "
        "st_kriging_diagnostics / st_smoothness / st_predict_grid / st_simulate -> space-time geostatistics."
    )
