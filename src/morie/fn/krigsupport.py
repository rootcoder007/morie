# morie.fn -- function file (rootcoder007/morie)
"""Kriging with change of support, nonstationarity and variogram uncertainty: area-to-point kriging,
Paciorek-Schervish nonstationary kriging, weighted least-squares and genetic-algorithm variogram fitting,
and empirical Bayesian kriging."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._s03core import chol
from .krgsys import krige, kriging_covariance

__all__ = [
    "area_to_point_kriging",
    "nonstationary_kriging",
    "empirical_variogram_bins",
    "fit_variogram_wls",
    "ga_kriging",
    "empirical_bayesian_kriging",
]


def _rows(a):
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in a]


def _dist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) * (x - y)
    return math.sqrt(s)


def _mv(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _ok(zv, C, c0, c00):
    """Ordinary kriging from a covariance matrix: (prediction, variance, weights)."""
    n = len(zv)
    Ci = inverse(C)
    Cic0 = _mv(Ci, c0)
    one = [ssum(Ci[i]) for i in range(n)]
    a = ssum(one)
    r = 1.0 - ssum(Cic0)
    lam = [Cic0[i] + one[i] * r / a for i in range(n)]
    return ssum(lam[i] * zv[i] for i in range(n)), c00 - ssum(c0[i] * Cic0[i] for i in range(n)) + r * r / a, lam


def area_to_point_kriging(values, supports, new_coords, model, *, mean: float | None = None) -> RichResult:
    r"""Area-to-point kriging (Kyriakidis 2004): point predictions from areal averages, coherent with the areal data.

    Area ``k`` is discretised by the points ``supports[k]`` (equal weights),
    so ``cov(Z(v_k), Z(v_l))`` is the mean of point covariances over the two
    discretisations and ``cov(Z(v_k), Z(s_0))`` the mean over ``v_k``.
    Ordinary kriging (weights summing to one) by default, simple kriging with
    ``mean``. Without a nugget, averaging the point predictions over the
    discretisation of an area returns its datum (the coherence property that
    makes it a downscaling method).

    References
    ----------
    Kyriakidis, P. C. (2004). A geostatistical framework for area-to-point
    spatial interpolation. *Geographical Analysis*, 36, 259-289.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> sup = [[(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)], [(1.25, 0.25), (1.75, 0.25), (1.25, 0.75), (1.75, 0.75)]]
    >>> r = area_to_point_kriging([1.0, 3.0], sup, sup[0], m)
    >>> round(sum(r.prediction) / 4, 12)
    1.0
    """
    zv = [float(v) for v in values]
    D = [_rows(s) for s in supports]
    Q = _rows(new_coords)
    K = len(zv)
    C = [
        [
            ssum(kriging_covariance(_dist(p, q), model) for p in D[a] for q in D[b]) / (len(D[a]) * len(D[b]))
            for b in range(K)
        ]
        for a in range(K)
    ]
    c00 = kriging_covariance(0.0, model)
    pred, var, W = [], [], []
    for s in Q:
        c0 = [ssum(kriging_covariance(_dist(p, s), model) for p in D[a]) / len(D[a]) for a in range(K)]
        if mean is None:
            pk, vk, lam = _ok(zv, C, c0, c00)
        else:
            lam = _mv(inverse(C), c0)
            pk = mean + ssum(lam[i] * (zv[i] - mean) for i in range(K))
            vk = c00 - ssum(lam[i] * c0[i] for i in range(K))
        pred.append(pk)
        var.append(vk)
        W.append(lam)
    return RichResult(payload={"prediction": pred, "variance": var, "weights": W})


def _ns_cov(a, b, la, lb, sigma2, nugget, d):
    if _dist(a, b) == 0.0 and la == lb:
        return sigma2 + nugget
    s = la * la + lb * lb
    q = math.sqrt(2.0 * _dist(a, b) ** 2 / s)
    return sigma2 * (2.0 * la * lb / s) ** (d / 2.0) * math.exp(-q)


def nonstationary_kriging(z, coords, new_coords, ell, ell_new, sigma2: float, *, nugget: float = 0.0) -> RichResult:
    r"""Ordinary kriging with the Paciorek-Schervish nonstationary exponential covariance.

    With a spatially varying isotropic kernel scale ``l(s)`` (``ell`` at the
    data, ``ell_new`` at the targets) in ``d`` dimensions,
    ``C(s, s') = sigma2 (2 l l' / (l^2 + l'^2))^{d/2} exp(-sqrt(Q))``,
    ``Q = 2 ||s - s'||^2 / (l^2 + l'^2)``, positive definite for any ``l(.)``;
    constant ``l`` gives the exponential model with range ``l``. ``nugget``
    is added at coincident locations.

    References
    ----------
    Paciorek, C. J. and Schervish, M. J. (2006). Spatial modelling using a
    new class of nonstationary covariance functions. *Environmetrics*, 17, 483-506.

    Examples
    --------
    >>> r = nonstationary_kriging([1.0, 2.0, 0.5], [(0, 0), (1, 0), (0, 1)], [(0.3, 0.3)], [0.5, 1.0, 2.0], [1.0], 1.0)
    >>> round(r.prediction[0], 10)
    1.1366813821
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    n, d = len(zv), len(P[0])
    L = [float(v) for v in ell]
    Ln = [float(v) for v in ell_new]
    C = [[_ns_cov(P[i], P[j], L[i], L[j], sigma2, nugget if i == j else 0.0, d) for j in range(n)] for i in range(n)]
    pred, var = [], []
    for k, q in enumerate(Q):
        c0 = [_ns_cov(P[i], q, L[i], Ln[k], sigma2, 0.0, d) for i in range(n)]
        pk, vk, _ = _ok(zv, C, c0, sigma2 + nugget)
        pred.append(pk)
        var.append(vk)
    return RichResult(payload={"prediction": pred, "variance": var})


def empirical_variogram_bins(z, coords, *, cutoff: float | None = None, width: float | None = None) -> RichResult:
    r"""Matheron's sample variogram ``gamma(h_j) = sum (z_i - z_k)^2 / (2 N_j)`` in distance bins (gstat defaults).

    ``cutoff`` defaults to one third of the bounding-box diagonal and ``width``
    to ``cutoff / 15``; ``dist`` is the mean pair distance in each non-empty
    bin (left-open, right-closed intervals, the first including 0).

    Examples
    --------
    >>> r = empirical_variogram_bins([0.0, 1.0, 3.0], [(0,), (1,), (2,)], cutoff=2.0, width=1.0)
    >>> r.np, r.gamma
    ([2, 1], [1.25, 4.5])
    """
    zv = [float(v) for v in z]
    P = _rows(coords)
    n = len(zv)
    pairs = [(_dist(P[i], P[j]), 0.5 * (zv[i] - zv[j]) ** 2) for i in range(n) for j in range(i + 1, n)]
    if cutoff is None:  # gstat: one third of the bounding-box diagonal
        cutoff = math.sqrt(ssum((max(p[t] for p in P) - min(p[t] for p in P)) ** 2 for t in range(len(P[0])))) / 3.0
    width = cutoff / 15.0 if width is None else width
    nb = int(math.ceil(cutoff / width - 1e-12))
    sh, sg, cnt = [0.0] * nb, [0.0] * nb, [0] * nb
    for h, g in pairs:
        if h > cutoff:
            continue
        b = min(max(int(math.ceil(h / width)) - 1, 0), nb - 1)
        sh[b] += h
        sg[b] += g
        cnt[b] += 1
    keep = [b for b in range(nb) if cnt[b] > 0]
    return RichResult(
        payload={
            "dist": [sh[b] / cnt[b] for b in keep],
            "gamma": [sg[b] / cnt[b] for b in keep],
            "np": [cnt[b] for b in keep],
        }
    )


def _rho(h, a, kind):
    t = h / a
    if kind == "Exp":
        return math.exp(-t)
    if kind == "Gau":
        return math.exp(-t * t)
    if kind == "Sph":
        return 1 - 1.5 * t + 0.5 * t**3 if t < 1 else 0.0
    raise ValueError("model must be Exp, Gau or Sph")


def _wls_given_range(ev, a, kind):
    """Best non-negative (nugget, psill) for range a by weighted least squares (weights N_j / h_j^2)."""
    h, g, w = ev["dist"], ev["gamma"], [ev["np"][j] / ev["dist"][j] ** 2 for j in range(len(ev["dist"]))]
    f = [1.0 - _rho(v, a, kind) for v in h]
    s11, s12, s22 = ssum(w), ssum(w[j] * f[j] for j in range(len(h))), ssum(w[j] * f[j] * f[j] for j in range(len(h)))
    t1, t2 = ssum(w[j] * g[j] for j in range(len(h))), ssum(w[j] * f[j] * g[j] for j in range(len(h)))
    det = s11 * s22 - s12 * s12
    if det <= 1e-12 * s11 * s22:  # sill shape constant over the bins: one free parameter
        c0, c1 = (0.0, max(t2 / s22, 0.0)) if s22 > 0 else (max(t1 / s11, 0.0), 0.0)
    else:
        c0, c1 = (s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det
    if c0 < 0:
        c0, c1 = 0.0, max(t2 / s22, 0.0)
    elif c1 < 0:
        c0, c1 = max(t1 / s11, 0.0), 0.0
    sse = ssum(w[j] * (g[j] - c0 - c1 * f[j]) ** 2 for j in range(len(h)))
    return c0, c1, sse


def _sse(ev, c0, c1, a, kind):
    h, g = ev["dist"], ev["gamma"]
    return ssum(ev["np"][j] / h[j] ** 2 * (g[j] - c0 - c1 * (1.0 - _rho(h[j], a, kind))) ** 2 for j in range(len(h)))


def fit_variogram_wls(ev, model: str = "Exp") -> RichResult:
    r"""Weighted least-squares fit of a nugget + ``Exp``/``Gau``/``Sph`` variogram (gstat ``fit.method = 7``).

    Minimises ``sum_j N_j / h_j^2 (gamma_j - c0 - c1 (1 - rho(h_j / a)))^2``;
    for fixed range ``a`` the sills are a non-negative linear least-squares
    problem, and ``a`` is profiled by a log-scale grid followed by golden-section search.

    References
    ----------
    Cressie, N. (1985). Fitting variogram models by weighted least squares.
    *Mathematical Geology*, 17, 563-586.

    Examples
    --------
    >>> ev = {"dist": [0.5, 1.0, 1.5, 2.0], "gamma": [0.45, 0.72, 0.86, 0.93], "np": [10, 12, 14, 9]}
    >>> r = fit_variogram_wls(ev)
    >>> round(r.nugget, 4), round(r.psill, 4), round(r.range, 4)
    (0.0, 1.0589, 0.8973)
    """
    ev = {"dist": list(ev["dist"]), "gamma": list(ev["gamma"]), "np": list(ev["np"])}
    hmin, hmax = min(ev["dist"]), max(ev["dist"])
    lo, hi = math.log(hmin / 20.0), math.log(hmax * 20.0)
    grid = [lo + (hi - lo) * i / 99 for i in range(100)]
    ss = [_wls_given_range(ev, math.exp(t), model)[2] for t in grid]
    b = min(range(100), key=lambda i: (ss[i], i))
    a_, b_ = grid[max(b - 1, 0)], grid[min(b + 1, 99)]
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    f1, f2 = _wls_given_range(ev, math.exp(x1), model)[2], _wls_given_range(ev, math.exp(x2), model)[2]
    for _ in range(200):
        if f1 <= f2:
            b_, x2, f2 = x2, x1, f1
            x1 = b_ - gr * (b_ - a_)
            f1 = _wls_given_range(ev, math.exp(x1), model)[2]
        else:
            a_, x1, f1 = x1, x2, f2
            x2 = a_ + gr * (b_ - a_)
            f2 = _wls_given_range(ev, math.exp(x2), model)[2]
        if b_ - a_ < 1e-13:
            break
    a = math.exp(0.5 * (a_ + b_))
    c0, c1, sse = _wls_given_range(ev, a, model)
    return RichResult(payload={"nugget": c0, "psill": c1, "range": a, "sse": sse, "model": model})


def _as_model(fit):
    return {"model": fit["model"], "psill": fit["psill"], "range": fit["range"], "nugget": fit["nugget"]}


def ga_kriging(
    z, coords, new_coords, *, model: str = "Exp", pop: int = 40, generations: int = 60, seed: int = 1, ev=None
) -> RichResult:
    r"""Kriging with a variogram fitted by a real-coded genetic algorithm (GA-kriging).

    Individuals ``(c0, c1, log a)`` in bounds ``[0, 2 max gamma]^2 x [log(h_min / 20), log(20 h_max)]``
    minimise the weighted least-squares criterion of :func:`fit_variogram_wls`;
    each generation keeps the best individual (elitism) and breeds the rest
    by binary tournaments, arithmetic crossover with a uniform weight and
    Gaussian mutation (sd one tenth of the bound width, clipped), all on
    Philox draws. The fitted model then drives ordinary kriging
    (:func:`morie.fn.krgsys.krige`).

    References
    ----------
    Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
    Cressie, N. (1985). Fitting variogram models by weighted least squares. *Mathematical Geology*, 17, 563-586.

    Examples
    --------
    >>> ev = {"dist": [0.5, 1.0, 1.5, 2.0], "gamma": [0.45, 0.72, 0.86, 0.93], "np": [10, 12, 14, 9]}
    >>> r = ga_kriging([1.0, 2.0, 0.5], [(0, 0), (1, 0), (0, 1)], [(0.3, 0.3)], ev=ev)
    >>> round(r.fit["range"], 2)
    0.79
    """
    zv = [float(v) for v in z]
    P = _rows(coords)
    ev = empirical_variogram_bins(zv, P) if ev is None else ev
    ev = {"dist": list(ev["dist"]), "gamma": list(ev["gamma"]), "np": list(ev["np"])}
    gmax = 2.0 * max(ev["gamma"])
    lo = [0.0, 0.0, math.log(min(ev["dist"]) / 20.0)]
    hi = [gmax, gmax, math.log(max(ev["dist"]) * 20.0)]
    u = [float(v) for v in random_uniform(pop * 3 + generations * pop * 5, seed=seed)]
    nz = [float(v) for v in random_normal(generations * pop * 3, seed=seed, stream=1)]
    iu, inn = 0, 0
    X = []
    for _ in range(pop):
        X.append([lo[t] + (hi[t] - lo[t]) * u[iu + t] for t in range(3)])
        iu += 3

    def fit(x):
        return _sse(ev, x[0], x[1], math.exp(x[2]), model)

    F = [fit(x) for x in X]
    for _ in range(generations):
        best = min(range(pop), key=lambda i: (F[i], i))
        new = [list(X[best])]
        while len(new) < pop:
            par = []
            for _t in range(2):
                a, b = min(int(u[iu] * pop), pop - 1), min(int(u[iu + 1] * pop), pop - 1)
                iu += 2
                par.append(X[a] if (F[a], a) <= (F[b], b) else X[b])
            w = u[iu]
            iu += 1
            child = []
            for t in range(3):
                v = w * par[0][t] + (1 - w) * par[1][t] + 0.1 * (hi[t] - lo[t]) * nz[inn]
                inn += 1
                child.append(min(max(v, lo[t]), hi[t]))
            new.append(child)
        X = new
        F = [fit(x) for x in X]
    best = min(range(pop), key=lambda i: (F[i], i))
    f = {"nugget": X[best][0], "psill": X[best][1], "range": math.exp(X[best][2]), "sse": F[best], "model": model}
    kr = krige(zv, P, _rows(new_coords), _as_model(f))
    return RichResult(payload={"prediction": kr.prediction, "variance": kr.variance, "fit": f})


def _gauss_loglik(zv, P, m):
    n = len(zv)
    C = [[kriging_covariance(_dist(P[i], P[j]), m) for j in range(n)] for i in range(n)]
    L = chol(C)
    mu = ssum(zv) / n
    y = []
    for i in range(n):
        y.append((zv[i] - mu - ssum(L[i][k] * y[k] for k in range(i))) / L[i][i])
    return -0.5 * n * math.log(2 * math.pi) - ssum(math.log(L[i][i]) for i in range(n)) - 0.5 * ssum(v * v for v in y)


def empirical_bayesian_kriging(
    z, coords, new_coords, *, nsim: int = 50, model: str = "Exp", seed: int = 1
) -> RichResult:
    r"""Empirical Bayesian kriging (Krivoruchko 2012): kriging averaged over a spectrum of simulated variograms.

    A variogram fitted to the data (:func:`fit_variogram_wls`) generates
    ``nsim`` unconditional Gaussian simulations at the data locations
    (Cholesky of the fitted covariance, Philox normals, data mean); each is
    refitted, giving a spectrum of variograms. Each is weighted by the
    Gaussian likelihood of the observed data (Bayes' rule, mean plugged in)
    and the prediction is the weighted mixture of the ordinary kriging
    predictions, with variance ``sum w_s (v_s + p_s^2) - p^2`` (the within-
    plus between-model variance). This is the global (single-subset) form.

    References
    ----------
    Krivoruchko, K. (2012). Empirical Bayesian kriging. *ArcUser*, Fall 2012, 6-10.
    Pilz, J. and Spoeck, G. (2008). Why do we need and how should we implement
    Bayesian kriging methods. *Stochastic Environmental Research and Risk
    Assessment*, 22, 621-632.

    Examples
    --------
    >>> pts = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> zz = [math.sin(x) + 0.3 * y for x, y in pts]
    >>> r = empirical_bayesian_kriging(zz, pts, [(1.5, 1.5)], nsim=5)
    >>> abs(r.prediction[0] - (math.sin(1.5) + 0.45)) < 0.3
    True
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    n = len(zv)
    base = fit_variogram_wls(empirical_variogram_bins(zv, P), model)
    m0 = _as_model(base)
    C = [[kriging_covariance(_dist(P[i], P[j]), m0) for j in range(n)] for i in range(n)]
    L = chol([[C[i][j] + (1e-10 if i == j else 0.0) for j in range(n)] for i in range(n)])
    mu = ssum(zv) / n
    fits, preds, varis, lls = [], [], [], []
    for s in range(nsim):
        e = [float(v) for v in random_normal(n, seed=seed, stream=s)]
        sim = [mu + ssum(L[i][k] * e[k] for k in range(i + 1)) for i in range(n)]
        fs = fit_variogram_wls(empirical_variogram_bins(sim, P), model)
        ms = _as_model(fs)
        if fs["psill"] + fs["nugget"] <= 0:
            continue
        kr = krige(zv, P, Q, ms)
        fits.append(fs)
        preds.append(kr.prediction)
        varis.append(kr.variance)
        lls.append(_gauss_loglik(zv, P, ms))
    mx = max(lls)
    w = [math.exp(v - mx) for v in lls]
    sw = ssum(w)
    w = [v / sw for v in w]
    pred, var = [], []
    for k in range(len(Q)):
        p = ssum(w[s] * preds[s][k] for s in range(len(w)))
        pred.append(p)
        var.append(ssum(w[s] * (varis[s][k] + preds[s][k] ** 2) for s in range(len(w))) - p * p)
    return RichResult(payload={"prediction": pred, "variance": var, "weights": w, "variograms": fits, "base": base})


def cheatsheet() -> str:
    return (
        "area_to_point_kriging / nonstationary_kriging / empirical_variogram_bins / fit_variogram_wls / ga_kriging / "
        "empirical_bayesian_kriging -> kriging with support change, nonstationarity and variogram uncertainty."
    )
