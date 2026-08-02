"""morie sci core: scipy.optimize / scipy.spatial / scipy.special subsets.

Native replacements for the non-stats scipy surfaces morie uses
(inventory: optimize.minimize 71+, spatial cdist/pdist/squareform ~100,
special.expit 24+).  Pure Python reference implementations; C kernels
in morie_core later.  Equivalence-tested against scipy in
tests/fn/test_sci_core.py.
"""

from __future__ import annotations

import math as _math

from . import _array_core as _ac


# ------------------------------------------------------------ special

def expit(x):
    def one(v):
        if v >= 0:
            return 1.0 / (1.0 + _math.exp(-v))
        e = _math.exp(v)
        return e / (1.0 + e)
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(one)
    return one(float(x))


def logit(p):
    def one(v):
        return _math.log(v / (1.0 - v))
    if isinstance(p, (list, tuple)) or hasattr(p, "tolist"):
        return _ac.asarray(p)._map(one)
    return one(float(p))


def gammaln(x):
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(_math.lgamma)
    return _math.lgamma(float(x))


def erf(x):
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(_math.erf)
    return _math.erf(float(x))


def erfc(x):
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(_math.erfc)
    return _math.erfc(float(x))


class special:  # namespace mirror for `from scipy import special`
    expit = staticmethod(expit)
    logit = staticmethod(logit)
    gammaln = staticmethod(gammaln)
    erf = staticmethod(erf)
    erfc = staticmethod(erfc)


# ------------------------------------------------------------ spatial

def _row(x, i):
    a = _ac.atleast_2d(x)
    return a.data[i]


def _metric_fn(metric):
    if metric in ("euclidean", None):
        return lambda u, v: _math.sqrt(_math.fsum(
            (a - b) ** 2 for a, b in zip(u, v)))
    if metric == "sqeuclidean":
        return lambda u, v: _math.fsum((a - b) ** 2 for a, b in zip(u, v))
    if metric == "cityblock":
        return lambda u, v: _math.fsum(abs(a - b) for a, b in zip(u, v))
    if metric == "chebyshev":
        return lambda u, v: max(abs(a - b) for a, b in zip(u, v))
    if metric == "cosine":
        def cos(u, v):
            nu = _math.sqrt(_math.fsum(a * a for a in u))
            nv = _math.sqrt(_math.fsum(b * b for b in v))
            dp = _math.fsum(a * b for a, b in zip(u, v))
            if nu == 0.0 or nv == 0.0:
                return float("nan")     # matches scipy (nan + warning)
            return 1.0 - dp / (nu * nv)
        return cos
    raise ValueError("unsupported metric %r" % metric)


def cdist(xa, xb, metric="euclidean"):
    a = _ac.atleast_2d(xa)
    b = _ac.atleast_2d(xb)
    fn = _metric_fn(metric)
    return _ac.marr([[fn(ra, rb) for rb in b.data] for ra in a.data])


def pdist(x, metric="euclidean"):
    a = _ac.atleast_2d(x)
    fn = _metric_fn(metric)
    out = []
    n = a.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            out.append(fn(a.data[i], a.data[j]))
    return _ac.marr(out)


def squareform(x):
    a = _ac.asarray(x)
    if len(a.shape) == 2:                       # square -> condensed
        n = a.shape[0]
        out = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                out.append(a.data[i][j])
        return _ac.marr(out)
    m = a.shape[0]                              # condensed -> square
    n = int(round((1 + _math.sqrt(1 + 8 * m)) / 2))
    if n * (n - 1) // 2 != m:
        raise ValueError("invalid condensed length")
    sq = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sq[i][j] = sq[j][i] = a.data[k]
            k += 1
    return _ac.marr(sq)


class _DistanceNS:
    cdist = staticmethod(cdist)
    pdist = staticmethod(pdist)
    squareform = staticmethod(squareform)


class spatial:  # namespace mirror
    distance = _DistanceNS()


# ------------------------------------------------------------ optimize

class OptimizeResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _nelder_mead(fun, x0, args=(), maxiter=None, xatol=1e-8, fatol=1e-8):
    n = len(x0)
    if maxiter is None:
        maxiter = 200 * n
    alpha, gamma_, rho_, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = [list(map(float, x0))]
    for i in range(n):
        pt = list(map(float, x0))
        pt[i] = pt[i] + (0.05 * pt[i] if pt[i] != 0 else 0.00025)
        simplex.append(pt)
    fvals = [float(fun(p, *args)) for p in simplex]
    nfev = n + 1
    for it in range(maxiter):
        order = sorted(range(n + 1), key=lambda k: fvals[k])
        simplex = [simplex[k] for k in order]
        fvals = [fvals[k] for k in order]
        spread = max(abs(fvals[k] - fvals[0]) for k in range(1, n + 1))
        width = max(max(abs(simplex[k][d] - simplex[0][d])
                        for d in range(n)) for k in range(1, n + 1))
        if spread <= fatol and width <= xatol:
            break
        centroid = [_math.fsum(simplex[k][d] for k in range(n)) / n
                    for d in range(n)]
        xr = [centroid[d] + alpha * (centroid[d] - simplex[-1][d])
              for d in range(n)]
        fr = float(fun(xr, *args))
        nfev += 1
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = [centroid[d] + gamma_ * (xr[d] - centroid[d])
                  for d in range(n)]
            fe = float(fun(xe, *args))
            nfev += 1
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = [centroid[d] + rho_ * (simplex[-1][d] - centroid[d])
              for d in range(n)]
        fc = float(fun(xc, *args))
        nfev += 1
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        for k in range(1, n + 1):
            simplex[k] = [simplex[0][d]
                          + sigma * (simplex[k][d] - simplex[0][d])
                          for d in range(n)]
            fvals[k] = float(fun(simplex[k], *args))
            nfev += n
    order = sorted(range(n + 1), key=lambda k: fvals[k])
    best = simplex[order[0]]
    return OptimizeResult(x=_ac.marr(best), fun=fvals[order[0]],
                          nit=it + 1, nfev=nfev,
                          success=True, message="nelder-mead converged")


def _num_grad(fun, x, args, eps=1e-7):
    f0 = float(fun(list(x), *args))
    g = []
    for i in range(len(x)):
        xp = list(x)
        h = eps * max(abs(xp[i]), 1.0)
        xp[i] += h
        g.append((float(fun(xp, *args)) - f0) / h)
    return g, f0


def _bfgs(fun, x0, args=(), maxiter=None, gtol=1e-6):
    n = len(x0)
    if maxiter is None:
        maxiter = 200 * n
    x = list(map(float, x0))
    hinv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    g, f = _num_grad(fun, x, args)
    nfev = n + 1
    for it in range(maxiter):
        gnorm = max(abs(v) for v in g)
        if gnorm < gtol:
            break
        # direction p = -Hinv g
        p = [-_math.fsum(hinv[i][j] * g[j] for j in range(n))
             for i in range(n)]
        # backtracking line search (Armijo)
        step = 1.0
        slope = _math.fsum(g[i] * p[i] for i in range(n))
        for _ in range(60):
            xn = [x[i] + step * p[i] for i in range(n)]
            fn_ = float(fun(xn, *args))
            nfev += 1
            if fn_ <= f + 1e-4 * step * slope:
                break
            step *= 0.5
        s = [xn[i] - x[i] for i in range(n)]
        gn, fn2 = _num_grad(fun, xn, args)
        nfev += n + 1
        yv = [gn[i] - g[i] for i in range(n)]
        sy = _math.fsum(s[i] * yv[i] for i in range(n))
        if sy > 1e-12:
            rho = 1.0 / sy
            # BFGS update: Hinv = (I - rho s y^T) Hinv (I - rho y s^T) + rho s s^T
            ihy = [[(1.0 if i == j else 0.0) - rho * s[i] * yv[j]
                    for j in range(n)] for i in range(n)]
            tmp = [[_math.fsum(ihy[i][k] * hinv[k][j] for k in range(n))
                    for j in range(n)] for i in range(n)]
            hinv = [[_math.fsum(tmp[i][k] * ((1.0 if k == j else 0.0)
                                             - rho * yv[k] * s[j])
                                for k in range(n)) + rho * s[i] * s[j]
                     for j in range(n)] for i in range(n)]
        x, g, f = xn, gn, fn2
    return OptimizeResult(x=_ac.marr(x), fun=f, nit=it + 1, nfev=nfev,
                          jac=_ac.marr(g),
                          success=max(abs(v) for v in g) < 1e-3,
                          message="bfgs")


def minimize(fun, x0, args=(), method=None, **kw):
    x0 = list(_ac.asarray(x0)._flat())
    if method is None:
        method = "BFGS"
    m = method.lower().replace("-", "")
    opts = kw.get("options", {}) or {}
    if m == "neldermead":
        return _nelder_mead(fun, x0, args=args,
                            maxiter=opts.get("maxiter"),
                            xatol=opts.get("xatol", 1e-8),
                            fatol=opts.get("fatol", 1e-8))
    if m in ("bfgs", "lbfgsb", "cg", "powell"):
        return _bfgs(fun, x0, args=args, maxiter=opts.get("maxiter"),
                     gtol=opts.get("gtol", 1e-6))
    raise ValueError("unsupported method %r" % method)


def minimize_scalar(fun, bounds=None, method=None, **kw):
    del method, kw
    lo, hi = (bounds if bounds else (-1e6, 1e6))
    # golden-section search
    gr = (_math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = float(fun(c)), float(fun(d))
    for _ in range(200):
        if abs(b - a) < 1e-10 * (abs(a) + abs(b) + 1.0):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(fun(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(fun(d))
    xm = 0.5 * (a + b)
    return OptimizeResult(x=xm, fun=float(fun(xm)), success=True)


class optimize:  # namespace mirror
    minimize = staticmethod(minimize)
    minimize_scalar = staticmethod(minimize_scalar)
    OptimizeResult = OptimizeResult
