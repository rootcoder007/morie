"""morie sci core: scipy.optimize / scipy.spatial / scipy.special subsets.

Native replacements for the non-stats scipy surfaces morie uses
(inventory: optimize.minimize 71+, spatial cdist/pdist/squareform ~100,
special.expit 24+).  Pure Python reference implementations; C kernels
in morie_core later.  Equivalence-tested against scipy in
tests/fn/test_sci_core.py.
"""

from __future__ import annotations

import builtins as _bi
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


# ------------------------------------------------------------ special tail

def gamma(x):
    def one(v):
        return _math.gamma(v)
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(one)
    return one(float(x))


def loggamma(x):
    return gammaln(x)


def digamma(x):
    def one(v):
        v = float(v)
        acc = 0.0
        while v < 12.0:
            acc -= 1.0 / v
            v += 1.0
        inv = 1.0 / v
        inv2 = inv * inv
        return acc + _math.log(v) - 0.5 * inv - inv2 * (
            1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (
                1.0 / 252.0 - inv2 / 240.0)))
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(one)
    return one(float(x))


psi = digamma


def betaln(a, b):
    return (_math.lgamma(float(a)) + _math.lgamma(float(b))
            - _math.lgamma(float(a) + float(b)))


def comb(n, k, exact=False):
    n, k = int(n), int(k)
    if k < 0 or k > n:
        return 0
    val = _math.comb(n, k)
    return val if exact else float(val)


def softmax(x):
    v = [float(u) for u in (x.tolist() if hasattr(x, "tolist") else x)]
    m = max(v)
    e = [_math.exp(u - m) for u in v]
    s = _math.fsum(e)
    return _ac.marr([u / s for u in e])


def erfcinv(y):
    # erfcinv(y) = -ndtri(y/2)/sqrt(2); bisect the erfc
    y = float(y)
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def erfinv(y):
    return erfcinv(1.0 - float(y))


def _sc_gammainc_p(a, x):
    """Regularized lower incomplete gamma (series + Lentz CF)."""
    a, x = float(a), float(x)
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * _math.exp(-x + a * _math.log(x)
                                 - _math.lgamma(a))
    # continued fraction for Q, P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    q = h * _math.exp(-x + a * _math.log(x) - _math.lgamma(a))
    return 1.0 - q


def gammainc(a, x):
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(lambda v: _sc_gammainc_p(a, v))
    return _sc_gammainc_p(a, x)


def gammaincc(a, x):
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(lambda v: 1.0 - _sc_gammainc_p(a, v))
    return 1.0 - _sc_gammainc_p(a, x)


def _sc_betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def betainc(a, b, x):
    def one(xx):
        xx = float(xx)
        if xx <= 0.0:
            return 0.0
        if xx >= 1.0:
            return 1.0
        ln_front = (_math.lgamma(a + b) - _math.lgamma(a)
                    - _math.lgamma(b) + a * _math.log(xx)
                    + b * _math.log1p(-xx))
        if xx < (a + 1.0) / (a + b + 2.0):
            return _math.exp(ln_front) * _sc_betacf(a, b, xx) / a
        return 1.0 - _math.exp(ln_front) * _sc_betacf(
            b, a, 1.0 - xx) / b
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(one)
    return one(x)


for _n in ("gamma", "loggamma", "digamma", "psi", "betaln", "comb",
           "softmax", "erfcinv", "erfinv", "gammainc", "gammaincc",
           "betainc"):
    setattr(special, _n, staticmethod(globals()[_n]))


# ------------------------------------------------------------ optimize tail

def brentq(f, a, b, args=(), xtol=2e-12, rtol=8.9e-16, maxiter=100):
    fa, fb = float(f(a, *args)), float(f(b, *args))
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * rtol * abs(b) + 0.5 * xtol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r)
                         - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        b += d if abs(d) > tol1 else _math.copysign(tol1, xm)
        fb = float(f(b, *args))
    return b


def approx_fprime(xk, f, epsilon=1.4901161193847656e-08, *args):
    x = list(_ac.asarray(xk)._flat())
    if isinstance(epsilon, (int, float)):
        eps = [float(epsilon)] * len(x)
    else:
        eps = [float(v) for v in epsilon]
    f0 = float(f(x, *args))
    g = []
    for i in range(len(x)):
        xp = list(x)
        xp[i] += eps[i]
        g.append((float(f(xp, *args)) - f0) / eps[i])
    return _ac.marr(g)


def curve_fit(f, xdata, ydata, p0=None, maxfev=2000):
    """Levenberg-Marquardt least squares with numeric Jacobian."""
    xs = list(_ac.asarray(xdata)._flat()) \
        if not isinstance(xdata, (list, tuple)) else list(xdata)
    ys = [float(v) for v in _ac.asarray(ydata)._flat()]
    import inspect
    if p0 is None:
        nparam = len(inspect.signature(f).parameters) - 1
        p = [1.0] * nparam
    else:
        p = [float(v) for v in p0]
    np_ = len(p)
    lam = 1e-3

    def resid(pv):
        try:
            model = f(_ac.marr([float(u) for u in xs]), *pv)
        except TypeError:
            model = [f(u, *pv) for u in xs]
        mv = [float(v) for v in _ac.asarray(model)._flat()] \
            if hasattr(model, "_flat") or isinstance(model, list) \
            else [float(model)] * len(ys)
        return [ys[i] - mv[i] for i in range(len(ys))]

    r = resid(p)
    ssr = _math.fsum(v * v for v in r)
    nfev = 1
    for _ in range(maxfev):
        # numeric jacobian of residuals
        J = []
        for j in range(np_):
            pj = list(p)
            h = 1e-7 * max(abs(pj[j]), 1.0)
            pj[j] += h
            rj = resid(pj)
            nfev += 1
            J.append([(rj[i] - r[i]) / h for i in range(len(r))])
        # normal equations (J^T J + lam diag) dp = -J^T r
        A = [[_math.fsum(J[a][i] * J[b][i] for i in range(len(r)))
              for b in range(np_)] for a in range(np_)]
        g = [_math.fsum(J[a][i] * r[i] for i in range(len(r)))
             for a in range(np_)]
        improved = False
        for _try in range(30):
            Ad = [[A[i][j] + (lam * A[i][i] if i == j else 0.0)
                   for j in range(np_)] for i in range(np_)]
            try:
                dp = _ac.linalg.solve(_ac.marr(Ad),
                                      _ac.marr([-v for v in g]))
            except Exception:
                lam *= 10.0
                continue
            pn = [p[i] + float(dp[i]) for i in range(np_)]
            rn = resid(pn)
            nfev += 1
            ssn = _math.fsum(v * v for v in rn)
            if ssn < ssr:
                p, r, ssr = pn, rn, ssn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or ssr < 1e-30:
            break
    # covariance = ssr/(m-n) * (J^T J)^-1
    dof = max(len(r) - np_, 1)
    try:
        pcov = _ac.linalg.inv(_ac.marr(A)) * (ssr / dof)
    except Exception:
        pcov = _ac.marr([[float("inf")] * np_ for _ in range(np_)])
    return _ac.marr(p), pcov


def nnls(A, b):
    """Lawson-Hanson non-negative least squares."""
    Am = _ac.atleast_2d(A)
    bv = [float(v) for v in _ac.asarray(b)._flat()]
    m, n = Am.shape
    cols = [[Am.data[i][j] for i in range(m)] for j in range(n)]
    x = [0.0] * n
    P = set()
    for _ in range(3 * n + 10):
        r = [bv[i] - _math.fsum(cols[j][i] * x[j] for j in range(n))
             for i in range(m)]
        w = [_math.fsum(cols[j][i] * r[i] for i in range(m))
             for j in range(n)]
        candidates = [j for j in range(n) if j not in P]
        if not candidates or max(w[j] for j in candidates) <= 1e-12:
            break
        P.add(max(candidates, key=lambda j: w[j]))
        while True:
            Ps = sorted(P)
            AtA = [[_math.fsum(cols[a][i] * cols[b_][i]
                               for i in range(m)) for b_ in Ps]
                   for a in Ps]
            Atb = [_math.fsum(cols[a][i] * bv[i] for i in range(m))
                   for a in Ps]
            z = list(_ac.linalg.solve(_ac.marr(AtA),
                                      _ac.marr(Atb))._flat())
            if all(v > 1e-14 for v in z):
                for k, j in enumerate(Ps):
                    x[j] = z[k]
                for j in range(n):
                    if j not in P:
                        x[j] = 0.0
                break
            alphas = [x[j] / (x[j] - z[k])
                      for k, j in enumerate(Ps) if z[k] <= 1e-14
                      and x[j] != z[k]]
            alpha = min(alphas) if alphas else 0.0
            for k, j in enumerate(Ps):
                x[j] += alpha * (z[k] - x[j])
            P = {j for j in P if x[j] > 1e-14}
            if not P:
                break
    rnorm = _math.sqrt(_math.fsum(
        (bv[i] - _math.fsum(cols[j][i] * x[j] for j in range(n))) ** 2
        for i in range(m)))
    return _ac.marr(x), rnorm


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=None, method=None, **kw):
    """Two-phase simplex; default bounds x >= 0 (scipy convention)."""
    del method, kw
    cv = [float(v) for v in _ac.asarray(c)._flat()]
    n = len(cv)
    rows = []
    rhs = []
    ineq = []
    if A_ub is not None:
        Au = _ac.atleast_2d(A_ub)
        for i, r in enumerate(Au.data):
            rows.append([float(v) for v in r])
            rhs.append(float(_ac.asarray(b_ub)._flat()[i]))
            ineq.append(True)
    if A_eq is not None:
        Ae = _ac.atleast_2d(A_eq)
        for i, r in enumerate(Ae.data):
            rows.append([float(v) for v in r])
            rhs.append(float(_ac.asarray(b_eq)._flat()[i]))
            ineq.append(False)
    if bounds is not None:
        bl = bounds if isinstance(bounds, list) else [bounds] * n
        for j, (lo, hi) in enumerate(bl):
            if lo not in (0, 0.0, None):
                r = [0.0] * n
                r[j] = -1.0
                rows.append(r)
                rhs.append(-float(lo))
                ineq.append(True)
            if hi is not None:
                r = [0.0] * n
                r[j] = 1.0
                rows.append(r)
                rhs.append(float(hi))
                ineq.append(True)
    m = len(rows)
    nslack = sum(1 for q in ineq if q)
    total = n + nslack + m          # slacks + artificials
    T = []
    si = 0
    basis = []
    for i in range(m):
        row = rows[i][:]
        sign = 1.0
        if rhs[i] < 0:
            row = [-v for v in row]
            rhs[i] = -rhs[i]
            sign = -1.0
        srow = [0.0] * nslack
        if ineq[i]:
            srow[si] = sign
            si += 1
        arow = [0.0] * m
        arow[i] = 1.0
        T.append(row + srow + arow + [rhs[i]])
        basis.append(n + nslack + i)

    def pivot(T, basis, obj):
        while True:
            piv_col = min(range(total),
                          key=lambda j: obj[j])
            if obj[piv_col] > -1e-10:
                return True
            ratios = [(T[i][-1] / T[i][piv_col], i)
                      for i in range(m) if T[i][piv_col] > 1e-10]
            if not ratios:
                return False        # unbounded
            _, piv_row = min(ratios)
            pv = T[piv_row][piv_col]
            T[piv_row] = [v / pv for v in T[piv_row]]
            for i in range(m):
                if i != piv_row and abs(T[i][piv_col]) > 1e-12:
                    fac = T[i][piv_col]
                    T[i] = [a - fac * b
                            for a, b in zip(T[i], T[piv_row])]
            fac = obj[piv_col]
            for j in range(total + 1):
                obj[j] -= fac * T[piv_row][j]
            basis[piv_row] = piv_col

    # phase 1: minimize sum of artificials
    obj1 = [0.0] * total + [0.0]
    for j in range(n + nslack, total):
        obj1[j] = 1.0
    for i in range(m):
        for j in range(total + 1):
            obj1[j] -= T[i][j]
    ok = pivot(T, basis, obj1)
    if not ok or obj1[-1] < -1e-7:
        return OptimizeResult(x=None, fun=None, success=False,
                              status=2, message="infeasible")
    # phase 2
    obj2 = [0.0] * (total + 1)
    for j in range(n):
        obj2[j] = cv[j]
    for i in range(m):
        if basis[i] < n:
            fac = obj2[basis[i]]
            if fac != 0.0:
                for j in range(total + 1):
                    obj2[j] -= fac * T[i][j]
    ok = pivot(T, basis, obj2)
    if not ok:
        return OptimizeResult(x=None, fun=None, success=False,
                              status=3, message="unbounded")
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][-1]
    fun = _math.fsum(cv[j] * x[j] for j in range(n))
    return OptimizeResult(x=_ac.marr(x), fun=fun, success=True,
                          status=0, message="optimal")


for _n in ("brentq", "approx_fprime", "curve_fit", "nnls", "linprog"):
    setattr(optimize, _n, staticmethod(globals()[_n]))


# ------------------------------------------------------------ integrate

def _adaptive_simpson(f, a, b, fa, fm, fb, whole, tol, depth):
    m = 0.5 * (a + b)
    lm, rm = 0.5 * (a + m), 0.5 * (m + b)
    flm, frm = f(lm), f(rm)
    left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
    right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
    if depth <= 0 or abs(left + right - whole) < 15.0 * tol:
        return left + right + (left + right - whole) / 15.0
    return (_adaptive_simpson(f, a, m, fa, flm, fm, left,
                              tol / 2.0, depth - 1)
            + _adaptive_simpson(f, m, b, fm, frm, fb, right,
                                tol / 2.0, depth - 1))


def quad(func, a, b, args=(), epsabs=1.49e-08, **kw):
    del kw

    def f(x):
        return float(func(x, *args))
    a, b = float(a), float(b)
    # substitution for infinite limits
    if _math.isinf(a) or _math.isinf(b):
        if _math.isinf(a) and _math.isinf(b):
            def g(t):
                x = t / (1.0 - t * t)
                return f(x) * (1.0 + t * t) / (1.0 - t * t) ** 2
            lo, hi = -1.0 + 1e-10, 1.0 - 1e-10
        elif _math.isinf(b):
            def g(t):
                x = a + t / (1.0 - t)
                return f(x) / (1.0 - t) ** 2
            lo, hi = 0.0, 1.0 - 1e-10
        else:
            def g(t):
                x = b - t / (1.0 - t)
                return f(x) / (1.0 - t) ** 2
            lo, hi = 0.0, 1.0 - 1e-10
        fn2 = g
    else:
        fn2 = f
        lo, hi = a, b
    m = 0.5 * (lo + hi)
    fa_, fm_, fb_ = fn2(lo), fn2(m), fn2(hi)
    whole = (hi - lo) / 6.0 * (fa_ + 4.0 * fm_ + fb_)
    val = _adaptive_simpson(fn2, lo, hi, fa_, fm_, fb_, whole,
                            epsabs, 48)
    if _math.isinf(a) and not _math.isinf(b) and False:
        val = -val
    return val, epsabs


def trapz(y, x=None, dx=1.0):
    return _ac.trapezoid(y, x=x, dx=dx) if hasattr(
        _ac, "trapezoid") else None


def simpson(y, x=None, dx=1.0):
    yv = [float(v) for v in _ac.asarray(y)._flat()]
    n = len(yv)
    xv = [float(v) for v in _ac.asarray(x)._flat()] \
        if x is not None else [i * dx for i in range(n)]
    total = 0.0
    i = 0
    while i + 2 < n:
        h0 = xv[i + 1] - xv[i]
        h1 = xv[i + 2] - xv[i + 1]
        total += (h0 + h1) / 6.0 * (
            (2.0 - h1 / h0) * yv[i]
            + (h0 + h1) ** 2 / (h0 * h1) * yv[i + 1]
            + (2.0 - h0 / h1) * yv[i + 2])
        i += 2
    if n % 2 == 0 and n >= 3:
        # trapezoid on the final interval (scipy: even='avg' differs;
        # scipy>=1.11 default handles last interval via cartwright)
        h = xv[-1] - xv[-2]
        total += 0.5 * h * (yv[-1] + yv[-2])
    return total


def cumulative_trapezoid(y, x=None, dx=1.0, initial=None):
    yv = [float(v) for v in _ac.asarray(y)._flat()]
    n = len(yv)
    xv = [float(v) for v in _ac.asarray(x)._flat()] \
        if x is not None else [i * dx for i in range(n)]
    out = []
    acc = 0.0
    for i in range(1, n):
        acc += 0.5 * (yv[i] + yv[i - 1]) * (xv[i] - xv[i - 1])
        out.append(acc)
    if initial is not None:
        out = [float(initial)] + out
    return _ac.marr(out)


def _rk45_step(f, t, y, h, args):
    # Dormand-Prince coefficients
    k1 = f(t, y, *args)
    k2 = f(t + h / 5.0, [y[i] + h / 5.0 * k1[i]
                         for i in range(len(y))], *args)
    k3 = f(t + 3.0 * h / 10.0,
           [y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i])
            for i in range(len(y))], *args)
    k4 = f(t + 4.0 * h / 5.0,
           [y[i] + h * (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i]
                        + 32.0 / 9.0 * k3[i])
            for i in range(len(y))], *args)
    k5 = f(t + 8.0 * h / 9.0,
           [y[i] + h * (19372.0 / 6561.0 * k1[i]
                        - 25360.0 / 2187.0 * k2[i]
                        + 64448.0 / 6561.0 * k3[i]
                        - 212.0 / 729.0 * k4[i])
            for i in range(len(y))], *args)
    k6 = f(t + h,
           [y[i] + h * (9017.0 / 3168.0 * k1[i]
                        - 355.0 / 33.0 * k2[i]
                        + 46732.0 / 5247.0 * k3[i]
                        + 49.0 / 176.0 * k4[i]
                        - 5103.0 / 18656.0 * k5[i])
            for i in range(len(y))], *args)
    y5 = [y[i] + h * (35.0 / 384.0 * k1[i] + 500.0 / 1113.0 * k3[i]
                      + 125.0 / 192.0 * k4[i]
                      - 2187.0 / 6784.0 * k5[i]
                      + 11.0 / 84.0 * k6[i])
          for i in range(len(y))]
    k7 = f(t + h, y5, *args)
    y4 = [y[i] + h * (5179.0 / 57600.0 * k1[i]
                      + 7571.0 / 16695.0 * k3[i]
                      + 393.0 / 640.0 * k4[i]
                      - 92097.0 / 339200.0 * k5[i]
                      + 187.0 / 2100.0 * k6[i] + 1.0 / 40.0 * k7[i])
          for i in range(len(y))]
    err = _math.sqrt(_math.fsum((y5[i] - y4[i]) ** 2
                                for i in range(len(y))))
    return y5, err


def _integrate_to(f, t0, y0, t1, args, rtol, atol):
    t, y = t0, list(y0)
    h = (t1 - t0) / 100.0 if t1 != t0 else 0.0
    if h == 0.0:
        return y
    while (t < t1) if h > 0 else (t > t1):
        if (h > 0 and t + h > t1) or (h < 0 and t + h < t1):
            h = t1 - t
        yn, err = _rk45_step(f, t, y, h, args)
        scale = atol + rtol * max(max(abs(v) for v in y), 1e-12)
        if err <= scale or abs(h) < 1e-14:
            t += h
            y = yn
            if err > 0:
                h *= min(5.0, max(0.2, 0.9 * (scale / err) ** 0.2))
        else:
            h *= max(0.2, 0.9 * (scale / err) ** 0.2)
    return y


def odeint(func, y0, t, args=(), rtol=1.49e-8, atol=1.49e-8, **kw):
    """scipy.integrate.odeint signature: func(y, t, *args)."""
    del kw
    ts = [float(v) for v in _ac.asarray(t)._flat()]
    y = [float(v) for v in (_ac.asarray(y0)._flat()
                            if not isinstance(y0, (int, float))
                            else [y0])]

    def f(tt, yy, *a):
        out = func(yy, tt, *a)
        if isinstance(out, (int, float)):
            return [float(out)]
        return [float(v) for v in (out._flat()
                                   if hasattr(out, "_flat") else out)]
    rows = [list(y)]
    for k in range(1, len(ts)):
        y = _integrate_to(f, ts[k - 1], y, ts[k], args, rtol, atol)
        rows.append(list(y))
    return _ac.marr(rows)


def solve_ivp(fun, t_span, y0, t_eval=None, args=(), rtol=1e-3,
              atol=1e-6, **kw):
    del kw
    t0, t1 = float(t_span[0]), float(t_span[1])
    ts = ([float(v) for v in _ac.asarray(t_eval)._flat()]
          if t_eval is not None else
          [t0 + (t1 - t0) * k / 100.0 for k in range(101)])
    y = [float(v) for v in _ac.asarray(y0)._flat()]

    def f(tt, yy, *a):
        out = fun(tt, yy, *a)
        return [float(v) for v in (out._flat()
                                   if hasattr(out, "_flat") else out)]
    cols = [list(y)]
    for k in range(1, len(ts)):
        y = _integrate_to(f, ts[k - 1], y, ts[k], args, rtol, atol)
        cols.append(list(y))
    ymat = [[cols[k][i] for k in range(len(ts))]
            for i in range(len(y))]
    return OptimizeResult(t=_ac.marr(ts), y=_ac.marr(ymat),
                          success=True)


class integrate:  # namespace mirror
    quad = staticmethod(quad)
    simpson = staticmethod(simpson)
    trapz = staticmethod(trapz)
    cumulative_trapezoid = staticmethod(cumulative_trapezoid)
    odeint = staticmethod(odeint)
    solve_ivp = staticmethod(solve_ivp)


# ------------------------------------------------------------ linalg extras

def toeplitz(c, r=None):
    cv = [float(v) for v in _ac.asarray(c)._flat()]
    rv = [float(v) for v in _ac.asarray(r)._flat()] if r is not None \
        else list(cv)
    n, m = len(cv), len(rv)
    return _ac.marr([[cv[i - j] if i >= j else rv[j - i]
                      for j in range(m)] for i in range(n)])


def solve_triangular(a, b, lower=False):
    A = _ac.atleast_2d(a)
    bv = [float(v) for v in _ac.asarray(b)._flat()]
    n = A.shape[0]
    x = [0.0] * n
    if lower:
        for i in range(n):
            s = _math.fsum(A.data[i][j] * x[j] for j in range(i))
            x[i] = (bv[i] - s) / A.data[i][i]
    else:
        for i in range(n - 1, -1, -1):
            s = _math.fsum(A.data[i][j] * x[j]
                           for j in range(i + 1, n))
            x[i] = (bv[i] - s) / A.data[i][i]
    return _ac.marr(x)


def lu(a):
    """Doolittle with partial pivoting; returns (P, L, U) like scipy."""
    A = _ac.atleast_2d(a)
    n = A.shape[0]
    U = [row[:] for row in A.data]
    L = [[0.0] * n for _ in range(n)]
    perm = list(range(n))
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(U[i][k]))
        if piv != k:
            U[k], U[piv] = U[piv], U[k]
            L[k], L[piv] = L[piv], L[k]
            perm[k], perm[piv] = perm[piv], perm[k]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            if U[k][k] == 0.0:
                continue
            fac = U[i][k] / U[k][k]
            L[i][k] = fac
            for j in range(k, n):
                U[i][j] -= fac * U[k][j]
    P = [[1.0 if perm[i] == j else 0.0 for j in range(n)]
         for i in range(n)]
    # scipy returns P with A = P @ L @ U
    PT = [[P[j][i] for j in range(n)] for i in range(n)]
    return _ac.marr(PT), _ac.marr(L), _ac.marr(U)


def sqrtm(a):
    """Symmetric square root via eigendecomposition (SPD input)."""
    A = _ac.atleast_2d(a)
    w, V = _ac.linalg.eigh(A)
    n = A.shape[0]
    wl = [max(float(v), 0.0) ** 0.5 for v in w._flat()]
    Vd = V.tolist()
    return _ac.marr([[_math.fsum(Vd[i][k] * wl[k] * Vd[j][k]
                                 for k in range(n))
                      for j in range(n)] for i in range(n)])


def expm(a):
    """Matrix exponential: scaling-and-squaring + Pade(6)."""
    A = _ac.atleast_2d(a)
    n = A.shape[0]
    norm = max(_math.fsum(abs(v) for v in row) for row in A.data)
    s = max(0, int(_math.ceil(_math.log2(max(norm, 1e-300)))) + 1) \
        if norm > 0.5 else 0
    Ad = [[v / (2.0 ** s) for v in row] for row in A.data]

    def mm(X, Y):
        return [[_math.fsum(X[i][k] * Y[k][j] for k in range(n))
                 for j in range(n)] for i in range(n)]

    def madd(X, Y, ca=1.0, cb=1.0):
        return [[ca * X[i][j] + cb * Y[i][j] for j in range(n)]
                for i in range(n)]
    ident = [[1.0 if i == j else 0.0 for j in range(n)]
             for i in range(n)]
    c = [1.0, 0.5, 12 / 120.0, 1 / 120.0 * 10 / 6.0]
    # Pade(6) coefficients: c_k = (6! (12-k)!) / (12! k! (6-k)!)
    coef = []
    for k in range(7):
        coef.append(_math.factorial(6) * _math.factorial(12 - k)
                    / (_math.factorial(12) * _math.factorial(k)
                       * _math.factorial(6 - k)))
    # N = sum c_k A^k ; D = sum c_k (-A)^k
    Ak = ident
    N = [[coef[0] * ident[i][j] for j in range(n)] for i in range(n)]
    D = [[coef[0] * ident[i][j] for j in range(n)] for i in range(n)]
    for k in range(1, 7):
        Ak = mm(Ak, Ad)
        sgn = -1.0 if k % 2 else 1.0
        N = madd(N, Ak, 1.0, coef[k])
        D = madd(D, Ak, 1.0, sgn * coef[k])
    # solve D X = N column-wise
    X = []
    Dm = _ac.marr(D)
    for j in range(n):
        col = _ac.linalg.solve(Dm, _ac.marr([N[i][j]
                                             for i in range(n)]))
        X.append(list(col._flat()))
    R = [[X[j][i] for j in range(n)] for i in range(n)]
    for _ in range(s):
        R = mm(R, R)
    return _ac.marr(R)


class linalg:  # namespace mirror for `from scipy import linalg`
    toeplitz = staticmethod(toeplitz)
    solve_triangular = staticmethod(solve_triangular)
    lu = staticmethod(lu)
    sqrtm = staticmethod(sqrtm)
    expm = staticmethod(expm)

    @staticmethod
    def solve(a, b):
        return _ac.linalg.solve(a, b)

    @staticmethod
    def inv(a):
        return _ac.linalg.inv(a)

    @staticmethod
    def cholesky(a, lower=False):
        L = _ac.linalg.cholesky(a)          # lower by convention
        if lower:
            return L
        Ld = L.tolist()
        n = len(Ld)
        return _ac.marr([[Ld[j][i] for j in range(n)]
                         for i in range(n)])

    @staticmethod
    def eigh(a):
        return _ac.linalg.eigh(a)

    @staticmethod
    def svd(a, **kw):
        return _ac.linalg.svd(a, **kw)

    @staticmethod
    def qr(a, mode="reduced"):
        return _ac.linalg.qr(a, mode=mode)

    @staticmethod
    def lstsq(a, b):
        return _ac.linalg.lstsq(a, b)


# ------------------------------------------------------------ interpolate

class interp1d:
    def __init__(self, x, y, kind="linear", fill_value=None,
                 bounds_error=True):
        self.x = [float(v) for v in _ac.asarray(x)._flat()]
        self.y = [float(v) for v in _ac.asarray(y)._flat()]
        self.kind = kind
        self.fill_value = fill_value
        self.bounds_error = bounds_error
        if kind == "cubic":
            self._spline = CubicSpline(self.x, self.y)

    def _one(self, v):
        v = float(v)
        xs, ys = self.x, self.y
        if v < xs[0] or v > xs[-1]:
            if self.fill_value == "extrapolate":
                pass
            elif self.bounds_error:
                raise ValueError("x out of interpolation range")
            elif self.fill_value is not None:
                return float(self.fill_value)
            else:
                return float("nan")
        if self.kind == "cubic":
            return self._spline(v)
        import bisect
        i = bisect.bisect_right(xs, v) - 1
        i = _bi.max(0, _bi.min(i, len(xs) - 2))
        t = (v - xs[i]) / (xs[i + 1] - xs[i])
        if self.kind in ("linear", None):
            return ys[i] + t * (ys[i + 1] - ys[i])
        if self.kind in ("nearest",):
            return ys[i] if t < 0.5 else ys[i + 1]
        if self.kind in ("previous", "zero"):
            return ys[i]
        if self.kind == "next":
            return ys[i + 1]
        raise ValueError("unsupported kind %r" % self.kind)

    def __call__(self, xnew):
        if isinstance(xnew, (int, float)):
            return self._one(xnew)
        return _ac.marr([self._one(v)
                         for v in _ac.asarray(xnew)._flat()])


class CubicSpline:
    """Not-a-knot cubic spline (scipy default bc_type)."""

    def __init__(self, x, y, bc_type="not-a-knot"):
        xs = [float(v) for v in _ac.asarray(x)._flat()]
        ys = [float(v) for v in _ac.asarray(y)._flat()]
        n = len(xs)
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]
        # solve for second-derivative-like coefficients via the
        # standard tridiagonal system on spline slopes (m = y')
        A = [[0.0] * n for _ in range(n)]
        rhs = [0.0] * n
        for i in range(1, n - 1):
            A[i][i - 1] = h[i]
            A[i][i] = 2.0 * (h[i - 1] + h[i])
            A[i][i + 1] = h[i - 1]
            rhs[i] = 3.0 * (h[i] * (ys[i] - ys[i - 1]) / h[i - 1]
                            + h[i - 1] * (ys[i + 1] - ys[i]) / h[i])
        if bc_type == "natural":
            A[0][0] = 2.0
            A[0][1] = 1.0
            rhs[0] = 3.0 * (ys[1] - ys[0]) / h[0]
            A[n - 1][n - 2] = 1.0
            A[n - 1][n - 1] = 2.0
            rhs[n - 1] = 3.0 * (ys[n - 1] - ys[n - 2]) / h[n - 2]
        else:  # not-a-knot
            A[0][0] = h[1]
            A[0][1] = h[0] + h[1]
            rhs[0] = ((h[0] + 2.0 * (h[0] + h[1])) * h[1]
                      * (ys[1] - ys[0]) / h[0]
                      + h[0] * h[0] * (ys[2] - ys[1]) / h[1]) \
                / (h[0] + h[1])
            A[n - 1][n - 2] = h[n - 2] + h[n - 3]
            A[n - 1][n - 1] = h[n - 3]
            rhs[n - 1] = (h[n - 2] * h[n - 2]
                          * (ys[n - 2] - ys[n - 3]) / h[n - 3]
                          + (2.0 * (h[n - 2] + h[n - 3]) + h[n - 2])
                          * h[n - 3]
                          * (ys[n - 1] - ys[n - 2]) / h[n - 2]) \
                / (h[n - 2] + h[n - 3])
        m = list(_ac.linalg.solve(_ac.marr(A), _ac.marr(rhs))._flat())
        self.x = xs
        self.y = ys
        self._m = m
        self._h = h

    def __call__(self, xnew):
        def one(v):
            v = float(v)
            xs, ys, m, h = self.x, self.y, self._m, self._h
            import bisect
            i = bisect.bisect_right(xs, v) - 1
            i = _bi.max(0, _bi.min(i, len(xs) - 2))
            t = (v - xs[i]) / h[i]
            h00 = 2 * t ** 3 - 3 * t ** 2 + 1
            h10 = t ** 3 - 2 * t ** 2 + t
            h01 = -2 * t ** 3 + 3 * t ** 2
            h11 = t ** 3 - t ** 2
            return (h00 * ys[i] + h10 * h[i] * m[i]
                    + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])
        if isinstance(xnew, (int, float)):
            return one(xnew)
        return _ac.marr([one(v) for v in _ac.asarray(xnew)._flat()])


class interpolate:  # namespace mirror
    interp1d = interp1d
    CubicSpline = CubicSpline
