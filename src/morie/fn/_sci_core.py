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
    fvals = [float(fun(_ac.marr(p), *args)) for p in simplex]
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
        fr = float(fun(_ac.marr(xr), *args))
        nfev += 1
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = [centroid[d] + gamma_ * (xr[d] - centroid[d])
                  for d in range(n)]
            fe = float(fun(_ac.marr(xe), *args))
            nfev += 1
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = [centroid[d] + rho_ * (simplex[-1][d] - centroid[d])
              for d in range(n)]
        fc = float(fun(_ac.marr(xc), *args))
        nfev += 1
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        for k in range(1, n + 1):
            simplex[k] = [simplex[0][d]
                          + sigma * (simplex[k][d] - simplex[0][d])
                          for d in range(n)]
            fvals[k] = float(fun(_ac.marr(simplex[k]), *args))
            nfev += n
    order = sorted(range(n + 1), key=lambda k: fvals[k])
    best = simplex[order[0]]
    return OptimizeResult(x=_ac.marr(best), fun=fvals[order[0]],
                          nit=it + 1, nfev=nfev,
                          success=True, message="nelder-mead converged")


def _num_grad(fun, x, args, eps=1e-7):
    f0 = float(fun(_ac.marr(list(x)), *args))
    g = []
    for i in range(len(x)):
        xp = list(x)
        h = eps * max(abs(xp[i]), 1.0)
        xp[i] += h
        g.append((float(fun(_ac.marr(xp), *args)) - f0) / h)
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
            fn_ = float(fun(_ac.marr(xn), *args))
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


def minimize(fun, x0, args=(), method=None, bounds=None, **kw):
    """Minimise fun.

    bounds used to fall into **kw and be discarded, so every bounded
    call here optimised UNCONSTRAINED: the GARCH/EGARCH/ARCH-t
    likelihoods could reach omega < 0 or alpha + beta >= 1 -- a negative
    conditional variance rather than a slightly wrong fit -- and the IRT
    graded-response fit could return a negative discrimination.  Bounds
    are enforced by projection now: the objective is evaluated at the
    clipped point and the returned x is clipped, so no caller can be
    handed an infeasible parameter.

    method still maps several names onto two engines, but no longer
    sends Powell to a quasi-Newton method: Powell is chosen when the
    objective is non-smooth, where a gradient method is the wrong tool,
    so it routes to Nelder-Mead.
    """
    x0 = list(_ac.asarray(x0)._flat())
    if method is None:
        method = "BFGS"
    m = method.lower().replace("-", "")
    opts = kw.get("options", {}) or {}

    lo = hi = None
    if bounds is not None:
        bl = list(bounds)
        if len(bl) != len(x0):
            raise ValueError("bounds has %d entries but x0 has %d"
                             % (len(bl), len(x0)))
        lo = [(-_math.inf if b is None or b[0] is None else float(b[0]))
              for b in bl]
        hi = [(_math.inf if b is None or b[1] is None else float(b[1]))
              for b in bl]
        for a, b in zip(lo, hi):
            if a > b:
                raise ValueError("lower bound %g exceeds upper bound %g"
                                 % (a, b))

        def clip(v):
            return [min(max(vi, a), b) for vi, a, b in zip(v, lo, hi)]

        x0 = clip(x0)
        _raw = fun

        def fun(z, *a):          # noqa: F811 -- deliberate shadow
            return _raw(clip(list(z)), *a)
    else:
        def clip(v):
            return list(v)

    if m in ("neldermead", "powell"):
        res = _nelder_mead(fun, x0, args=args,
                           maxiter=opts.get("maxiter"),
                           xatol=opts.get("xatol", 1e-8),
                           fatol=opts.get("fatol", 1e-8))
    elif m in ("bfgs", "lbfgsb", "cg"):
        res = _bfgs(fun, x0, args=args, maxiter=opts.get("maxiter"),
                    gtol=opts.get("gtol", 1e-6))
    else:
        raise ValueError("unsupported method %r" % method)

    if bounds is not None:
        xc = clip(list(_ac.asarray(res.x)._flat()))
        res = OptimizeResult(x=_ac.asarray(xc), fun=float(res.fun),
                             success=getattr(res, "success", True),
                             nit=getattr(res, "nit", 0))
    return res


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

def _adaptive_simpson(f, a, b, fa, fm, fb, whole, tol, depth,
                      budget=None):
    """budget: single-element list of remaining f-evals; when it hits
    zero the current Richardson estimate is returned (rough
    integrands would otherwise recurse for hours)."""
    if budget is None:
        budget = [200000]
    m = 0.5 * (a + b)
    lm, rm = 0.5 * (a + m), 0.5 * (m + b)
    flm, frm = f(lm), f(rm)
    budget[0] -= 2
    left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
    right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
    if depth <= 0 or budget[0] <= 0             or abs(left + right - whole) < 15.0 * tol:
        return left + right + (left + right - whole) / 15.0
    return (_adaptive_simpson(f, a, m, fa, flm, fm, left,
                              tol / 2.0, depth - 1, budget)
            + _adaptive_simpson(f, m, b, fm, frm, fb, right,
                                tol / 2.0, depth - 1, budget))


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
                            epsabs, 24)
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


# ------------------------------------------------------------ Bessel K

def kv(v, x):
    """Modified Bessel K_v via integral representation (adaptive quad).

    K_v(x) = int_0^inf exp(-x cosh t) cosh(v t) dt, x > 0.
    """
    def one(xx):
        xx = float(xx)
        if xx <= 0:
            return float("inf")

        def f(t):
            e = -xx * _math.cosh(t)
            if e < -700.0:
                return 0.0
            return _math.exp(e) * _math.cosh(float(v) * t)
        # integrand decays like exp(-x cosh t); upper cut where dead
        hi = 1.0
        while xx * _math.cosh(hi) < 720.0 and hi < 60.0:
            hi += 1.0
        val, _err = quad(f, 0.0, hi, epsabs=1e-12)
        return val
    if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        return _ac.asarray(x)._map(one)
    return one(x)


special.kv = staticmethod(kv)
LinAlgError = _ac.linalg.LinAlgError if hasattr(
    _ac.linalg, "LinAlgError") else ValueError
cholesky = linalg.cholesky
solve = linalg.solve


# ------------------------------------------------------------ fft (dct)

def dct(x, type=2, norm=None):
    """DCT-II (default) via FFT of the even extension."""
    xs = [float(v) for v in _ac.asarray(x)._flat()]
    n = len(xs)
    if type != 2:
        raise NotImplementedError("only DCT-II implemented")
    ext = xs + xs[::-1]
    F = _ac.fft.fft(ext).tolist()
    out = []
    for k in range(n):
        w = complex(_math.cos(-_math.pi * k / (2.0 * n)),
                    _math.sin(-_math.pi * k / (2.0 * n)))
        out.append((w * F[k]).real)
    if norm == "ortho":
        out[0] *= _math.sqrt(1.0 / (4.0 * n))
        for k in range(1, n):
            out[k] *= _math.sqrt(1.0 / (2.0 * n))
    return _ac.marr(out)


def idct(x, type=2, norm=None):
    """Inverse of DCT-II (= DCT-III up to scaling)."""
    xs = [float(v) for v in _ac.asarray(x)._flat()]
    n = len(xs)
    if type != 2:
        raise NotImplementedError("only DCT-II inverse implemented")
    if norm == "ortho":
        xs = [xs[0] / _math.sqrt(1.0 / (4.0 * n))] \
            + [v / _math.sqrt(1.0 / (2.0 * n)) for v in xs[1:]]
    out = []
    for i in range(n):
        acc = xs[0] / 2.0
        for k in range(1, n):
            acc += xs[k] * _math.cos(_math.pi * k * (2 * i + 1)
                                     / (2.0 * n))
        out.append(acc * 2.0 / (2.0 * n))
    return _ac.marr(out)


class fft:  # namespace mirror for `from scipy import fft`
    dct = staticmethod(dct)
    idct = staticmethod(idct)

    @staticmethod
    def fft(x, n=None):
        return _ac.fft.fft(x, n)

    @staticmethod
    def ifft(x, n=None):
        return _ac.fft.ifft(x, n)

    @staticmethod
    def rfft(x, n=None):
        return _ac.fft.rfft(x, n)

    @staticmethod
    def irfft(x, n=None):
        return _ac.fft.irfft(x, n)

    @staticmethod
    def fftfreq(n, d=1.0):
        return _ac.fft.fftfreq(n, d)

    @staticmethod
    def rfftfreq(n, d=1.0):
        return _ac.fft.rfftfreq(n, d)


# ------------------------------------------------------------ cluster

def kmeans2(data, k, iter=10, seed=1, minit="points"):
    X = _ac.atleast_2d(data)
    n, d = X.shape
    rng = _ac.random.default_rng(seed)
    if minit == "points" or True:
        idx = []
        while len(idx) < int(k):
            j = int(rng.integers(0, n))
            if j not in idx:
                idx.append(j)
        cents = [list(X.data[j]) for j in idx]
    labels = [0] * n
    for _ in range(int(iter)):
        for i in range(n):
            best, bj = None, 0
            for j in range(int(k)):
                dist = _math.fsum((X.data[i][t] - cents[j][t]) ** 2
                                  for t in range(d))
                if best is None or dist < best:
                    best, bj = dist, j
            labels[i] = bj
        for j in range(int(k)):
            members = [i for i in range(n) if labels[i] == j]
            if members:
                cents[j] = [
                    _math.fsum(X.data[i][t] for i in members)
                    / len(members) for t in range(d)]
    return _ac.marr(cents), _ac.marr([float(v) for v in labels])


def linkage(y, method="single"):
    """Agglomerative clustering (Lance-Williams); y condensed or (n,d)."""
    a = _ac.asarray(y)
    if len(a.shape) == 2:
        D = {}
        n = a.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[(i, j)] = _math.sqrt(_math.fsum(
                    (a.data[i][t] - a.data[j][t]) ** 2
                    for t in range(a.shape[1])))
    else:
        cond = [float(v) for v in a._flat()]
        m = len(cond)
        n = int(round((1 + _math.sqrt(1 + 8 * m)) / 2))
        D = {}
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[(i, j)] = cond[idx]
                idx += 1

    def dget(i, j):
        return D[(i, j) if i < j else (j, i)]

    active = {i: 1 for i in range(n)}   # cluster id -> size
    Z = []
    next_id = n
    for _step in range(n - 1):
        ids = sorted(active)
        best = None
        for ii in range(len(ids) - 1):
            for jj in range(ii + 1, len(ids)):
                dv = dget(ids[ii], ids[jj])
                if best is None or dv < best[0]:
                    best = (dv, ids[ii], ids[jj])
        dv, ci, cj = best
        si, sj = active[ci], active[cj]
        Z.append([float(min(ci, cj)), float(max(ci, cj)), dv,
                  float(si + sj)])
        # Lance-Williams update
        for ck in ids:
            if ck in (ci, cj):
                continue
            dik = dget(ci, ck)
            djk = dget(cj, ck)
            sk = active[ck]
            if method == "single":
                dnew = min(dik, djk)
            elif method == "complete":
                dnew = max(dik, djk)
            elif method == "average":
                dnew = (si * dik + sj * djk) / (si + sj)
            elif method == "ward":
                tot = si + sj + sk
                dnew = _math.sqrt(
                    ((si + sk) * dik * dik + (sj + sk) * djk * djk
                     - sk * dv * dv) / tot)
            else:
                raise ValueError("unsupported method %r" % method)
            D[(min(ck, next_id), max(ck, next_id))] = dnew
        del active[ci], active[cj]
        active[next_id] = si + sj
        next_id += 1
    return _ac.marr(Z)


def fcluster(Z, t, criterion="distance"):
    Zd = [list(map(float, r)) for r in _ac.atleast_2d(Z).data]
    n = len(Zd) + 1
    if criterion == "maxclust":
        # find smallest distance threshold giving <= t clusters
        heights = sorted(r[2] for r in Zd)
        for h in heights:
            labels = fcluster(Zd, h + 1e-12, "distance")
            if len(set(labels._flat())) <= int(t):
                return labels
        return fcluster(Zd, heights[-1] + 1.0, "distance")
    parent = list(range(2 * n - 1))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for k, (a, b, dist, _size) in enumerate(Zd):
        if dist <= float(t):
            parent[find(int(a))] = n + k
            parent[find(int(b))] = n + k
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots) + 1
        labels.append(float(roots[r]))
    return _ac.marr(labels)


def cophenet(Z, Y=None):
    Zd = [list(map(float, r)) for r in _ac.atleast_2d(Z).data]
    n = len(Zd) + 1
    members = {i: [i] for i in range(n)}
    coph = [[0.0] * n for _ in range(n)]
    for k, (a, b, dist, _s) in enumerate(Zd):
        ma, mb = members[int(a)], members[int(b)]
        for i in ma:
            for j in mb:
                coph[i][j] = coph[j][i] = dist
        members[n + k] = ma + mb
    cond = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            cond.append(coph[i][j])
    if Y is None:
        return _ac.marr(cond)
    yv = [float(v) for v in _ac.asarray(Y)._flat()]
    my, mc = _math.fsum(yv) / len(yv), _math.fsum(cond) / len(cond)
    num = _math.fsum((yv[i] - my) * (cond[i] - mc)
                     for i in range(len(yv)))
    den = _math.sqrt(_math.fsum((v - my) ** 2 for v in yv)
                     * _math.fsum((v - mc) ** 2 for v in cond))
    return num / den, _ac.marr(cond)


class _ClusterHierarchy:
    linkage = staticmethod(linkage)
    fcluster = staticmethod(fcluster)
    cophenet = staticmethod(cophenet)


class _ClusterVQ:
    kmeans2 = staticmethod(kmeans2)


class cluster:  # namespace mirror
    hierarchy = _ClusterHierarchy()
    vq = _ClusterVQ()


# ------------------------------------------------------------ ndimage

def uniform_filter1d(x, size, mode="reflect"):
    xs = [float(v) for v in _ac.asarray(x)._flat()]
    n = len(xs)
    half_lo = size // 2
    out = []
    for i in range(n):
        acc = 0.0
        for o in range(-half_lo, size - half_lo):
            idx = i + o
            if idx < 0:
                idx = -idx - 1 if mode == "reflect" else 0
            elif idx >= n:
                idx = 2 * n - idx - 1 if mode == "reflect" else n - 1
            acc += xs[_bi.max(0, _bi.min(idx, n - 1))]
        out.append(acc / size)
    return _ac.marr(out)


def gaussian_filter1d(x, sigma, truncate=4.0, mode="reflect"):
    xs = [float(v) for v in _ac.asarray(x)._flat()]
    n = len(xs)
    r = int(truncate * float(sigma) + 0.5)
    w = [_math.exp(-0.5 * (o / float(sigma)) ** 2)
         for o in range(-r, r + 1)]
    s = _math.fsum(w)
    w = [v / s for v in w]
    out = []
    for i in range(n):
        acc = 0.0
        for k, o in enumerate(range(-r, r + 1)):
            idx = i + o
            if idx < 0:
                idx = -idx - 1
            elif idx >= n:
                idx = 2 * n - idx - 1
            acc += w[k] * xs[_bi.max(0, _bi.min(idx, n - 1))]
        out.append(acc)
    return _ac.marr(out)


def gaussian_filter(x, sigma, truncate=4.0, mode="reflect"):
    a = _ac.asarray(x)
    if len(a.shape) == 1:
        return gaussian_filter1d(a, sigma, truncate, mode)
    # separable: rows then columns
    rows = [gaussian_filter1d(r, sigma, truncate, mode)._flat()
            for r in a.data]
    nr, nc = len(rows), len(rows[0])
    cols = []
    for j in range(nc):
        col = gaussian_filter1d([rows[i][j] for i in range(nr)],
                                sigma, truncate, mode)
        cols.append(list(col._flat()))
    return _ac.marr([[cols[j][i] for j in range(nc)]
                     for i in range(nr)])


def median_filter(x, size=3, mode="reflect"):
    a = _ac.asarray(x)
    if len(a.shape) == 1:
        xs = list(a._flat())
        n = len(xs)
        half_lo = size // 2
        out = []
        for i in range(n):
            win = []
            for o in range(-half_lo, size - half_lo):
                idx = i + o
                if idx < 0:
                    idx = -idx - 1
                elif idx >= n:
                    idx = 2 * n - idx - 1
                win.append(xs[_bi.max(0, _bi.min(idx, n - 1))])
            win.sort()
            out.append(win[len(win) // 2])
        return _ac.marr(out)
    raise NotImplementedError("median_filter: 1-D only for now")


def _nd_convolve(x, weights, mode="reflect"):
    xs = [float(v) for v in _ac.asarray(x)._flat()]
    wv = [float(v) for v in _ac.asarray(weights)._flat()]
    n, m = len(xs), len(wv)
    # ndimage convolve centers the (flipped) kernel
    origin = m // 2
    out = []
    for i in range(n):
        acc = 0.0
        for k in range(m):
            idx = i + origin - k
            if idx < 0:
                idx = -idx - 1
            elif idx >= n:
                idx = 2 * n - idx - 1
            acc += wv[k] * xs[_bi.max(0, _bi.min(idx, n - 1))]
        out.append(acc)
    return _ac.marr(out)


class ndimage:  # namespace mirror
    uniform_filter1d = staticmethod(uniform_filter1d)
    gaussian_filter1d = staticmethod(gaussian_filter1d)
    gaussian_filter = staticmethod(gaussian_filter)
    median_filter = staticmethod(median_filter)
    convolve = staticmethod(_nd_convolve)


# ------------------------------------------------------------ optimize: root etc.

class LinearConstraint:
    def __init__(self, A, lb=-_math.inf, ub=_math.inf):
        self.A = _ac.atleast_2d(A)
        self.lb = lb
        self.ub = ub


def least_squares(fun, x0, args=(), **kw):
    del kw
    x0v = list(_ac.asarray(x0)._flat())

    def sse(p, *a):
        r = fun(p, *a)
        rv = [float(v) for v in (r._flat() if hasattr(r, "_flat")
                                 else r)]
        return _math.fsum(v * v for v in rv)
    res = minimize(sse, x0v, args=args, method="BFGS")
    r = fun(list(res.x._flat()), *args)
    rv = [float(v) for v in (r._flat() if hasattr(r, "_flat") else r)]
    return OptimizeResult(x=res.x, cost=0.5 * _math.fsum(
        v * v for v in rv), fun=_ac.marr(rv), success=res.success,
        nfev=res.nfev)


def root(fun, x0, args=(), method=None, tol=None, **kw):
    """Multidimensional root via damped Newton with numeric Jacobian."""
    del method, kw
    x = list(_ac.asarray(x0)._flat()) \
        if not isinstance(x0, (int, float)) else [float(x0)]
    n = len(x)

    def fv(p):
        out = fun(p if n > 1 else (p if isinstance(x0, (list, tuple))
                                   or hasattr(x0, "tolist")
                                   else p[0]), *args)
        if isinstance(out, (int, float)):
            return [float(out)]
        return [float(v) for v in (out._flat()
                                   if hasattr(out, "_flat") else out)]
    ftol = tol or 1e-10
    F = fv(x)
    for _ in range(200):
        nrm = max(abs(v) for v in F)
        if nrm < ftol:
            break
        J = []
        for j in range(n):
            xp = list(x)
            h = 1e-7 * _bi.max(abs(xp[j]), 1.0)
            xp[j] += h
            Fp = fv(xp)
            J.append([(Fp[i] - F[i]) / h for i in range(n)])
        Jm = _ac.marr([[J[j][i] for j in range(n)]
                       for i in range(n)])
        try:
            dx = _ac.linalg.solve(Jm, _ac.marr([-v for v in F]))
        except Exception:
            break
        step = 1.0
        for _ls in range(40):
            xn = [x[i] + step * float(dx[i]) for i in range(n)]
            Fn = fv(xn)
            if max(abs(v) for v in Fn) < nrm:
                x, F = xn, Fn
                break
            step *= 0.5
        else:
            break
    return OptimizeResult(x=_ac.marr(x), fun=_ac.marr(F),
                          success=max(abs(v) for v in F) < 1e-6)


def differential_evolution(func, bounds, args=(), maxiter=200,
                           popsize=15, seed=1, tol=1e-8, **kw):
    del kw
    rng = _ac.random.default_rng(seed)
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    d = len(bounds)
    np_ = _bi.max(popsize * d, 8)
    pop = [[lo[j] + (hi[j] - lo[j]) * rng.uniform()
            for j in range(d)] for _ in range(np_)]
    fit = [float(func(p, *args)) for p in pop]
    for _gen in range(maxiter):
        for i in range(np_):
            idxs = [k for k in range(np_) if k != i]
            a = pop[int(rng.integers(0, len(idxs)))]
            b = pop[int(rng.integers(0, len(idxs)))]
            c = pop[int(rng.integers(0, len(idxs)))]
            jrand = int(rng.integers(0, d))
            trial = []
            for j in range(d):
                if rng.uniform() < 0.7 or j == jrand:
                    v = a[j] + 0.8 * (b[j] - c[j])
                else:
                    v = pop[i][j]
                trial.append(_bi.min(hi[j], _bi.max(lo[j], v)))
            ft = float(func(trial, *args))
            if ft < fit[i]:
                pop[i], fit[i] = trial, ft
        best = min(fit)
        worst = max(fit)
        if worst - best < tol * (abs(best) + 1e-12):
            break
    bi_ = min(range(np_), key=lambda k: fit[k])
    return OptimizeResult(x=_ac.marr(pop[bi_]), fun=fit[bi_],
                          success=True)


for _n in ("root", "least_squares", "differential_evolution"):
    setattr(optimize, _n, staticmethod(globals()[_n]))
optimize.LinearConstraint = LinearConstraint


# ------------------------------------------------------------ misc tail

def logsumexp(x, axis=None):
    if axis is None:
        v = [float(u) for u in _ac.asarray(x)._flat()]
        m = max(v)
        return m + _math.log(_math.fsum(_math.exp(u - m) for u in v))
    a = _ac.atleast_2d(x)
    if axis == 1:
        return _ac.marr([logsumexp(row) for row in a.data])
    return _ac.marr([logsumexp([a.data[i][j]
                                for i in range(a.shape[0])])
                     for j in range(a.shape[1])])


def beta(a, b):
    return _math.exp(betaln(a, b))


class _Poly1d:
    def __init__(self, coeffs):
        self.coeffs = list(coeffs)

    def __call__(self, x):
        def one(v):
            acc = 0.0
            for c in self.coeffs:
                acc = acc * v + c
            return acc
        if isinstance(x, (int, float)):
            return one(float(x))
        return _ac.asarray(x)._map(one)


def hermite(n):
    """Physicists' Hermite polynomial H_n as a callable."""
    # recurrence H_{k+1} = 2x H_k - 2k H_{k-1}
    h0 = [1.0]
    if n == 0:
        return _Poly1d(h0)
    h1 = [2.0, 0.0]
    for k in range(1, n):
        nxt = [2.0 * c for c in h1] + [0.0]
        for i, c in enumerate(h0):
            nxt[len(nxt) - len(h0) + i] -= 2.0 * k * c
        h0, h1 = h1, nxt
    return _Poly1d(h1)


special.logsumexp = staticmethod(logsumexp)
special.beta = staticmethod(beta)
special.hermite = staticmethod(hermite)


def fsolve(func, x0, args=(), **kw):
    del kw
    res = root(func, x0, args=args)
    return res.x


optimize.fsolve = staticmethod(fsolve)


def eigvals(a):
    """General (nonsymmetric) eigenvalues via Faddeev-LeVerrier
    characteristic polynomial + Durand-Kerner roots.

    ponytail: fine for the small (n <= ~10) stability matrices morie
    uses; a Francis-QR implementation if larger cases ever appear.
    """
    A = _ac.atleast_2d(a)
    n = A.shape[0]
    Ad = [list(map(float, r)) for r in A.data]

    def mm(X, Y):
        return [[_math.fsum(X[i][k] * Y[k][j] for k in range(n))
                 for j in range(n)] for i in range(n)]
    ident = [[1.0 if i == j else 0.0 for j in range(n)]
             for i in range(n)]
    coeffs = [1.0]
    M = [row[:] for row in ident]
    for k in range(1, n + 1):
        M = mm(Ad, M)
        c = -_math.fsum(M[i][i] for i in range(n)) / k
        coeffs.append(c)
        for i in range(n):
            M[i][i] += c
    # Durand-Kerner
    rs = [complex(0.4, 0.9) ** k for k in range(n)]
    for _ in range(500):
        new = []
        for i in range(n):
            num = complex(1.0)
            for j in range(n):
                if j != i:
                    num *= (rs[i] - rs[j])
            pv = complex(0.0)
            for cf in coeffs:
                pv = pv * rs[i] + cf
            new.append(rs[i] - pv / num if num != 0 else rs[i])
        if max(abs(x - y) for x, y in zip(new, rs)) < 1e-13:
            rs = new
            break
        rs = new
    from . import _array_core as _ac2
    return _ac2.carr(rs)


def solve_continuous_lyapunov(a, q):
    """Solve A X + X A^T + Q = 0 via the Kronecker linear system."""
    A = _ac.atleast_2d(a)
    Q = _ac.atleast_2d(q)
    n = A.shape[0]
    # scipy convention: A X + X A^H = Q  -> solve for X
    big = [[0.0] * (n * n) for _ in range(n * n)]
    rhs = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            r = i * n + j
            rhs[r] = float(Q.data[i][j])
            for k in range(n):
                big[r][k * n + j] += float(A.data[i][k])
                big[r][i * n + k] += float(A.data[j][k])
    x = _ac.linalg.solve(_ac.marr(big), _ac.marr(rhs))
    xv = list(x._flat())
    return _ac.marr([[xv[i * n + j] for j in range(n)]
                     for i in range(n)])


linalg.eigvals = staticmethod(eigvals)
linalg.eigvalsh = staticmethod(
    lambda a: _ac.linalg.eigvalsh(a))
linalg.solve_continuous_lyapunov = staticmethod(
    solve_continuous_lyapunov)
linalg.LinAlgError = LinAlgError


# ------------------------------------------------------------ splines

class BSpline:
    """B-spline evaluation via Cox-de Boor recursion."""

    def __init__(self, t, c, k):
        self.t = [float(v) for v in _ac.asarray(t)._flat()]
        self.c = [float(v) for v in _ac.asarray(c)._flat()]
        self.k = int(k)

    def _basis(self, i, k, x):
        t = self.t
        if k == 0:
            # right-closed at the last interval
            if t[i] <= x < t[i + 1]:
                return 1.0
            if x == t[-1] and t[i] < t[i + 1] <= t[-1] \
                    and t[i + 1] == t[-1]:
                return 1.0
            return 0.0
        out = 0.0
        d1 = t[i + k] - t[i]
        if d1 > 0:
            out += (x - t[i]) / d1 * self._basis(i, k - 1, x)
        d2 = t[i + k + 1] - t[i + 1]
        if d2 > 0:
            out += (t[i + k + 1] - x) / d2 * self._basis(i + 1,
                                                         k - 1, x)
        return out

    def __call__(self, x):
        def one(v):
            v = float(v)
            return _math.fsum(self.c[i] * self._basis(i, self.k, v)
                              for i in range(len(self.c)))
        if isinstance(x, (int, float)):
            return one(x)
        return _ac.marr([one(v) for v in _ac.asarray(x)._flat()])


class UnivariateSpline:
    """Least-squares cubic B-spline with interior knots at quantiles.

    s=0 gives interpolation via CubicSpline; s>0 uses a smoothing fit
    with fewer knots (scipy's exact GCV knot placement differs, but the
    fitted curve agrees closely on smooth data).
    """

    def __init__(self, x, y, s=None, k=3):
        xs = [float(v) for v in _ac.asarray(x)._flat()]
        ys = [float(v) for v in _ac.asarray(y)._flat()]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        self._k = int(k)
        n = len(xs)
        if s in (None, 0, 0.0) or n <= self._k + 1:
            self._cs = CubicSpline(xs, ys)
            self._ls = None
            return
        self._cs = None
        # number of interior knots shrinks as s grows
        nint = _bi.max(1, _bi.min(n - self._k - 1,
                                  int(n / (1.0 + s))))
        qs = [xs[int((i + 1) * (n - 1) / (nint + 1))]
              for i in range(nint)]
        t = [xs[0]] * (self._k + 1) + qs + [xs[-1]] * (self._k + 1)
        nb = len(t) - self._k - 1
        sp = BSpline(t, [0.0] * nb, self._k)
        B = [[sp._basis(j, self._k, v) for j in range(nb)]
             for v in xs]
        BtB = [[_math.fsum(B[r][i] * B[r][j] for r in range(n))
                for j in range(nb)] for i in range(nb)]
        Bty = [_math.fsum(B[r][i] * ys[r] for r in range(n))
               for i in range(nb)]
        for i in range(nb):
            BtB[i][i] += 1e-10
        coef = _ac.linalg.solve(_ac.marr(BtB), _ac.marr(Bty))
        self._ls = BSpline(t, list(coef._flat()), self._k)

    def __call__(self, x):
        f = self._cs if self._cs is not None else self._ls
        return f(x)


interpolate.BSpline = BSpline
interpolate.UnivariateSpline = UnivariateSpline


# ------------------------------------------------------------ sparse

class csc_matrix:
    """Dense-backed sparse stand-in (morie matrices are small)."""

    def __init__(self, arg, shape=None):
        if isinstance(arg, tuple) and shape is None \
                and len(arg) == 2 and all(isinstance(v, int)
                                          for v in arg):
            self._m = [[0.0] * arg[1] for _ in range(arg[0])]
        elif isinstance(arg, tuple) and len(arg) == 3:
            data, indices, indptr = arg
            raise NotImplementedError("csc triplet init unused")
        else:
            A = _ac.atleast_2d(arg)
            self._m = [list(map(float, r)) for r in A.data]
        self.shape = (len(self._m), len(self._m[0]))

    def toarray(self):
        return _ac.marr([r[:] for r in self._m])

    todense = toarray

    def __matmul__(self, other):
        return self.toarray() @ _ac.asarray(other)

    @property
    def T(self):
        m, n = self.shape
        return csc_matrix([[self._m[i][j] for i in range(m)]
                           for j in range(n)])


csr_matrix = csc_matrix


def spsolve(a, b):
    A = a.toarray() if hasattr(a, "toarray") else _ac.atleast_2d(a)
    return _ac.linalg.solve(A, _ac.asarray(b))


def eigsh(a, k=6, which="LM"):
    A = a.toarray() if hasattr(a, "toarray") else _ac.atleast_2d(a)
    w, V = _ac.linalg.eigh(A)
    wl = list(w._flat())
    n = len(wl)
    if which == "LM":
        order = sorted(range(n), key=lambda i: -abs(wl[i]))[:k]
    else:                       # "SM" / "SA"
        order = sorted(range(n), key=lambda i: abs(wl[i]))[:k]
    order = sorted(order, key=lambda i: wl[i])
    Vd = V.tolist()
    return (_ac.marr([wl[i] for i in order]),
            _ac.marr([[Vd[r][i] for i in order] for r in range(n)]))


class _SparseLinalg:
    spsolve = staticmethod(spsolve)
    eigsh = staticmethod(eigsh)


class sparse:  # namespace mirror
    csc_matrix = csc_matrix
    csr_matrix = csr_matrix
    linalg = _SparseLinalg()

    @staticmethod
    def issparse(x):
        return isinstance(x, csc_matrix)


# ------------------------------------------------------------ io (MAT v5)

def loadmat(path, **kw):
    """Minimal MAT-file v5 reader: numeric/logical/char 2-D matrices,
    uncompressed or zlib-compressed elements."""
    del kw
    import struct
    import zlib
    out = {}
    with open(path, "rb") as fh:
        header = fh.read(128)
        if not header[:4] in (b"MATL",):
            raise ValueError("not a MAT v5 file")
        data = fh.read()

    def parse_element(buf, pos):
        dtype, nbytes = struct.unpack_from("<II", buf, pos)
        small = dtype >> 16
        if small:                       # small data element
            nbytes = small
            dtype &= 0xFFFF
            payload = buf[pos + 4:pos + 4 + nbytes]
            return dtype, payload, pos + 8
        payload = buf[pos + 8:pos + 8 + nbytes]
        adv = 8 + nbytes
        if nbytes % 8:
            adv += 8 - nbytes % 8
        return dtype, payload, pos + adv

    MI_MATRIX, MI_COMPRESSED = 14, 15
    NUM_FMT = {1: ("b", 1), 2: ("B", 1), 3: ("h", 2), 4: ("H", 2),
               5: ("i", 4), 6: ("I", 4), 7: ("f", 4), 9: ("d", 8),
               12: ("q", 8), 13: ("Q", 8)}

    def parse_matrix(payload):
        import struct as _st
        p = 0
        _t, flags, p = parse_element(payload, p)
        _t, dims_raw, p = parse_element(payload, p)
        _t, name_raw, p = parse_element(payload, p)
        name = name_raw.rstrip(b"\x00").decode("latin1")
        ndim = len(dims_raw) // 4
        dims = _st.unpack("<%di" % ndim, dims_raw)
        cls = flags[0] if flags else 0
        if cls in (1, 2, 5) or cls > 15:      # cell/struct/sparse: skip
            return name, None
        t, real_raw, p = parse_element(payload, p)
        if t == 16 or cls == 4:               # mxCHAR
            try:
                txt = real_raw.decode("utf-16-le") \
                    if t in (17, 16) and b"\x00" in real_raw \
                    else real_raw.decode("latin1")
            except UnicodeDecodeError:
                txt = real_raw.decode("latin1", "replace")
            return name, txt.replace("\x00", "")
        fmt, size = NUM_FMT.get(t, ("d", 8))
        cnt = len(real_raw) // size
        vals = list(_st.unpack("<%d%s" % (cnt, fmt), real_raw))
        if len(dims) == 2:
            r, c = dims
            # column-major
            mat = [[float(vals[j * r + i]) for j in range(c)]
                   for i in range(r)]
            return name, _ac.marr(mat)
        return name, _ac.marr([float(v) for v in vals])

    pos = 0
    while pos < len(data) - 8:
        dtype, payload, pos = parse_element(data, pos)
        if dtype == MI_COMPRESSED:
            sub = zlib.decompress(payload)
            t2, pl2, _ = parse_element(sub, 0)
            if t2 == MI_MATRIX:
                nm, val = parse_matrix(pl2)
                if val is not None:
                    out[nm] = val
        elif dtype == MI_MATRIX:
            nm, val = parse_matrix(payload)
            if val is not None:
                out[nm] = val
    return out


class io:  # namespace mirror
    loadmat = staticmethod(loadmat)


convolve = _nd_convolve          # scipy.ndimage.convolve import site


# ------------------------------------------------------------ spatial 2-D

class Delaunay:
    """2-D Delaunay triangulation (Bowyer-Watson)."""

    def __init__(self, points):
        P = _ac.atleast_2d(points)
        pts = [(float(r[0]), float(r[1])) for r in P.data]
        self.points = _ac.marr([[x, y] for x, y in pts])
        n = len(pts)
        # super-triangle
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dx = max(xs) - min(xs) or 1.0
        dy = max(ys) - min(ys) or 1.0
        mx = (max(xs) + min(xs)) / 2.0
        my = (max(ys) + min(ys)) / 2.0
        d = 20.0 * _bi.max(dx, dy)
        sp = [(mx - d, my - d), (mx + d, my - d), (mx, my + d)]
        allp = pts + sp
        tris = [(n, n + 1, n + 2)]

        def circum(tri):
            (ax, ay), (bx, by), (cx, cy) = (allp[tri[0]],
                                            allp[tri[1]],
                                            allp[tri[2]])
            dd = 2.0 * (ax * (by - cy) + bx * (cy - ay)
                        + cx * (ay - by))
            if dd == 0:
                return (0.0, 0.0), _math.inf
            ux = ((ax * ax + ay * ay) * (by - cy)
                  + (bx * bx + by * by) * (cy - ay)
                  + (cx * cx + cy * cy) * (ay - by)) / dd
            uy = ((ax * ax + ay * ay) * (cx - bx)
                  + (bx * bx + by * by) * (ax - cx)
                  + (cx * cx + cy * cy) * (bx - ax)) / dd
            r2 = (ax - ux) ** 2 + (ay - uy) ** 2
            return (ux, uy), r2

        for pi in range(n):
            px, py = allp[pi]
            bad = []
            for t in tris:
                (ux, uy), r2 = circum(t)
                if (px - ux) ** 2 + (py - uy) ** 2 <= r2 * (
                        1.0 + 1e-12):
                    bad.append(t)
            # boundary polygon of the bad region
            edges = {}
            for t in bad:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    key = (min(e), max(e))
                    edges[key] = edges.get(key, 0) + 1
            boundary = [e for e, c in edges.items() if c == 1]
            tris = [t for t in tris if t not in bad]
            for e in boundary:
                tris.append((e[0], e[1], pi))
        # drop super-triangle members
        self.simplices = _ac.marr(
            [[float(a), float(b), float(c)] for a, b, c in tris
             if a < n and b < n and c < n])

    def find_simplex(self, xi):
        pts = self.points.data
        tris = [[int(v) for v in row] for row in
                _ac.atleast_2d(self.simplices).data]
        q = [float(v) for v in _ac.asarray(xi)._flat()]

        def inside(t, x, y):
            (ax, ay), (bx, by), (cx, cy) = (pts[t[0]], pts[t[1]],
                                            pts[t[2]])
            d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
            d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
            d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
            neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (neg and pos)
        for ti, t in enumerate(tris):
            if inside(t, q[0], q[1]):
                return ti
        return -1


class Voronoi:
    """2-D Voronoi from the Delaunay dual (circumcenters)."""

    def __init__(self, points):
        self._tri = Delaunay(points)
        self.points = self._tri.points
        pts = self.points.data
        tris = [[int(v) for v in row] for row in
                _ac.atleast_2d(self._tri.simplices).data]
        verts = []
        for t in tris:
            (ax, ay), (bx, by), (cx, cy) = (pts[t[0]], pts[t[1]],
                                            pts[t[2]])
            dd = 2.0 * (ax * (by - cy) + bx * (cy - ay)
                        + cx * (ay - by))
            ux = ((ax * ax + ay * ay) * (by - cy)
                  + (bx * bx + by * by) * (cy - ay)
                  + (cx * cx + cy * cy) * (ay - by)) / dd
            uy = ((ax * ax + ay * ay) * (cx - bx)
                  + (bx * bx + by * by) * (ax - cx)
                  + (cx * cx + cy * cy) * (bx - ax)) / dd
            verts.append([ux, uy])
        self.vertices = _ac.marr(verts) if verts else _ac.marr([[]])
        # ridges: triangles sharing an edge -> segment between their
        # circumcenters; region per point = incident circumcenters
        edge_tris = {}
        for ti, t in enumerate(tris):
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(e), max(e))
                edge_tris.setdefault(key, []).append(ti)
        self.ridge_points = _ac.marr(
            [[float(a), float(b)] for (a, b), ts in edge_tris.items()
             if len(ts) == 2])
        self.ridge_vertices = [ts for ts in edge_tris.values()
                               if len(ts) == 2]
        npts = len(pts)
        regions = [[] for _ in range(npts)]
        for ti, t in enumerate(tris):
            for v in t:
                regions[v].append(ti)
        # order each region's circumcenters by angle around the point
        self.regions = []
        self.point_region = _ac.marr(
            [float(i) for i in range(npts)])
        for i in range(npts):
            cx, cy = pts[i]
            reg = sorted(regions[i], key=lambda ti: _math.atan2(
                verts[ti][1] - cy, verts[ti][0] - cx))
            self.regions.append(reg)


class spatial_full(spatial):
    Delaunay = Delaunay
    Voronoi = Voronoi


spatial.Delaunay = Delaunay
spatial.Voronoi = Voronoi


# ------------------------------------------------------------ schur

def schur(a, output="real"):
    """Real Schur decomposition via Hessenberg + shifted QR iteration.

    Returns (T, Z) with A = Z T Z^T, Z orthogonal, T quasi-upper-
    triangular. Verified by reconstruction + eigenvalue agreement.
    """
    del output
    A = _ac.atleast_2d(a)
    n = A.shape[0]
    H = [list(map(float, r)) for r in A.data]
    Z = [[1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]

    def apply_house(v, lo):
        m = len(v)
        vn2 = _math.fsum(u * u for u in v)
        if vn2 == 0.0:
            return
        # H rows and cols, Z cols
        for j in range(n):
            dot = _math.fsum(v[i] * H[lo + i][j] for i in range(m))
            c = 2.0 * dot / vn2
            for i in range(m):
                H[lo + i][j] -= c * v[i]
        for i in range(n):
            dot = _math.fsum(H[i][lo + t] * v[t] for t in range(m))
            c = 2.0 * dot / vn2
            for t in range(m):
                H[i][lo + t] -= c * v[t]
        for i in range(n):
            dot = _math.fsum(Z[i][lo + t] * v[t] for t in range(m))
            c = 2.0 * dot / vn2
            for t in range(m):
                Z[i][lo + t] -= c * v[t]

    # Hessenberg reduction
    for k in range(n - 2):
        x = [H[i][k] for i in range(k + 1, n)]
        nx = _math.sqrt(_math.fsum(u * u for u in x))
        if nx == 0.0:
            continue
        v = list(x)
        v[0] += _math.copysign(nx, x[0])
        apply_house(v, k + 1)

    # shifted QR with deflation (Givens-based, Wilkinson shift)
    def givens(i, j, cth, sth):
        for col in range(n):
            hi, hj = H[i][col], H[j][col]
            H[i][col] = cth * hi + sth * hj
            H[j][col] = -sth * hi + cth * hj
        for row in range(n):
            hi, hj = H[row][i], H[row][j]
            H[row][i] = cth * hi + sth * hj
            H[row][j] = -sth * hi + cth * hj
            zi, zj = Z[row][i], Z[row][j]
            Z[row][i] = cth * zi + sth * zj
            Z[row][j] = -sth * zi + cth * zj

    hi_idx = n - 1
    for _sweep in range(100 * n):
        # deflate
        while hi_idx > 0 and abs(H[hi_idx][hi_idx - 1]) < 1e-13 * (
                abs(H[hi_idx][hi_idx])
                + abs(H[hi_idx - 1][hi_idx - 1]) + 1e-300):
            H[hi_idx][hi_idx - 1] = 0.0
            hi_idx -= 1
        if hi_idx == 0:
            break
        # 2x2 block with complex eigenvalues? test and deflate pair
        if hi_idx >= 1:
            a11 = H[hi_idx - 1][hi_idx - 1]
            a12 = H[hi_idx - 1][hi_idx]
            a21 = H[hi_idx][hi_idx - 1]
            a22 = H[hi_idx][hi_idx]
            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr * tr - 4.0 * det
            if disc < 0 and (hi_idx == 1 or abs(
                    H[hi_idx - 1][hi_idx - 2]) < 1e-13 * (
                    abs(a11) + 1e-300)):
                if hi_idx >= 2:
                    H[hi_idx - 1][hi_idx - 2] = 0.0
                hi_idx -= 2
                if hi_idx <= 0:
                    break
                continue
        # Wilkinson shift from trailing 2x2
        a11 = H[hi_idx - 1][hi_idx - 1]
        a12 = H[hi_idx - 1][hi_idx]
        a21 = H[hi_idx][hi_idx - 1]
        a22 = H[hi_idx][hi_idx]
        tr = a11 + a22
        det = a11 * a22 - a12 * a21
        disc = tr * tr - 4.0 * det
        if disc >= 0:
            r1 = (tr + _math.copysign(_math.sqrt(disc), tr)) / 2.0
            r2 = det / r1 if r1 != 0 else 0.0
            mu = r1 if abs(r1 - a22) < abs(r2 - a22) else r2
        else:
            mu = a22
        # implicit single-shift QR on active block via Givens chase
        x = H[0][0] - mu if hi_idx == n - 1 else None
        lo = 0
        # find start of active block
        lo = hi_idx
        while lo > 0 and H[lo][lo - 1] != 0.0:
            lo -= 1
        x = H[lo][lo] - mu
        y = H[lo + 1][lo]
        for k in range(lo, hi_idx):
            r = _math.hypot(x, y)
            if r == 0.0:
                x = H[k + 1][k + 1] - mu if k + 1 < hi_idx else 0.0
                y = H[k + 2][k + 1] if k + 2 <= hi_idx else 0.0
                continue
            cth, sth = x / r, y / r
            givens(k, k + 1, cth, sth)
            if k + 2 <= hi_idx:
                x = H[k + 1][k]
                y = H[k + 2][k]
    return _ac.marr(H), _ac.marr(Z)


linalg.schur = staticmethod(schur)
