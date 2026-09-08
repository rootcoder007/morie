"""Numerics shared by the wave-3 modules, written for cross-arm equality.

Nothing here is novel. It exists because the obvious call in each
language is a DIFFERENT function from the obvious call in the other, and
a module that reaches for the obvious one cannot then be held to its R
arm at the twelfth digit:

  sum()        CPython 3.12+ applies Neumaier compensation to a run of
               floats; R accumulates in 80-bit long double. Two good
               answers, not the same answer.
  %*% / dot    goes through BLAS in R, which may reassociate and use
               FMA. Written out here instead.
  ^ vs **      R special-cases an integer exponent to repeated squaring
               and 0.5 to sqrt(); Python calls libm pow(). Never used
               here for anything iterated.
  pnorm, pt,   separate implementations in the two languages, agreeing
  lgamma, erf  to about 1e-15 and no further. Written out here so both
               arms run the SAME algorithm.

The R mirror is R/aaa_helpers_w3num.R, function for function, with the
prefix .w3_ in place of the leading underscore.
"""

import math

__all__ = ["csum", "dot", "chol", "solve_chol", "inv_from_chol", "ols",
           "lgamma", "gammp", "gammq", "ncdf", "npdf", "nppf", "betainc",
           "t_sf", "bisect", "simpson", "nelder_mead", "logsumexp"]


def csum(vals):
    """Neumaier-compensated sum. Not sum(): see the module docstring."""
    s = 0.0
    c = 0.0
    for v in vals:
        t = float(v)
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def dot(a, b):
    """Compensated dot product. Not sum(x*y for ...), same reason."""
    s = 0.0
    c = 0.0
    for x, y in zip(a, b):
        t = float(x) * float(y)
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def logsumexp(vals):
    """log(sum(exp(v))), shifted by the maximum so it cannot overflow."""
    vs = [float(v) for v in vals]
    if not vs:
        return float("-inf")
    m = max(vs)
    if m == float("-inf"):
        return m
    return m + math.log(csum(math.exp(v - m) for v in vs))


def chol(a):
    """Lower Cholesky factor L with A = L L'. Explicit, not a library."""
    p = len(a)
    lo = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            s = a[i][j] - dot(lo[i][:j], lo[j][:j])
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lo[i][j] = math.sqrt(s)
            else:
                lo[i][j] = s / lo[j][j]
    return lo


def solve_chol(lo, b):
    """Solve L L' x = b by forward then back substitution."""
    p = len(lo)
    z = [0.0] * p
    for i in range(p):
        z[i] = (b[i] - dot(lo[i][:i], z[:i])) / lo[i][i]
    x = [0.0] * p
    for i in range(p - 1, -1, -1):
        acc = csum(lo[k][i] * x[k] for k in range(i + 1, p))
        x[i] = (z[i] - acc) / lo[i][i]
    return x


def inv_from_chol(lo):
    """(L L')^-1, column by column from the factor."""
    p = len(lo)
    cols = [solve_chol(lo, [1.0 if k == j else 0.0 for k in range(p)])
            for j in range(p)]
    return [[cols[j][i] for j in range(p)] for i in range(p)]


def ols(y, design):
    """Least squares by the normal equations.

    Returns beta, the residual sum of squares, the residual df, sigma2,
    (X'X)^-1 and the fitted values.
    """
    n = len(y)
    p = len(design[0])
    xtx = [[csum(design[i][a] * design[i][b] for i in range(n))
            for b in range(p)] for a in range(p)]
    xty = [csum(design[i][a] * y[i] for i in range(n)) for a in range(p)]
    lo = chol(xtx)
    beta = solve_chol(lo, xty)
    fitted = [dot(design[i], beta) for i in range(n)]
    rss = csum((y[i] - fitted[i]) * (y[i] - fitted[i]) for i in range(n))
    df = n - p
    return {"beta": beta, "rss": rss, "df": df,
            "sigma2": rss / df if df > 0 else float("nan"),
            "xtx_inv": inv_from_chol(lo), "fitted": fitted, "chol": lo}


_LG = (76.18009172947146, -86.50532032941677, 24.01409824083091,
       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)


def lgamma(z):
    """Lanczos log-gamma. Not math.lgamma: R's is a different routine."""
    x = float(z)
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for j in range(6):
        x += 1.0
        ser += _LG[j] / x
    return -tmp + math.log(2.5066282746310005 * ser / z)


def gammp(a, x):
    """Regularised lower incomplete gamma P(a, x).

    Series below a+1, Lentz continued fraction for the complement above:
    the crossover is where each is the convergent one, and using the
    wrong side is where a hand-rolled version loses its digits.
    """
    a = float(a)
    x = float(x)
    if x < 0.0 or a <= 0.0:
        raise ValueError("gammp: need a > 0 and x >= 0")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 3e-16:
                break
        return s * math.exp(-x + a * math.log(x) - lgamma(a))
    return 1.0 - _gammcf(a, x)


def gammq(a, x):
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x).

    Computed without forming 1 - P when x is large, so the far tail
    keeps its significant digits instead of cancelling against 1.
    """
    a = float(a)
    x = float(x)
    if x < 0.0 or a <= 0.0:
        raise ValueError("gammq: need a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - gammp(a, x)
    return _gammcf(a, x)


def _gammcf(a, x):
    """Q(a, x) by Lentz's continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 501):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return math.exp(-x + a * math.log(x) - lgamma(a)) * h


def ncdf(z):
    """Standard normal CDF, via the incomplete gamma.

    Phi(z) = 1/2 (1 + sign(z) P(1/2, z^2/2)) is an identity, not an
    approximation, so this is exactly as accurate as gammp -- and the R
    arm runs the same series rather than R's own pnorm.
    """
    z = float(z)
    if z == 0.0:
        return 0.5
    p = gammp(0.5, 0.5 * z * z)
    return 0.5 * (1.0 + p) if z > 0.0 else 0.5 * (1.0 - p)


def npdf(z):
    z = float(z)
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def nppf(p, lo=-40.0, hi=40.0):
    """Standard normal quantile by bisection on ncdf.

    Slower than a rational approximation and exactly as accurate as the
    CDF it inverts, which is the property that matters here.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("nppf: p must lie strictly inside (0, 1)")
    return bisect(lambda z: ncdf(z) - p, lo, hi)


def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
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
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x)
                          - lb) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Upper tail of Student's t."""
    return 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))


def bisect(f, lo, hi, iters=200):
    """Root of f on a bracketing interval, by plain bisection.

    A fixed iteration count rather than a tolerance loop: the two arms
    then take exactly the same number of steps and end at exactly the
    same point, which a tolerance test cannot guarantee.
    """
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0.0) == (fhi > 0.0):
        raise ValueError("bisect: the interval does not bracket a root")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simpson(f, a, b, n=200):
    """Composite Simpson on n panels (rounded up to an even number).

    Fixed panels, not adaptive: an adaptive rule branches on a
    tolerance test and the two arms would take different numbers of
    subdivisions on the very integrals where it matters.
    """
    n = int(n)
    if n % 2:
        n += 1
    h = (b - a) / n
    terms = [f(a), f(b)]
    terms.extend(4.0 * f(a + h * k) for k in range(1, n, 2))
    terms.extend(2.0 * f(a + h * k) for k in range(2, n, 2))
    return h / 3.0 * csum(terms)


def nelder_mead(f, x0, step=0.1, iters=400, alpha=1.0, gamma=2.0,
                rho=0.5, sigma=0.5):
    """Nelder-Mead simplex minimisation.

    Derivative-free and deterministic given the starting point, which is
    what a likelihood with a step-function selection term needs -- the
    objective is not differentiable at the cutoffs, so a gradient method
    would be answering a question the model does not pose.

    The initial simplex is the standard one: x0 plus a step along each
    coordinate, scaled by the coordinate when it is non-zero so the
    simplex is not tiny in a large parameter and huge in a small one.
    A fixed iteration count keeps the two arms in lockstep.
    """
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] = p[i] + (step * p[i] if p[i] != 0.0 else step)
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(int(iters)):
        order = sorted(range(n + 1), key=lambda i: (vals[i], i))
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        cen = [csum(pts[i][j] for i in range(n)) / n for j in range(n)]
        xr = [cen[j] + alpha * (cen[j] - pts[n][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + gamma * (xr[j] - cen[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                pts[n], vals[n] = xe, fe
            else:
                pts[n], vals[n] = xr, fr
            continue
        if fr < vals[n - 1]:
            pts[n], vals[n] = xr, fr
            continue
        if fr < vals[n]:
            xc = [cen[j] + rho * (xr[j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc <= fr:
                pts[n], vals[n] = xc, fc
                continue
        else:
            xc = [cen[j] + rho * (pts[n][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[n]:
                pts[n], vals[n] = xc, fc
                continue
        for i in range(1, n + 1):
            pts[i] = [pts[0][j] + sigma * (pts[i][j] - pts[0][j])
                      for j in range(n)]
            vals[i] = f(pts[i])
    best = min(range(n + 1), key=lambda i: (vals[i], i))
    return {"x": pts[best], "value": vals[best]}
