# morie.fn -- slice s03 shared helpers (rootcoder007/morie)
"""Private numeric helpers shared by the slice-s03 function modules.

Nothing here is public API.  Every routine is written so that the R
mirror ``aaa_helpers_s03.R`` performs the *same* floating-point
operations in the *same* order, which is what lets the three-way parity
harness assert agreement at 1e-9 rather than at some looser tolerance.

Two rules are enforced throughout:

* no pseudo-random numbers -- anything that would classically be a draw
  is either supplied by the caller or replaced by a deterministic
  low-discrepancy / fixed-index construction;
* no library linear algebra where a hand-written loop will do, because
  LAPACK and a native kernel need not round identically.
"""

from __future__ import annotations

import math

__all__: list[str] = []


# --------------------------------------------------------------- shapes


def vec(x):
    """Flatten any nested sequence (or scalar) to a list of floats."""
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for e in x:
        if isinstance(e, (list, tuple)):
            out.extend(vec(e))
        else:
            try:
                out.append(float(e))
            except (TypeError, ValueError):
                out.append(float("nan"))
    return out


def mat(x):
    """Coerce to a list-of-rows matrix of floats."""
    if x is None:
        return []
    rows = list(x)
    if not rows:
        return []
    if not isinstance(rows[0], (list, tuple)):
        return [[float(e)] for e in rows]
    return [[float(e) for e in r] for r in rows]


def nrow(A):
    return len(A)


def ncol(A):
    return len(A[0]) if A else 0


def tr(A):
    """Transpose."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += Ai[p] * B[p][j]
            out[i][j] = s
    return out


def matvec(A, v):
    out = [0.0] * len(A)
    for i in range(len(A)):
        s = 0.0
        Ai = A[i]
        for p in range(len(v)):
            s += Ai[p] * v[p]
        out[i] = s
    return out


def crossprod(A):
    """A' A."""
    return matmul(tr(A), A)


# ------------------------------------------------------ linear algebra


def chol(A):
    """Lower Cholesky factor, plain Cholesky-Banachiewicz order."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for p in range(j):
                s += L[i][p] * L[j][p]
            if i == j:
                d = A[i][i] - s
                L[i][j] = math.sqrt(d) if d > 0.0 else 0.0
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] != 0.0 else 0.0
    return L


def cholsolve(A, b):
    """Solve a symmetric positive-definite system by Cholesky."""
    n = len(A)
    L = chol(A)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for p in range(i):
            s -= L[i][p] * y[p]
        y[i] = s / L[i][i] if L[i][i] != 0.0 else 0.0
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for p in range(i + 1, n):
            s -= L[p][i] * x[p]
        x[i] = s / L[i][i] if L[i][i] != 0.0 else 0.0
    return x


def ridgesolve(A, b, ridge=1e-10):
    """Cholesky solve with a small ridge, for near-singular normal equations."""
    n = len(A)
    M = [[A[i][j] + (ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
    return cholsolve(M, b)


def lstsq(X, y, ridge=1e-10):
    """Ridge-stabilised least squares via the normal equations."""
    XtX = crossprod(X)
    Xty = matvec(tr(X), y)
    return ridgesolve(XtX, Xty, ridge)


def jacobi(A, sweeps=60):
    """Symmetric eigenproblem by cyclic Jacobi.

    Returns (values, vectors) with values ascending and each eigenvector
    sign-fixed so that its largest-magnitude entry is positive.  The sign
    fix is what makes the R mirror agree: eigenvector signs are not
    determined by the eigenproblem itself.
    """
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += M[i][j] * M[i][j]
        if off <= 1e-30:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(M[p][q]) <= 1e-300:
                    continue
                theta = (M[q][q] - M[p][p]) / (2.0 * M[p][q])
                t = (1.0 if theta >= 0.0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp = M[k][p]
                    mkq = M[k][q]
                    M[k][p] = c * mkp - s * mkq
                    M[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk = M[p][k]
                    mqk = M[q][k]
                    M[p][k] = c * mpk - s * mqk
                    M[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [M[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (vals[i], i))
    ev = [vals[i] for i in order]
    vec_ = [[V[r][order[j]] for j in range(n)] for r in range(n)]
    for j in range(n):
        big = 0
        for r in range(n):
            if abs(vec_[r][j]) > abs(vec_[big][j]) + 1e-15:
                big = r
        if vec_[big][j] < 0.0:
            for r in range(n):
                vec_[r][j] = -vec_[r][j]
    return ev, vec_


# ------------------------------------------------------------- scalars


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def gelu(z):
    """Exact GELU, x * Phi(x) (Hendrycks and Gimpel 2016)."""
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))


def swish(z, beta=1.0):
    """Swish_beta(x) = x sigma(beta x) (Ramachandran et al. 2017)."""
    return z * sigmoid(beta * z)


def relu(z):
    return z if z > 0.0 else 0.0


def softmax(v):
    if not v:
        return []
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = 0.0
    for x in e:
        s += x
    return [x / s for x in e]


def logsumexp(v):
    if not v:
        return float("-inf")
    m = max(v)
    if m == float("-inf"):
        return m
    s = 0.0
    for x in v:
        s += math.exp(x - m)
    return m + math.log(s)


def mean(v):
    n = len(v)
    if n == 0:
        return float("nan")
    s = 0.0
    for x in v:
        s += x
    return s / n


def variance(v, ddof=1):
    n = len(v)
    if n - ddof <= 0:
        return float("nan")
    m = mean(v)
    s = 0.0
    for x in v:
        s += (x - m) * (x - m)
    return s / (n - ddof)


def sd(v, ddof=1):
    return math.sqrt(variance(v, ddof))


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return float("nan")
    h = n // 2
    return s[h] if n % 2 == 1 else 0.5 * (s[h - 1] + s[h])


def mad(v, constant=1.4826):
    m = median(v)
    return constant * median([abs(x - m) for x in v])


def quantile7(v, p):
    """Type-7 quantile, the default of R's ``quantile()``."""
    s = sorted(v)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = lo + 1 if lo + 1 < n else n - 1
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def rank_avg(v):
    """Average ranks, ties resolved as in R's ``rank()``."""
    n = len(v)
    order = sorted(range(n), key=lambda i: (v[i], i))
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def corr(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxy = sxx = syy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    d = math.sqrt(sxx * syy)
    return sxy / d if d > 0.0 else float("nan")


# ------------------------------------------------ deterministic "draws"


def vdc(i, base=2):
    """Van der Corput point -- the deterministic stand-in for a uniform.

    Using a low-discrepancy sequence rather than a pseudo-random stream
    keeps the Python and R arms bit-identical without either having to
    reimplement the other's generator.
    """
    f = 1.0
    r = 0.0
    k = int(i) + 1
    while k > 0:
        f = f / base
        r = r + f * (k % base)
        k = k // base
    return r


def unif(n, base=2):
    return [vdc(i, base) for i in range(n)]


def qnorm(p):
    """Normal quantile, Wichura's AS 241 -- the algorithm R's qnorm uses."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (
            ((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                 + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
               + 133.14166789178437745) * r + 3.387132872796366608)
        den = (
            ((((((5226.495278852854561 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                 + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
               + 42.313330701600911252) * r + 1.0)
        return q * num / den
    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (
            ((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                 + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
               + 4.6303378461565452959) * r + 1.42343711074968357734)
        den = (
            ((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                 + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
               + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5.0
        num = (
            ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                 + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
               + 5.4637849111641143699) * r + 6.6579046435011037772)
        den = (
            ((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                 + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
               + 0.59983220655588793769) * r + 1.0)
    val = num / den
    return -val if q < 0.0 else val


def pnorm(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normdraws(n, base=2):
    """Deterministic standard-normal points: AS 241 applied to van der Corput."""
    return [qnorm(vdc(i, base)) for i in range(n)]


def lgamma(x):
    return math.lgamma(x)


def digamma(x):
    """Digamma by recurrence plus the standard asymptotic series."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return (
        r
        + math.log(x)
        - 0.5 / x
        + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f * (-1.0 / 132.0)))))
    )


def besselk(nu, x, terms=160):
    """Modified Bessel K_nu(x) for x > 0, by the series for small x and the
    uniform asymptotic continued fraction for large x.

    Used only by the Matern variogram, where nu is a fixed smoothness and
    x = h/a.  Written as a plain series so the R mirror matches term for
    term rather than deferring to R's ``besselK``.
    """
    if x <= 0.0:
        return float("inf")
    if x < 2.0:
        # K_nu = pi/2 * (I_{-nu} - I_nu) / sin(nu pi); half-integer nu is
        # handled by the closed forms below to avoid the removable pole.
        if abs(nu - round(nu)) < 1e-12:
            nu = round(nu) + 1e-8

        def bessel_i(order, z):
            s = 0.0
            for k in range(terms):
                s += math.exp(
                    (2.0 * k + order) * math.log(z / 2.0)
                    - math.lgamma(k + 1.0)
                    - math.lgamma(k + order + 1.0)
                )
            return s

        return (
            0.5
            * math.pi
            * (bessel_i(-nu, x) - bessel_i(nu, x))
            / math.sin(nu * math.pi)
        )
    # large x: K_nu(x) ~ sqrt(pi/(2x)) e^-x sum_k a_k / x^k  (Hankel)
    mu = 4.0 * nu * nu
    term = 1.0
    s = 1.0
    for k in range(1, 24):
        term *= (mu - (2.0 * k - 1.0) ** 2) / (8.0 * k * x)
        s += term
    return math.sqrt(math.pi / (2.0 * x)) * math.exp(-x) * s
