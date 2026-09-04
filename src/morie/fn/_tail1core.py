# morie.fn -- function file (rootcoder007/morie)
"""Shared numeric helpers for the tail1 batch.

Pure-Python linear algebra and distribution functions used by the
long-tail single-paper modules in this batch.  Nothing here is public
API; the module is imported only by the batch's own function modules.
Deliberately self-contained so that the R mirrors (which lean on base
R's ``solve``/``chol``/``eigen``/``pnorm``) have an exact counterpart.
"""

import math

__all__ = []


# --------------------------------------------------------------- shapes


def _is_seq(v):
    """Anything that iterates like a sequence, INCLUDING the native
    array type. Testing for list/tuple alone sent a marr straight to
    float(), which rejects anything longer than one element -- so every
    tail1 entry point refused the package's own array."""
    return (not isinstance(v, (str, bytes))
            and hasattr(v, "__iter__"))


def vec(x):
    """Flatten any nested sequence to a flat list of floats."""
    out = []

    def walk(v):
        if _is_seq(v):
            for e in v:
                walk(e)
        else:
            out.append(float(v))

    walk(x)
    return out


def mat(X):
    """Coerce to a list-of-rows matrix of floats."""
    if not _is_seq(X):
        return [[float(X)]]
    rows = list(X)
    if len(rows) == 0:
        return []
    if _is_seq(rows[0]):
        return [[float(v) for v in row] for row in rows]
    return [[float(v)] for v in rows]


def rowmat(x):
    """Coerce a flat sequence to a 1-row matrix."""
    return [vec(x)]


def shape(A):
    return (len(A), len(A[0]) if A else 0)


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(col) for col in zip(*A)]


def matmul(A, B):
    p = len(B)
    q = len(B[0])
    Bt = transpose(B)
    return [[sum(row[k] * Bt[j][k] for k in range(p)) for j in range(q)] for row in A]


def matvec(A, b):
    return [sum(a * bb for a, bb in zip(row, b)) for row in A]


def cbind1(X):
    """Prepend an intercept column."""
    return [[1.0] + list(row) for row in X]


# --------------------------------------------------------- basic stats


def mean(x):
    x = vec(x)
    return sum(x) / len(x) if x else float("nan")


def var(x, ddof=1):
    x = vec(x)
    n = len(x)
    if n - ddof <= 0:
        return float("nan")
    m = sum(x) / n
    return sum((v - m) ** 2 for v in x) / (n - ddof)


def sd(x, ddof=1):
    v = var(x, ddof)
    return math.sqrt(v) if v == v and v >= 0 else float("nan")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm2(a):
    return math.sqrt(sum(v * v for v in a))


def cumsum(x):
    out, s = [], 0.0
    for v in x:
        s += v
        out.append(s)
    return out


# ------------------------------------------------------- linear algebra


def solve(A, B):
    """Gaussian elimination with partial pivoting; B a matrix."""
    n = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    m = len(B[0])
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("singular matrix")
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:n + m] for row in M]


def solvev(A, b):
    return [row[0] for row in solve(A, [[v] for v in b])]


def inv(A):
    return solve(A, eye(len(A)))


def chol(A):
    """Lower-triangular Cholesky factor."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix not positive definite")
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def eigsym(A, sweeps=60):
    """Jacobi eigendecomposition of a symmetric matrix.

    Returns ``(values, vectors)`` with values in DECREASING order and
    ``vectors[i][j]`` the i-th component of eigenvector j.  Each column
    is sign-fixed so that its largest-magnitude entry is positive, which
    is what makes cross-language parity possible at all.
    """
    n = len(A)
    a = [list(r) for r in A]
    v = eye(n)
    for _ in range(sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off <= 1e-300:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    vals = [vals[i] for i in order]
    vecs = [[v[r][i] for i in order] for r in range(n)]
    for j in range(n):
        col = [vecs[r][j] for r in range(n)]
        k = max(range(n), key=lambda r: abs(col[r]))
        if col[k] < 0:
            for r in range(n):
                vecs[r][j] = -vecs[r][j]
    return vals, vecs


def lstsq(X, y):
    """Least squares by modified Gram-Schmidt QR.

    Returns ``(beta, fitted, resid, xtxinv)``.  Normal equations are
    avoided on purpose -- squaring the design squares its condition
    number and silently loses the small singular values.
    """
    n, p = shape(X)
    Q = [list(row) for row in X]
    R = [[0.0] * p for _ in range(p)]
    for j in range(p):
        for i in range(j):
            R[i][j] = sum(Q[r][i] * Q[r][j] for r in range(n))
            for r in range(n):
                Q[r][j] -= R[i][j] * Q[r][i]
        R[j][j] = math.sqrt(sum(Q[r][j] ** 2 for r in range(n)))
        if R[j][j] < 1e-12:
            R[j][j] = 1e-12
        for r in range(n):
            Q[r][j] /= R[j][j]
    qty = [sum(Q[r][j] * y[r] for r in range(n)) for j in range(p)]
    beta = [0.0] * p
    for j in range(p - 1, -1, -1):
        beta[j] = (qty[j] - sum(R[j][k] * beta[k] for k in range(j + 1, p))) / R[j][j]
    fitted = [sum(X[r][j] * beta[j] for j in range(p)) for r in range(n)]
    resid = [y[r] - fitted[r] for r in range(n)]
    rinv = inv(R)
    xtxinv = matmul(rinv, transpose(rinv))
    return beta, fitted, resid, xtxinv


def hatdiag(X, xtxinv):
    """Leverages h_ii = x_i' (X'X)^-1 x_i."""
    return [dot(row, matvec(xtxinv, row)) for row in X]


# ---------------------------------------------------- distributions


def pnorm(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def dnorm(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


_A241 = (3.387132872796366608, 133.14166789178437745, 1971.5909503065514427,
         13731.693765509461125, 45921.953931549871457, 67265.770927008700853,
         33430.575583588128105, 2509.0809287301226727)
_B241 = (42.313330701600911252, 687.1870074920579083, 5394.1960214247511077,
         21213.794301586595867, 39307.89580009271061, 28729.085735721942674,
         5226.495278852545925)
_C241 = (1.42343711074968357734, 4.6303378461565452959, 5.7694972214606914055,
         3.64784832476320460504, 1.27045825245236838258, 0.24178072517745061177,
         0.0227238449892691845833, 7.7454501427834140764e-4)
_D241 = (2.05319162663775882187, 1.6763848301838038494, 0.68976733498510000455,
         0.14810397642748007459, 0.0151986665636164571966,
         5.475938084995344946e-4, 1.05075007164441684324e-9)
_E241 = (6.6579046435011037772, 5.4637849111641143699, 1.7848265399172913358,
         0.29656057182850489123, 0.026532189526576123093,
         0.0012426609473880784386, 2.71155556874348757815e-5,
         2.01033439929228813265e-7)
_F241 = (0.59983220655588793769, 0.13692988092273580531, 0.0148753612908506148525,
         7.868691311456132591e-4, 1.8463183175100546818e-5,
         1.4215117583164458887e-7, 2.04426310338993978564e-15)


def qnorm(p):
    """Wichura's AS 241 -- accurate to full double precision."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = _A241[7]
        den = _B241[6]
        for k in range(6, -1, -1):
            num = num * r + _A241[k]
        for k in range(5, -1, -1):
            den = den * r + _B241[k]
        return q * num / (den * r + 1.0)
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num, den = _C241[7], _D241[6]
        for k in range(6, -1, -1):
            num = num * r + _C241[k]
        for k in range(5, -1, -1):
            den = den * r + _D241[k]
    else:
        r -= 5.0
        num, den = _E241[7], _F241[6]
        for k in range(6, -1, -1):
            num = num * r + _E241[k]
        for k in range(5, -1, -1):
            den = den * r + _F241[k]
    val = num / (den * r + 1.0)
    return -val if q < 0 else val


def _gammap(a, x):
    """Regularised lower incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
        return s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def pchisq(x, df):
    return _gammap(df / 2.0, x / 2.0)


def pgamma(x, shape_, rate=1.0):
    return _gammap(shape_, x * rate)


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
    return h


def pbeta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def pt(t, df):
    x = df / (df + t * t)
    p = 0.5 * pbeta(x, df / 2.0, 0.5)
    return p if t <= 0 else 1.0 - p


def pf(f, df1, df2):
    if f <= 0.0:
        return 0.0
    return pbeta(df1 * f / (df1 * f + df2), df1 / 2.0, df2 / 2.0)


# ------------------------------------------------------------- rng


class Lcg:
    """Lehmer minstd generator, ``s <- 48271 s mod (2^31 - 1)``.

    Chosen because every intermediate fits exactly in a float64, so the
    Python and R arms produce bit-identical streams -- which is the only
    reason a projection or sign-flip default can be compared across
    languages at 1e-9.  Callers who care should pass their own noise.
    """

    M = 2147483647

    def __init__(self, seed=1):
        s = int(seed) % self.M
        self.s = s if s > 0 else 1

    def unif(self):
        self.s = (48271 * self.s) % self.M
        return self.s / self.M

    def norm(self):
        return qnorm(self.unif())

    def rademacher(self):
        return 1.0 if self.unif() < 0.5 else -1.0
