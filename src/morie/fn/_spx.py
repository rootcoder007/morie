# morie.fn -- function file (rootcoder007/morie)
"""Shared numeric primitives for the sp* fill-in modules.

Pure standard library. Every routine here is mirrored line-for-line as a
``.morie_spx_*`` helper in the R file ``aaa_sp_fill.R`` so that the two
language arms execute the SAME arithmetic in the SAME order; that is what
lets ``/tmp/dscratch/sp_parity.{py,R}`` compare to an absolute 1e-9 rather
than to a hand-waved tolerance.

Two deliberate choices exist only for parity:

* every iterative routine runs a FIXED number of iterations instead of
  stopping on a tolerance, because an early exit taken on one arm and not
  the other silently changes the answer;
* eigenvectors are sign-fixed (largest-magnitude component made positive)
  because the sign of an eigenvector is arbitrary and R and Python would
  otherwise disagree by a factor of -1.
"""

from math import atan2, cos, fsum, log, pi, sin, sqrt

from ._rgcore import aslist

__all__ = []


def vec(x, name="x"):
    v = aslist(x)
    if not v:
        raise ValueError("`%s` must contain at least one value" % name)
    for t in v:
        if t != t or t in (float("inf"), float("-inf")):
            raise ValueError("`%s` must be finite" % name)
    return v


def mat(a, name="a"):
    try:
        rows = [aslist(r) for r in a]
    except TypeError:
        raise ValueError("`%s` must be a matrix" % name)
    if not rows or not rows[0]:
        raise ValueError("`%s` must be a non-empty matrix" % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("`%s` must be rectangular" % name)
        for t in r:
            if t != t or t in (float("inf"), float("-inf")):
                raise ValueError("`%s` must be finite" % name)
    return rows


def sqmat(a, n=None, name="w"):
    m = mat(a, name)
    if len(m) != len(m[0]):
        raise ValueError("`%s` must be square" % name)
    if n is not None and len(m) != n:
        raise ValueError("`%s` must be %d by %d" % (name, n, n))
    return m


def mean(v):
    return fsum(v) / len(v)


def dev(v):
    m = mean(v)
    return [t - m for t in v]


def dot(a, b):
    return fsum([a[i] * b[i] for i in range(len(a))])


def transpose(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def matvec(a, b):
    return [dot(row, b) for row in a]


def matmul(a, b):
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def trace(a):
    return fsum([a[i][i] for i in range(len(a))])


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve(a, b):
    """Gauss-Jordan with partial pivoting; raises on a singular system."""
    n = len(a)
    if len(b) != n or len(a[0]) != n:
        raise ValueError("linear system is not square or is inconsistent")
    m = [list(a[i]) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(m[r][c]) > abs(m[p][c]):
                p = r
        if abs(m[p][c]) < 1e-300:
            raise ValueError("linear system is singular")
        if p != c:
            m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / pv
            if f == 0.0:
                continue
            for k in range(c, n + 1):
                m[r][k] = m[r][k] - f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def logabsdet(a):
    """(sign, log|det|) by LU with partial pivoting."""
    n = len(a)
    m = [list(r) for r in a]
    sign = 1.0
    acc = 0.0
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(m[r][c]) > abs(m[p][c]):
                p = r
        if abs(m[p][c]) < 1e-300:
            return 0.0, float("-inf")
        if p != c:
            m[c], m[p] = m[p], m[c]
            sign = -sign
        pv = m[c][c]
        if pv < 0:
            sign = -sign
        acc = acc + log(abs(pv))
        for r in range(c + 1, n):
            f = m[r][c] / pv
            if f == 0.0:
                continue
            for k in range(c, n):
                m[r][k] = m[r][k] - f * m[c][k]
    return sign, acc


def lstsq(a, y, ridge=0.0):
    """Normal-equation least squares; `ridge` only guards exact collinearity."""
    at = transpose(a)
    g = matmul(at, a)
    if ridge:
        for i in range(len(g)):
            g[i][i] = g[i][i] + ridge
    return solve(g, matvec(at, y))


def fixsign(v):
    """Make the largest-magnitude component positive (ties -> first index)."""
    j = 0
    for i in range(len(v)):
        if abs(v[i]) > abs(v[j]):
            j = i
    if v[j] < 0:
        return [-t for t in v]
    return list(v)


def topeigs(a, k, iters=400):
    """Top-`k` eigenpairs of a SYMMETRIC matrix by power iteration + deflation.

    The start vector is fixed and slightly non-uniform, so it is not
    orthogonal to the leading eigenvector of a matrix with constant rows
    (an all-ones start is, and that failure is silent).
    """
    n = len(a)
    if k < 1 or k > n:
        raise ValueError("k must lie between 1 and the matrix order")
    m = [list(r) for r in a]
    vals = []
    vecs = []
    for _ in range(k):
        v = [float((i % 7) + 1) for i in range(n)]
        nv = sqrt(dot(v, v))
        v = [t / nv for t in v]
        for _ in range(iters):
            u = matvec(m, v)
            nu = sqrt(dot(u, u))
            if nu < 1e-300:
                break
            v = [t / nu for t in u]
        lam = dot(v, matvec(m, v))
        v = fixsign(v)
        vals.append(lam)
        vecs.append(v)
        for i in range(n):
            for j in range(n):
                m[i][j] = m[i][j] - lam * v[i] * v[j]
    return vals, vecs


def dft(x):
    """Plain O(n^2) DFT, X_k = sum_u x_u exp(-i w_k u), u and k from 0."""
    n = len(x)
    re = []
    im = []
    for k in range(n):
        w = -2.0 * pi * k / n
        re.append(fsum([x[u] * cos(w * u) for u in range(n)]))
        im.append(fsum([x[u] * sin(w * u) for u in range(n)]))
    return re, im


def idftre(re, im):
    n = len(re)
    out = []
    for i in range(n):
        w = 2.0 * pi * i / n
        out.append(fsum([re[k] * cos(w * k) - im[k] * sin(w * k)
                         for k in range(n)]) / n)
    return out


def phase(re, im):
    return [atan2(im[k], re[k]) for k in range(len(re))]


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        raise ValueError("the median of an empty vector is undefined")
    h = n // 2
    if n % 2:
        return s[h]
    return 0.5 * (s[h - 1] + s[h])


def eucdist(a, b):
    return sqrt(fsum([(a[i] - b[i]) ** 2 for i in range(len(a))]))


def normcdf(z):
    """Phi(z) via erf; used only for the two-sided normal p-values."""
    from math import erf
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def twosidep(z):
    return 2.0 * (1.0 - normcdf(abs(z)))


def cheatsheet():
    return "_spx: shared pure-stdlib numeric kit for the sp* modules"
