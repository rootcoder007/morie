"""Small dense linear-algebra and special-function helpers for the multivariate distribution modules."""

import math


def as_matrix(M):
    A = [[float(v) for v in row] for row in M]
    if not A or any(len(r) != len(A) for r in A):
        raise ValueError("matrix must be square and non-empty")
    return A


def cholesky(A):
    """Lower-triangular L with L L' = A (A symmetric positive definite)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def logdet_chol(L):
    return 2 * sum(math.log(L[i][i]) for i in range(len(L)))


def forward(L, b):
    z = []
    for i in range(len(L)):
        z.append((b[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i])
    return z


def inverse_spd(A):
    L = cholesky(A)
    n = len(A)
    cols = []
    for j in range(n):
        e = [float(i == j) for i in range(n)]
        z = forward(L, e)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (z[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        cols.append(x)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def trace_prod(A, B):
    return sum(A[i][k] * B[k][i] for i in range(len(A)) for k in range(len(A)))


def log_mvgamma(a, p):
    """log of the multivariate gamma function Gamma_p(a)."""
    return p * (p - 1) / 4 * math.log(math.pi) + sum(math.lgamma(a - j / 2) for j in range(p))


def log_bessel_i(nu, z):
    """log I_nu(z) for nu >= 0, z > 0, from the ascending series summed in log space."""
    if z <= 0:
        return 0.0 if nu == 0 else -math.inf
    lh = math.log(z / 2)
    terms = []
    k = 0
    best = -math.inf
    while True:
        t = (2 * k + nu) * lh - math.lgamma(k + 1) - math.lgamma(k + nu + 1)
        terms.append(t)
        best = max(best, t)
        if k > z and t < best - 40:
            break
        k += 1
    return best + math.log(sum(math.exp(t - best) for t in terms))


def points(x, d=None):
    """A single point (flat sequence) or a list of points."""
    if x and isinstance(x[0], (int, float)):
        return [[float(v) for v in x]], True
    return [[float(v) for v in row] for row in x], False
