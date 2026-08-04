# morie.fn -- function file (rootcoder007/morie)
"""Shared numeric helpers for the s04 long-tail batch.

Private.  Imported only by the s04 function modules.  Everything here
has a line-for-line counterpart in ``aaa_s04core.R`` so the Python and
R arms can be compared value-for-value; where base R already supplies
the primitive (``solve``, ``quantile``) the R side is a naming shim.

All iterative routines run a FIXED number of iterations with no
tolerance-based early exit -- an early exit is the one thing guaranteed
to make two arms disagree, because the arms round differently in the
last bit and then stop on different sweeps.
"""

import math

from . import _tail1core as C

__all__ = []


def expit(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p):
    return math.log(p / (1.0 - p))


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def median(x):
    x = sorted(C.vec(x))
    n = len(x)
    if n == 0:
        return float("nan")
    m = n // 2
    return x[m] if n % 2 else 0.5 * (x[m - 1] + x[m])


def quantile7(x, p):
    """R default quantile, type 7."""
    x = sorted(C.vec(x))
    n = len(x)
    if n == 0:
        return float("nan")
    if n == 1:
        return x[0]
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return x[lo] + (h - lo) * (x[hi] - x[lo])


def order(x):
    """Indices that sort x ascending, ties broken by original position."""
    x = C.vec(x)
    return sorted(range(len(x)), key=lambda i: (x[i], i))


def rank_avg(x):
    """Average ranks, 1-based -- the R rank() default, ties.method average."""
    x = C.vec(x)
    n = len(x)
    o = order(x)
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[o[j + 1]] == x[o[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[o[k]] = avg
        i = j + 1
    return r


def softmax(v):
    m = max(v)
    e = [math.exp(t - m) for t in v]
    s = sum(e)
    return [t / s for t in e]


def glmbin(X, y, iters=25, ridge=1e-8):
    """Logistic IRLS over a design that already carries its intercept.

    FIXED iteration count.  A tiny ridge keeps the weighted normal
    equations solvable under separation; at 1e-8 it sits far below any
    digit anyone reports.
    """
    n, p = C.shape(X)
    y = C.vec(y)
    beta = [0.0] * p
    for _ in range(iters):
        eta = [C.dot(X[i], beta) for i in range(n)]
        mu = [expit(e) for e in eta]
        w = [clip(m * (1.0 - m), 1e-10, 0.25) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) + (ridge if a == b else 0.0)
              for b in range(p)] for a in range(p)]
        rhs = [sum(X[i][a] * w[i] * z[i] for i in range(n)) for a in range(p)]
        beta = C.solvev(A, rhs)
    return beta


def rbf(X, Z, ell=1.0):
    """Squared-exponential kernel matrix with unit signal variance."""
    out = []
    for a in X:
        row = []
        for b in Z:
            s = sum((u - v) ** 2 for u, v in zip(a, b))
            row.append(math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def gppost(K, Ks, Kss, y, noise=1e-6):
    """GP posterior mean and variance at the test points.

    ``K`` train-train, ``Ks`` train-test, ``Kss`` the test prior
    variances.  Returns ``(mean, var)``.
    """
    n = len(K)
    A = [[K[i][j] + (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
    alpha = C.solvev(A, C.vec(y))
    m = len(Kss)
    mean = [sum(Ks[i][j] * alpha[i] for i in range(n)) for j in range(m)]
    V = C.solve(A, Ks)
    var = [Kss[j] - sum(Ks[i][j] * V[i][j] for i in range(n)) for j in range(m)]
    return mean, var


def colstd(X):
    """Column-standardise; a zero-variance column is left at zero."""
    n, p = C.shape(X)
    out = [[0.0] * p for _ in range(n)]
    for j in range(p):
        col = [X[i][j] for i in range(n)]
        m = sum(col) / n
        s = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        for i in range(n):
            out[i][j] = (col[i] - m) / s if s > 0 else 0.0
    return out


def euclid(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def sgn(v):
    """Sign mapped onto {-1, +1}; zero goes to +1 so the range is binary."""
    return 1.0 if v >= 0.0 else -1.0


def rnd(v):
    """Half-away-from-zero rounding.

    Deliberately not the language round().  Python and R both round half
    to even, but they disagree about which values are exactly half once
    binary representation is involved, and a quantiser that flips a
    level on that disagreement is a parity failure waiting to happen.
    """
    return sgn(v) * float(int(abs(v) + 0.5))


def qr_mgs(A):
    """Thin QR by modified Gram-Schmidt; returns ``(Q, R)``.

    Modified rather than classical Gram-Schmidt, and certainly not the
    normal equations: squaring the matrix squares its condition number
    and loses the small singular values outright.  Diagonal entries of
    ``R`` come out non-negative by construction, so ``Q`` is unique and
    there is no sign convention left to disagree about across arms --
    which is exactly the freedom that LAPACK and LINPACK QR use
    differently.
    """
    n, p = C.shape(A)
    Q = [list(row) for row in A]
    R = [[0.0] * p for _ in range(p)]
    for j in range(p):
        for i in range(j):
            R[i][j] = sum(Q[r][i] * Q[r][j] for r in range(n))
            for r in range(n):
                Q[r][j] -= R[i][j] * Q[r][i]
        R[j][j] = math.sqrt(sum(Q[r][j] ** 2 for r in range(n)))
        d = R[j][j] if R[j][j] > 1e-300 else 1e-300
        for r in range(n):
            Q[r][j] /= d
    return Q, R


def rank_first(x):
    """Ranks 1..n with ties broken by original position."""
    x = C.vec(x)
    o = order(x)
    r = [0] * len(x)
    for pos, i in enumerate(o):
        r[i] = pos + 1
    return r


def medmodels(Y, A, M, Cc=None):
    """Fit the VanderWeele mediation pair and return ``(theta, beta, cbar)``.

    Outcome model ``Y = th0 + th1 a + th2 m + th3 a m + th4' c`` and
    mediator model ``M = b0 + b1 a + b2' c``.  ``cbar`` is the covariate
    mean vector, which is where the decomposition is evaluated.
    """
    Y = C.vec(Y)
    A = C.vec(A)
    M = C.vec(M)
    n = len(Y)
    Cm = C.mat(Cc) if Cc is not None else [[] for _ in range(n)]
    XO = [[1.0, A[i], M[i], A[i] * M[i]] + list(Cm[i]) for i in range(n)]
    XM = [[1.0, A[i]] + list(Cm[i]) for i in range(n)]
    theta, _, _, _ = ols(XO, Y)
    beta, _, _, _ = ols(XM, M)
    q = len(Cm[0]) if Cm and Cm[0] else 0
    cbar = [sum(Cm[i][j] for i in range(n)) / n for j in range(q)]
    return theta, beta, cbar


def fourway(theta, beta, cbar, a=1.0, astar=0.0, m=0.0):
    """The VanderWeele four-way decomposition from fitted coefficients.

    Returns ``(cde, intref, intmed, pie, te)``.  At ``m = 0`` these are
    the expressions printed in VanderWeele (2014); the ``- m`` inside
    ``intref`` is the general controlled level.
    """
    d = a - astar
    bc = beta[0] + beta[1] * astar + sum(beta[2 + j] * cbar[j] for j in range(len(cbar)))
    cde = (theta[1] + theta[3] * m) * d
    intref = theta[3] * (bc - m) * d
    intmed = theta[3] * beta[1] * d * d
    pie = (theta[2] * beta[1] + theta[3] * beta[1] * astar) * d
    return cde, intref, intmed, pie, cde + intref + intmed + pie


def tmle(y, D, W, gbound=0.025):
    """One targeted-maximum-likelihood pass for a binary point treatment.

    ``W`` must already carry its intercept column.  Propensity by
    fixed-iteration IRLS, initial outcome model by least squares, and a
    closed-form linear fluctuation -- no line search, no tolerance.
    Returns a dict with ``psi``, ``se``, ``eps``, ``g``, ``H``, ``Q1``,
    ``Q0`` (targeted), and ``ic``.
    """
    y = C.vec(y)
    D = C.vec(D)
    n = len(y)
    gb = glmbin(W, D)
    g = [clip(expit(C.dot(W[i], gb)), gbound, 1.0 - gbound) for i in range(n)]
    des = [[D[i]] + list(W[i]) for i in range(n)]
    qb, _, _, _ = ols(des, y)
    Q1 = [C.dot([1.0] + list(W[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]), qb) for i in range(n)]
    Q = [Q1[i] if D[i] > 0.5 else Q0[i] for i in range(n)]
    H = [D[i] / g[i] - (1.0 - D[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (y[i] - Q[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    Qs = [Q[i] + eps * H[i] for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (y[i] - Qs[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return {"psi": psi, "se": se, "eps": eps, "g": g, "H": H,
            "Q1": Q1s, "Q0": Q0s, "ic": ic, "n": n}


def ols(X, y):
    """Least squares by the SAME modified Gram-Schmidt in both arms.

    The shared tail1 core solves this with each language own QR --
    modified Gram-Schmidt on the Python side, LINPACK/LAPACK Householder
    on the R side.  On a well-conditioned design they agree to the last
    bit; on an ill-conditioned one they part company around 1e-8, which
    is above the parity threshold and below anything a user would
    notice, so it hides until a cross-language check finds it.  Routing
    the s04 modules through one algorithm removes the whole class.

    Returns ``(beta, fitted, resid, xtxinv)``.
    """
    X = C.mat(X) if not (X and isinstance(X[0], list)) else X
    n, p = C.shape(X)
    y = C.vec(y)
    Q, R = qr_mgs(X)
    qty = [sum(Q[r][j] * y[r] for r in range(n)) for j in range(p)]
    beta = [0.0] * p
    for j in range(p - 1, -1, -1):
        d = R[j][j] if abs(R[j][j]) > 1e-300 else 1e-300
        beta[j] = (qty[j] - sum(R[j][k] * beta[k] for k in range(j + 1, p))) / d
    fitted = [sum(X[r][j] * beta[j] for j in range(p)) for r in range(n)]
    resid = [y[r] - fitted[r] for r in range(n)]
    rinv = _triinv(R, p)
    xtxinv = C.matmul(rinv, C.transpose(rinv))
    return beta, fitted, resid, xtxinv


def _triinv(R, p):
    """Inverse of an upper-triangular matrix by back substitution.

    Floors the pivot rather than testing a condition number, so a
    rank-deficient design produces the same large numbers in both arms
    instead of one language raising and the other returning.
    """
    out = [[0.0] * p for _ in range(p)]
    for j in range(p):
        for i in range(j, -1, -1):
            d = R[i][i] if abs(R[i][i]) > 1e-300 else 1e-300
            s = (1.0 if i == j else 0.0) - sum(R[i][k] * out[k][j] for k in range(i + 1, p))
            out[i][j] = s / d
    return out
