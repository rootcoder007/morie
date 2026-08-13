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
                if not (d > 0.0):
                    # A Cholesky factor exists only for a positive-definite
                    # matrix. Returning 0 here used to make cholsolve hand
                    # back the ZERO VECTOR, which a Newton step on a
                    # log-likelihood Hessian (negative definite) accepted
                    # silently: beta stayed at 0 and three-way parity passed
                    # at 1e-9 because both arms did the same thing. Solve
                    # against -H, or ridge the matrix into positive
                    # definiteness, but do not accept a wrong answer.
                    raise ValueError(
                        "chol: matrix is not positive definite "
                        "(pivot %d is %.17g); a Cholesky factor does not "
                        "exist. If this is a log-likelihood Hessian, solve "
                        "against -H." % (i, d))
                L[i][j] = math.sqrt(d)
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
    """Modified Bessel K_nu(x) for x > 0 and any real nu > 0.

    Computed from the integral representation

        K_nu(x) = Integral_0^inf exp(-x cosh t) cosh(nu t) dt

    (Watson 1944, section 6.22) by the trapezoidal rule on a fixed grid,
    step 0.01 out to t = 25.  The integrand is smooth, even in t and
    exponentially decaying, so the trapezoidal rule converges
    geometrically; the grid is fixed rather than adaptive so that the R
    mirror performs the identical arithmetic in the identical order.
    Each term is formed in logs and dropped once it underflows, which is
    what keeps cosh(nu t) from overflowing at large nu before
    exp(-x cosh t) has annihilated it.

    This replaced an earlier small-x branch that evaluated
    pi/2 (I_-nu - I_nu) / sin(nu pi).  That form has a removable pole at
    every integer nu, and stepping nu by 1e-8 to dodge it does not work:
    sin((1 + eps) pi) is NEGATIVE, so the old code returned K_1 with the
    wrong sign, and near the pole the I difference cancels to nothing.
    The integral has no pole, no branch and no cancellation.  Checked
    against base R besselK at nu = 0.5, 1, 2, 2.5, 4, 12 and x from
    0.001 to 6: 14 significant figures throughout.

    ``terms`` is retained for backward compatibility and is not used.
    """
    if x <= 0.0:
        return float("inf")
    h = 0.01
    n = 2500  # t runs 0 .. 25
    s = 0.0
    for i in range(n + 1):
        t = i * h
        az = abs(nu * t)
        lg = -x * math.cosh(t) + az + math.log1p(math.exp(-2.0 * az)) - math.log(2.0)
        term = math.exp(lg) if lg > -740.0 else 0.0
        s += term * (0.5 if (i == 0 or i == n) else 1.0)
    return s * h


# ---------------------------------------------- regression workhorses
#
# Written out rather than delegating to a library so that the R mirror
# performs the identical arithmetic in the identical order.


def logit_irls(X, y, iters=60, ridge=1e-10, tol=1e-13, penalty=0.0):
    """Logistic regression by iteratively reweighted least squares.

    Newton-Raphson on the log-likelihood, which for the canonical link
    is exactly IRLS: beta <- beta + (X' W X)^-1 X' (y - p), with
    W = diag(p (1 - p)).  Starting value zero, so the first step is the
    usual one from the sample mean.

    `penalty` adds a ridge term to the OBJECTIVE, maximising
    loglik(beta) - (penalty/2) sum_{j>=1} beta_j^2, so the score gains
    -penalty*beta and the Hessian gains penalty*I on the slopes. The
    intercept is left unpenalized: shrinking it would move the fitted
    marginal prevalence rather than the confounding relationship.

    `ridge` is a different thing and keeps its old meaning -- a
    numerical stabilizer inside the Newton-step solve, with no effect
    on the fitted coefficients.
    """
    n = len(X)
    p = len(X[0]) if n else 0
    beta = [0.0] * p
    for _ in range(iters):
        eta = matvec(X, beta)
        mu = [sigmoid(z) for z in eta]
        w = [mu[i] * (1.0 - mu[i]) for i in range(n)]
        XtWX = [[0.0] * p for _ in range(p)]
        Xtr = [0.0] * p
        for i in range(n):
            r = y[i] - mu[i]
            for a in range(p):
                Xtr[a] += X[i][a] * r
                for b in range(p):
                    XtWX[a][b] += X[i][a] * w[i] * X[i][b]
        pen = float(penalty)
        if pen > 0.0:
            for a in range(1, p):          # slopes only, not the intercept
                Xtr[a] -= pen * beta[a]
                XtWX[a][a] += pen
        step = ridgesolve(XtWX, Xtr, ridge)
        mx = 0.0
        for a in range(p):
            beta[a] += step[a]
            if abs(step[a]) > mx:
                mx = abs(step[a])
        if mx < tol:
            break
    return beta


def design(X, n):
    """Prepend an intercept column to a (possibly empty) covariate block."""
    if X is None:
        return [[1.0] for _ in range(n)]
    rows = mat(X)
    if not rows:
        return [[1.0] for _ in range(n)]
    return [[1.0] + list(rows[i]) for i in range(len(rows))]


def drdid_panel(dy, D, X=None, weights=None):
    """Doubly robust DiD for panel data, Sant'Anna and Zhao (2020) eq. (2.6).

    Sant'Anna, P. H. C. & Zhao, J. (2020) "Doubly robust
    difference-in-differences estimators", *Journal of Econometrics*
    219(1), 101-122, doi:10.1016/j.jeconom.2020.06.003.

        tau = E[(w1(D) - w0(D, X; pi)) (dY - mu_0(X))]
        w1  = D / E[D]
        w0  = [pi(X)(1-D)/(1-pi(X))] / E[pi(X)(1-D)/(1-pi(X))]

    with pi a logistic propensity score and mu_0 the outcome regression
    of dY on X among the untreated.  Returns tau, the influence function,
    and the standard error implied by it.
    """
    dyv = vec(dy)
    d = vec(D)
    n = len(dyv)
    Z = design(X, n)
    w = vec(weights) if weights is not None else [1.0] * n
    gam = logit_irls(Z, d, 60)
    # bounded away from 0 and 1: with a covariate that separates D the
    # IRLS fit diverges and 1 - pi underflows to zero, which is a
    # positivity violation, not an arithmetic accident.  Bounding makes
    # it visible in the weights rather than raising.
    pi = [min(max(sigmoid(v), 1e-12), 1.0 - 1e-12) for v in matvec(Z, gam)]
    # outcome regression among the untreated
    Z0 = [Z[i] for i in range(n) if d[i] < 0.5]
    y0 = [dyv[i] for i in range(n) if d[i] < 0.5]
    b0 = lstsq(Z0, y0) if Z0 else [0.0] * len(Z[0])
    mu0 = matvec(Z, b0)
    s1 = 0.0
    s0 = 0.0
    for i in range(n):
        s1 += w[i] * d[i]
        s0 += w[i] * pi[i] * (1.0 - d[i]) / (1.0 - pi[i])
    w1 = [w[i] * d[i] / s1 if s1 > 0.0 else 0.0 for i in range(n)]
    w0 = [w[i] * pi[i] * (1.0 - d[i]) / (1.0 - pi[i]) / s0 if s0 > 0.0 else 0.0
          for i in range(n)]
    tau = 0.0
    for i in range(n):
        tau += (w1[i] - w0[i]) * (dyv[i] - mu0[i])
    inf = [n * (w1[i] - w0[i]) * (dyv[i] - mu0[i]) - tau for i in range(n)]
    v = 0.0
    for x in inf:
        v += x * x
    se = math.sqrt(v / (n * n)) if n else float("nan")
    return {"tau": tau, "inf": inf, "se": se, "pi": pi, "mu0": mu0,
            "w1": w1, "w0": w0, "gamma": gam, "beta0": b0}


def mammen(i):
    """Mammen's two-point multiplier, taken at a van der Corput point.

    k1 = (1 - sqrt 5)/2 with probability (sqrt 5 + 1)/(2 sqrt 5), else
    k2 = (1 + sqrt 5)/2.  Mean 1, variance 1, third moment 1 -- the
    properties the multiplier bootstrap needs -- and here the "draw" is
    a deterministic low-discrepancy point, so both arms agree exactly.
    """
    r5 = math.sqrt(5.0)
    p = (r5 + 1.0) / (2.0 * r5)
    return (1.0 - r5) / 2.0 if vdc(i, 2) < p else (1.0 + r5) / 2.0


# --------------------------------------------------------------------
# IP weighting for time-varying treatments
#
# Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
# Boca Raton: Chapman & Hall/CRC, Ch. 12 and Ch. 21. Section 12.3 gives
# the time-fixed weights
#
#     W^A  = 1 / f(A | L)              (nonstabilized)
#     SW^A = f(A) / f(A | L)           (stabilized)
#
# and Sec. 21.2 generalises them to a treatment history A-bar with
# time-varying covariates L-bar as the product over time
#
#     W^Abar  = prod_k 1 / f(A_k | Abar_{k-1}, Lbar_k)
#     SW^Abar = prod_k f(A_k | Abar_{k-1}) / f(A_k | Abar_{k-1}, Lbar_k)
#
# The numerator model conditions on past treatment only; that is the
# whole difference between the two, and it is why the stabilized
# weights have the narrower range.
# --------------------------------------------------------------------


def treatment_density(A, X, kind="binary", ridge=1e-8, penalty=0.0):
    """f(A | X) evaluated at each observed A.

    "binary" fits a logistic model and returns g^A (1-g)^(1-A), the
    probability of the treatment value actually received -- which is
    what the denominator of an IP weight is, informally, in the book's
    own words. "normal" fits a linear model and returns the normal
    density at the observed dose, which is the generalized propensity
    score of a continuous treatment.
    """
    av = vec(A)
    n = len(av)
    Z = design(X, n)
    if kind == "binary":
        b = logit_irls(Z, av, 60, ridge, penalty=float(penalty))
        g = [sigmoid(v) for v in matvec(Z, b)]
        dens = [g[i] if av[i] > 0.5 else 1.0 - g[i] for i in range(n)]
        return dens, {"coef": b, "prob": g}
    if kind == "normal":
        b = lstsq(Z, av, ridge)
        mu = matvec(Z, b)
        resid = [av[i] - mu[i] for i in range(n)]
        k = len(b)
        df = n - k if n > k else 1
        s2 = sum(r * r for r in resid) / df
        if s2 <= 0.0:
            raise ValueError(
                "treatment_density: the treatment model fits the dose "
                "exactly, so f(A|X) is degenerate and no IP weight exists")
        c = 1.0 / math.sqrt(2.0 * math.pi * s2)
        dens = [c * math.exp(-0.5 * r * r / s2) for r in resid]
        return dens, {"coef": b, "mu": mu, "sigma2": s2}
    raise ValueError("treatment_density: kind must be 'binary' or "
                     "'normal', got %r" % (kind,))


def ip_weights(A, X_denom, X_num=None, kind="binary", stabilize=True,
               trim=None, ridge=1e-8, penalty=0.0):
    """One time point's IP weight, Sec. 12.3.

    `X_num` is the numerator model's covariates. Passing None gives the
    marginal numerator f(A), the book's usual choice; passing past
    treatment gives f(A_k | Abar_{k-1}), which is what Sec. 21.2 needs.
    `stabilize=False` gives the nonstabilized 1/f(A|L).

    `trim` truncates the weights at the given upper quantile and its
    complement -- not part of the book's estimator, and off by default,
    because truncation trades variance for bias and that is a decision
    the caller should make deliberately.

    The returned info carries diagnostics: the mean weight (which
    Sec. 12.3 says to check is 1), the largest weight, the Kish
    effective sample size, and for a Gaussian continuous treatment
    ``finite_variance``, which is False when the confounders explain
    more than half the variance of the dose. In that regime the
    stabilized weight has infinite variance: its expectation is still
    exactly 1, but the sample mean converges arbitrarily slowly and
    comes out low, so the mean-1 check stops being informative. It is
    reported rather than silently tolerated.
    """
    den, dinfo = treatment_density(A, X_denom, kind=kind, ridge=ridge,
                                   penalty=penalty)
    n = len(den)
    for i in range(n):
        if den[i] <= 0.0:
            raise ValueError(
                "ip_weights: f(A|L) is zero for observation %d, so the "
                "positivity condition fails in the sample and the weight "
                "is undefined" % i)
    if not stabilize:
        num = [1.0] * n
        ninfo = None
    else:
        # the numerator model is not penalized: it is the stabilizing
        # factor f(A) or f(A | Abar), and shrinking it would change the
        # estimand rather than the weights' variance
        num, ninfo = treatment_density(A, X_num, kind=kind, ridge=ridge)
    w = [num[i] / den[i] for i in range(n)]
    diag = _weight_diagnostics(w, dinfo, ninfo, kind, stabilize)
    if trim is not None:
        q = float(trim)
        if not 0.5 < q < 1.0:
            raise ValueError("ip_weights: trim must be in (0.5, 1), got %r"
                             % (trim,))
        hi = quantile7(w, q)
        lo = quantile7(w, 1.0 - q)
        w = [min(max(v, lo), hi) for v in w]
    out = {"denominator": den, "numerator": num,
           "denominator_fit": dinfo, "numerator_fit": ninfo}
    out.update(diag)
    return w, out


def _weight_diagnostics(w, dinfo, ninfo, kind, stabilize):
    """Kish effective sample size, and the finite-variance condition.

    The effective sample size (sum w)^2 / sum w^2 is Kish, L. (1965)
    *Survey Sampling*, Wiley, and says how many equally weighted
    observations the weighted sample is worth. A value far below n is
    the honest signal that a few observations are carrying the estimate.

    `finite_variance` is None unless both models are Gaussian, where the
    condition sigma_c^2 > sigma_m^2 / 2 can be checked exactly.
    """
    n = len(w)
    s1 = sum(w)
    s2 = sum(v * v for v in w)
    ess = (s1 * s1 / s2) if s2 > 0.0 else 0.0
    out = {"mean_weight": s1 / n if n else float("nan"),
           "max_weight": max(w) if w else float("nan"),
           "effective_sample_size": ess,
           "ess_fraction": ess / n if n else float("nan"),
           "finite_variance": None,
           "variance_ratio": None}
    if kind == "normal" and stabilize and ninfo is not None:
        sc2 = float(dinfo["sigma2"])
        sm2 = float(ninfo["sigma2"])
        out["variance_ratio"] = sc2 / (0.5 * sm2) if sm2 > 0.0 else \
            float("inf")
        out["finite_variance"] = bool(sc2 > 0.5 * sm2)
    return out


def ip_weights_history(A_hist, L_hist, kind="binary", stabilize=True,
                       trim=None, ridge=1e-8, penalty=0.0):
    """Sec. 21.2's product over time.

    `A_hist` is a list of K treatment vectors, one per time point;
    `L_hist` a list of K covariate matrices, each holding the
    covariates measured AT that time. At time k the denominator model
    conditions on L-bar_k -- every covariate block from time 0 to time
    k, accumulated here rather than expected from the caller -- and on
    A-bar_{k-1}; the numerator model on A-bar_{k-1} alone.
    """
    K = len(A_hist)
    if K == 0:
        raise ValueError("ip_weights_history: need at least one time point")
    if len(L_hist) != K:
        raise ValueError("ip_weights_history: %d treatment times but %d "
                         "covariate blocks" % (K, len(L_hist)))
    n = len(vec(A_hist[0]))
    w = [1.0] * n
    per_time = []
    past = []                       # columns of treatment history so far
    lbar = []                       # columns of COVARIATE history so far
    for k in range(K):
        ak = vec(A_hist[k])
        if len(ak) != n:
            raise ValueError("ip_weights_history: time %d has %d rows, "
                             "time 0 has %d" % (k, len(ak), n))
        if L_hist[k] is not None:
            block = mat(L_hist[k])
            if len(block) != n:
                raise ValueError(
                    "ip_weights_history: covariate block at time %d has "
                    "%d rows, treatment has %d" % (k, len(block), n))
            for c in range(len(block[0])):
                lbar.append([row[c] for row in block])
        # Sec. 21.2: the denominator conditions on L-bar_k, the whole
        # covariate history to date, and on A-bar_{k-1}; the numerator
        # on A-bar_{k-1} alone.
        den_X = _bind_cols(None, lbar + past, n)
        num_X = _bind_cols(None, past, n) if past else None
        wk, info = ip_weights(ak, den_X, num_X, kind=kind,
                              stabilize=stabilize, ridge=ridge,
                              penalty=penalty)
        for i in range(n):
            w[i] *= wk[i]
        per_time.append({"time": k, "weight": wk, "info": info})
        past = [list(col) for col in past] + [list(ak)]
    if trim is not None:
        q = float(trim)
        if not 0.5 < q < 1.0:
            raise ValueError("ip_weights_history: trim must be in (0.5, 1)")
        hi = quantile7(w, q)
        lo = quantile7(w, 1.0 - q)
        w = [min(max(v, lo), hi) for v in w]
    return w, per_time


def _bind_cols(M, extra_cols, n):
    """Column-bind a matrix (or None) with a list of length-n columns."""
    if M is None and not extra_cols:
        return None
    rows = []
    for i in range(n):
        r = []
        if M is not None:
            r.extend(list(M[i]))
        for col in extra_cols:
            r.append(float(col[i]))
        rows.append(r)
    return rows


def wls(X, y, w, ridge=1e-10):
    """Weighted least squares, the fitting step of every MSM here.

    Returns the coefficients and the robust (sandwich) standard errors.
    The robust variance is not optional: the pseudo-population created
    by the weights has the wrong sample size, so the model-based
    standard error is wrong by construction, and the book says to use a
    robust variance estimator throughout Part II.
    """
    yv = vec(y)
    n = len(yv)
    Z = design(X, n)
    wv = [float(v) for v in vec(w)]
    if len(wv) != n:
        raise ValueError("wls: %d weights for %d observations"
                         % (len(wv), n))
    if any(v < 0.0 for v in wv):
        raise ValueError("wls: weights must be non-negative")
    k = len(Z[0])
    XtWX = [[sum(wv[i] * Z[i][a] * Z[i][b] for i in range(n))
             for b in range(k)] for a in range(k)]
    XtWy = [sum(wv[i] * Z[i][a] * yv[i] for i in range(n))
            for a in range(k)]
    for a in range(k):
        XtWX[a][a] += ridge
    beta = cholsolve(XtWX, XtWy)
    fitted = matvec(Z, beta)
    resid = [yv[i] - fitted[i] for i in range(n)]
    # sandwich: (X'WX)^-1 (sum w^2 e^2 x x') (X'WX)^-1
    meat = [[sum((wv[i] * resid[i]) ** 2 * Z[i][a] * Z[i][b]
                 for i in range(n))
             for b in range(k)] for a in range(k)]
    bread_cols = [cholsolve(XtWX, [1.0 if j == a else 0.0
                                   for j in range(k)]) for a in range(k)]
    var = []
    for a in range(k):
        row = []
        for b in range(k):
            t = 0.0
            for u in range(k):
                for v in range(k):
                    t += bread_cols[a][u] * meat[u][v] * bread_cols[b][v]
            row.append(t)
        var.append(row)
    se = [math.sqrt(var[a][a]) if var[a][a] > 0.0 else float("nan")
          for a in range(k)]
    return {"coef": beta, "se": se, "vcov": var, "fitted": fitted,
            "resid": resid, "n": n}


def logistic_fluctuation(outcome, offset_logit, H, rows=None, iters=100,
                         tol=1e-12):
    """One-dimensional logistic fluctuation of a bounded regression.

    Solves the score equation sum_i H_i (Y_i - expit(offset_i + eps H_i))
    = 0 by Newton, which is exactly a univariate logistic regression of
    `outcome` on the clever covariate `H` with `offset_logit` as offset.
    This is the targeting step shared by the TMLEs in this package.

    van der Laan, M. J. & Rubin, D. (2006) "Targeted maximum likelihood
    learning", *The International Journal of Biostatistics* 2(1),
    article 11, doi:10.2202/1557-4679.1043.
    """
    n = len(outcome)
    idx = list(range(n)) if rows is None else list(rows)
    if not idx or all(abs(H[i]) < 1e-14 for i in idx):
        return 0.0
    eps = 0.0
    for _ in range(iters):
        num = den = 0.0
        for i in idx:
            pr = sigmoid(offset_logit[i] + eps * H[i])
            num += H[i] * (outcome[i] - pr)
            den += H[i] * H[i] * pr * (1.0 - pr)
        if den < 1e-14:
            break
        step = num / den
        eps += step
        if abs(step) < tol:
            break
    return eps


def tmle_ate(y, D, X=None, trim=0.0, link="logit"):
    """Targeted maximum likelihood for the ATE.

    van der Laan, M. J. & Rubin, D. (2006) "Targeted maximum likelihood
    learning", *The International Journal of Biostatistics* 2(1),
    article 11, doi:10.2202/1557-4679.1043,
    *The International Journal of Biostatistics* 2(1), article 11.  The
    initial outcome fit Qbar is fluctuated along the logistic submodel
    whose score is the clever covariate

        H(D, X) = D / g(X) - (1 - D) / (1 - g(X))

    with g the propensity score; the fluctuation parameter eps solves
    the score equation and is found by a Newton step, after which
    psi = mean( Qbar*(1, X) - Qbar*(0, X) ).  y is scaled to [0, 1] so
    the logistic fluctuation is valid for continuous outcomes too, which
    is the bounded-continuous trick of Gruber, S. & van der Laan, M. J.
    (2010) "A targeted maximum likelihood estimator of a causal effect
    on a bounded continuous outcome", *The International Journal of
    Biostatistics* 6(1), article 26, doi:10.2202/1557-4679.1260.
    """
    yv = vec(y)
    d = vec(D)
    n = len(yv)
    Z = design(X, n)
    g = [sigmoid(v) for v in matvec(Z, logit_irls(Z, d, 60))]
    t = float(trim)
    if t > 0.0:
        g = [min(max(v, t), 1.0 - t) for v in g]
    lo = min(yv)
    hi = max(yv)
    rng = hi - lo if hi > lo else 1.0
    ys = [(v - lo) / rng for v in yv]
    # initial outcome regression, with treatment as a covariate
    Q = [[1.0] + [d[i]] + list(Z[i][1:]) for i in range(n)]
    bq = lstsq(Q, ys)
    q1 = [0.0] * n
    q0 = [0.0] * n
    qa = [0.0] * n
    for i in range(n):
        row1 = [1.0, 1.0] + list(Z[i][1:])
        row0 = [1.0, 0.0] + list(Z[i][1:])
        s1 = 0.0
        s0 = 0.0
        for j in range(len(bq)):
            s1 += bq[j] * row1[j]
            s0 += bq[j] * row0[j]
        q1[i] = min(max(s1, 1e-8), 1.0 - 1e-8)
        q0[i] = min(max(s0, 1e-8), 1.0 - 1e-8)
        qa[i] = q1[i] if d[i] > 0.5 else q0[i]
    H = [d[i] / g[i] - (1.0 - d[i]) / (1.0 - g[i]) for i in range(n)]
    # one-dimensional logistic fluctuation, solved by Newton on eps
    eps = 0.0
    for _ in range(80):
        num = 0.0
        den = 0.0
        for i in range(n):
            z = math.log(qa[i] / (1.0 - qa[i])) + eps * H[i]
            p = sigmoid(z)
            num += H[i] * (ys[i] - p)
            den += H[i] * H[i] * p * (1.0 - p)
        if den <= 0.0:
            break
        step = num / den
        eps += step
        if abs(step) < 1e-13:
            break
    q1s = [sigmoid(math.log(q1[i] / (1.0 - q1[i])) + eps / g[i]) for i in range(n)]
    q0s = [sigmoid(math.log(q0[i] / (1.0 - q0[i])) - eps / (1.0 - g[i]))
           for i in range(n)]
    psi_s = 0.0
    for i in range(n):
        psi_s += (q1s[i] - q0s[i]) / n
    psi = psi_s * rng
    m1 = 0.0
    m0 = 0.0
    for i in range(n):
        m1 += q1s[i] / n
        m0 += q0s[i] / n
    inf = []
    for i in range(n):
        qas = q1s[i] if d[i] > 0.5 else q0s[i]
        inf.append(rng * (H[i] * (ys[i] - qas) + (q1s[i] - q0s[i]) - psi_s))
    v = 0.0
    for x in inf:
        v += x * x
    se = math.sqrt(v / (n * n)) if n else float("nan")
    return {"psi": psi, "se": se, "eps": eps, "g": g, "q1": q1s, "q0": q0s,
            "inf": inf, "ey1": lo + rng * m1, "ey0": lo + rng * m0,
            "scale": rng, "shift": lo}
