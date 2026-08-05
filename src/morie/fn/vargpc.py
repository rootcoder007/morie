# morie.fn -- function file (rootcoder007/morie)
"""Scalable variational Gaussian process classification (SVGP).

SOURCE.  Hensman, J., Matthews, A. and Ghahramani, Z. (2015), "Scalable
Variational Gaussian Process Classification", *Proceedings of the 18th
International Conference on Artificial Intelligence and Statistics*,
PMLR 38:351-360.

The inducing-point variational posterior q(u) = N(m, S) at inputs Z
induces, for each data point,

    q(f_n) = N( a_n' m ,  k_nn - a_n' k_mn + a_n' S a_n ),
    a_n    = K_mm^{-1} k_mn                                     (Eqs. 6-8)

and the bound the paper maximises is

    ELBO = sum_n E_{q(f_n)}[ log p(y_n | f_n) ] - KL( q(u) || p(u) ),

with the Bernoulli-through-a-sigmoid expectation done by Gauss-Hermite
quadrature -- the paper's own choice for the intractable one-dimensional
expectation.  The KL between two Gaussians is closed form,

    KL = (1/2)[ tr(K^{-1} S) + m' K^{-1} m - M + log|K| - log|S| ].

QUADRATURE.  The Gauss-Hermite nodes and weights are not tabulated here:
they are computed by Golub-Welsch, as the eigenvalues of the symmetric
tridiagonal Jacobi matrix with zero diagonal and off-diagonal
sqrt(k/2), with weights sqrt(pi) v_{1i}^2.  That makes them checkable --
``quad_check`` in the payload is the error in
E_{N(0,1)}[f^2] = 1, which a wrong node set cannot pass.

OPTIMISATION.  m and the Cholesky factor of S are fitted by gradient
ascent with central finite differences and deterministic backtracking
(halve the step while the ELBO would fall).  The paper uses analytic
gradients and stochastic optimisation; a deterministic, derivative-free
ascent is used here so both language arms perform the same arithmetic
and land on the same numbers.  That substitution is this
implementation's choice, stated rather than attributed.  The ELBO is
recorded every step and its monotonicity is asserted.

INDUCING POINTS Z are placed deterministically at evenly spaced ranks of
the data ordered by its first coordinate, not sampled.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["variational_gp_classifier"]


def _gh(n):
    """Gauss-Hermite nodes/weights by Golub-Welsch (physicists' weight)."""
    J = [[0.0] * n for _ in range(n)]
    for k in range(1, n):
        b = math.sqrt(k / 2.0)
        J[k][k - 1] = b
        J[k - 1][k] = b
    val, vec = core.jacobi(J)
    w = [math.sqrt(math.pi) * vec[0][i] * vec[0][i] for i in range(n)]
    return val, w


def _rbf(a, b, ls, var):
    s = 0.0
    for j in range(len(a)):
        s += (a[j] - b[j]) ** 2
    return var * math.exp(-0.5 * s / (ls * ls))


def _ltri(p, M):
    L = [[0.0] * M for _ in range(M)]
    t = 0
    for i in range(M):
        for j in range(i + 1):
            L[i][j] = p[t]
            t += 1
    return L


def variational_gp_classifier(X, y, X_test=None, m_inducing=4, lengthscale=1.0,
                              variance=1.0, n_quad=20, steps=40,
                              step_size=0.1, jitter=1e-8):
    """Fit an SVGP binary classifier and predict.

    Parameters
    ----------
    X : array-like
        n-by-d training inputs.
    y : array-like
        Binary labels in {0, 1}.
    X_test : array-like or None
        Test inputs; ``None`` predicts at ``X``.
    m_inducing : int
        M, 1 <= M <= n.
    lengthscale, variance : float
        RBF kernel hyperparameters, > 0.  Held fixed.
    n_quad : int
        Gauss-Hermite nodes, >= 2.
    steps : int
        Ascent steps.
    step_size : float
        Initial step, > 0.
    jitter : float
        Added to the diagonal of K_mm, >= 0.

    Returns
    -------
    RichResult
        ``prob`` (test predictive probabilities), ``pred`` (0/1),
        ``fit_prob``, ``fit_pred``, ``elbo``, ``elbo_path``,
        ``elbo_monotone``, ``kl``, ``m``, ``S``, ``Z``, ``quad_check``,
        ``n``, ``d``, ``m_inducing``.

    Raises
    ------
    ValueError
        Empty inputs, mismatched lengths, labels outside {0, 1},
        M outside 1..n, non-positive hyperparameters, ``n_quad`` < 2, or
        a test matrix of the wrong width.

    References
    ----------
    Hensman, J., Matthews, A. and Ghahramani, Z. (2015).  PMLR
    38:351-360.
    """
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("variational_gp_classifier: X is empty")
    d = len(A[0])
    yv = core.vec(y)
    if len(yv) != n:
        raise ValueError("variational_gp_classifier: X and y must have the same length")
    for v in yv:
        if v != 0.0 and v != 1.0:
            raise ValueError("variational_gp_classifier: y must be binary 0/1")
    M = int(m_inducing)
    if M < 1 or M > n:
        raise ValueError("variational_gp_classifier: m_inducing must lie in 1 .. n")
    ls = float(lengthscale)
    var = float(variance)
    if not (ls > 0.0) or not (var > 0.0):
        raise ValueError("variational_gp_classifier: lengthscale and variance must be positive")
    Q = int(n_quad)
    if Q < 2:
        raise ValueError("variational_gp_classifier: n_quad must be at least 2")
    if float(step_size) <= 0.0:
        raise ValueError("variational_gp_classifier: step_size must be positive")
    if float(jitter) < 0.0:
        raise ValueError("variational_gp_classifier: jitter must be non-negative")
    T = A if X_test is None else core.mat(X_test)
    if len(T) and len(T[0]) != d:
        raise ValueError("variational_gp_classifier: X_test must have d columns")
    order = sorted(range(n), key=lambda i: (A[i][0], i))
    Z = []
    for t in range(M):
        pos = 0 if M == 1 else int(round(t * (n - 1) / (M - 1.0)))
        Z.append(list(A[order[pos]]))
    Kmm = [[_rbf(Z[i], Z[j], ls, var) + (float(jitter) if i == j else 0.0)
            for j in range(M)] for i in range(M)]
    Lk = core.chol(Kmm)
    logdetK = 0.0
    for i in range(M):
        logdetK += 2.0 * math.log(Lk[i][i])
    Kinv = [[0.0] * M for _ in range(M)]
    for j in range(M):
        e = [1.0 if i == j else 0.0 for i in range(M)]
        col = core.cholsolve(Kmm, e)
        for i in range(M):
            Kinv[i][j] = col[i]
    Amat = []
    knn = []
    for i in range(n):
        kmn = [_rbf(Z[t], A[i], ls, var) for t in range(M)]
        Amat.append(core.cholsolve(Kmm, kmn))
        q = _rbf(A[i], A[i], ls, var)
        for t in range(M):
            q -= Amat[i][t] * kmn[t]
        knn.append(q)
    gx, gw = _gh(Q)
    sq = math.sqrt(math.pi)
    r2 = math.sqrt(2.0)

    def elbo_of(p):
        mu_u = p[:M]
        Lm = _ltri(p[M:], M)
        S = core.matmul(Lm, core.tr(Lm))
        tot = 0.0
        for i in range(n):
            a = Amat[i]
            mn = 0.0
            for t in range(M):
                mn += a[t] * mu_u[t]
            v = knn[i]
            for t in range(M):
                for u_ in range(M):
                    v += a[t] * S[t][u_] * a[u_]
            if v < 0.0:
                v = 0.0
            sd = math.sqrt(v)
            sgn = 1.0 if yv[i] == 1.0 else -1.0
            acc = 0.0
            for k in range(Q):
                f = mn + r2 * sd * gx[k]
                z = -sgn * f
                acc += gw[k] * (-(z if z > 0.0 else 0.0)
                                - math.log1p(math.exp(-abs(z))))
            tot += acc / sq
        tr = 0.0
        for t in range(M):
            for u_ in range(M):
                tr += Kinv[t][u_] * S[u_][t]
        qf = 0.0
        for t in range(M):
            for u_ in range(M):
                qf += mu_u[t] * Kinv[t][u_] * mu_u[u_]
        logdetS = 0.0
        for t in range(M):
            if Lm[t][t] == 0.0:
                return float("-inf"), 0.0
            logdetS += 2.0 * math.log(abs(Lm[t][t]))
        kl = 0.5 * (tr + qf - M + logdetK - logdetS)
        return tot - kl, kl

    p = [0.0] * M
    for i in range(M):
        for j in range(i + 1):
            p.append(Lk[i][j])
    cur, kl = elbo_of(p)
    path = [cur]
    h = 1e-5
    st = float(step_size)
    for _ in range(int(steps)):
        g = [0.0] * len(p)
        for t in range(len(p)):
            p[t] += h
            up, _ = elbo_of(p)
            p[t] -= 2.0 * h
            dn, _ = elbo_of(p)
            p[t] += h
            g[t] = (up - dn) / (2.0 * h)
        gs = 0.0
        for v in g:
            gs += v * v
        gs = math.sqrt(gs)
        if gs == 0.0:
            break
        s = st
        moved = False
        for _b in range(20):
            cand = [p[t] + s * g[t] / gs for t in range(len(p))]
            nv, nkl = elbo_of(cand)
            if nv > cur:
                p = cand
                cur = nv
                kl = nkl
                moved = True
                break
            s *= 0.5
        path.append(cur)
        if not moved:
            break
    mu_u = p[:M]
    Lm = _ltri(p[M:], M)
    S = core.matmul(Lm, core.tr(Lm))

    def predict(pts):
        out = []
        for x0 in pts:
            kmn = [_rbf(Z[t], x0, ls, var) for t in range(M)]
            a = core.cholsolve(Kmm, kmn)
            mn = 0.0
            for t in range(M):
                mn += a[t] * mu_u[t]
            v = _rbf(x0, x0, ls, var)
            for t in range(M):
                v -= a[t] * kmn[t]
            for t in range(M):
                for u_ in range(M):
                    v += a[t] * S[t][u_] * a[u_]
            if v < 0.0:
                v = 0.0
            sd = math.sqrt(v)
            acc = 0.0
            for k in range(Q):
                f = mn + r2 * sd * gx[k]
                acc += gw[k] * core.sigmoid(f)
            out.append(acc / sq)
        return out

    pr = predict(T)
    fp = pr if X_test is None else predict(A)
    qc = 0.0
    for k in range(Q):
        qc += gw[k] * (r2 * gx[k]) ** 2
    qc = abs(qc / sq - 1.0)
    mono = True
    for i in range(1, len(path)):
        if path[i] < path[i - 1] - 1e-10:
            mono = False
    return RichResult(
        title="Scalable variational GP classification",
        summary_lines=[("obs", n), ("inducing", M), ("ELBO", cur)],
        payload={
            "estimate": cur,
            "prob": pr,
            "pred": [1.0 if v >= 0.5 else 0.0 for v in pr],
            "fit_prob": fp,
            "fit_pred": [1.0 if v >= 0.5 else 0.0 for v in fp],
            "elbo": cur,
            "elbo_path": path,
            "elbo_monotone": 1.0 if mono else 0.0,
            "kl": kl,
            "m": mu_u,
            "S": S,
            "Z": Z,
            "quad_check": qc,
            "n": n,
            "d": d,
            "m_inducing": M,
            "method": "SVGP with Gauss-Hermite Bernoulli quadrature (Hensman, Matthews and Ghahramani 2015 Eqs. 6-8)",
        },
    )


def cheatsheet():
    return "vargpc: scalable variational GP classification (Hensman et al. 2015)"
