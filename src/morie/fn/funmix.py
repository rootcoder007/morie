# morie.fn -- function file (rootcoder007/morie)
r"""Clustering curves, not the numbers that happen to be on them.

Growth curves, spectra and load profiles arrive as many measurements per
subject, and clustering them by treating each time point as a separate
variable throws away the one thing that makes them curves: neighbouring
times carry nearly the same information. James and Sugar's functional
clustering model puts the mixture where it belongs -- on the
coefficients of a spline basis rather than on the raw grid:

.. math:: y_i(t) = \mathbf b(t)'c_i + \varepsilon_i(t), \qquad
          c_i \sim \sum_{j=1}^K \pi_j\,N(\mu_j, \Sigma_j),

so a curve's cluster is decided by its shape. Fitting is EM: the E step
is the posterior class probability of each curve, the M step the
weighted mean and covariance of the coefficients within each class. The
observed-data log likelihood cannot decrease along the way, and the path
is returned so that can be checked rather than believed.

Two details are what make the result reproducible. Initialisation is
deterministic -- curves are ordered along their first principal
direction in coefficient space and cut into K equal groups -- so there
is no random restart to disagree about. And the fitted components are
returned in a canonical order, sorted by the integral of the mean curve,
because a mixture is only identified up to relabelling and an
arbitrary label order is what makes two correct implementations look
like they disagree.

The covariances are diagonal in the coefficient basis. That is a real
restriction -- it says the spline coefficients are uncorrelated within a
class -- and it is what keeps the fit stable when the number of curves
is not much larger than the number of basis functions, which is the
usual functional-data situation.

References
----------
James, G. M. and Sugar, C. A. (2003) "Clustering for sparsely sampled
functional data", *Journal of the American Statistical Association*
**98**(462), 397-408, doi:10.1198/016214503000189.

Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977) "Maximum
likelihood from incomplete data via the EM algorithm", *Journal of the
Royal Statistical Society B* **39**(1), 1-38.

Ramsay, J. O. and Silverman, B. W. (2005) *Functional Data Analysis*,
2nd ed., Springer, Ch. 3 (spline bases for functional data),
doi:10.1007/b98888.

de Boor, C. (1978) *A Practical Guide to Splines*, Springer, Ch. IX (the
Cox-de Boor recursion used for the basis).

Fraley, C. and Raftery, A. E. (2002) "Model-based clustering,
discriminant analysis, and density estimation", *Journal of the American
Statistical Association* **97**(458), 611-631,
doi:10.1198/016214502760047131. The BIC reported here.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["functional_mixture"]

_EPS = 1e-12


def _knots(tmin, tmax, n_basis, degree):
    """Clamped uniform knot vector for ``n_basis`` B-splines."""
    n_int = n_basis - degree - 1
    if n_int < 0:
        raise ValueError("funmix: n_basis must be at least degree + 1")
    inner = [tmin + (tmax - tmin) * (i + 1.0) / (n_int + 1.0)
             for i in range(n_int)]
    return [tmin] * (degree + 1) + inner + [tmax] * (degree + 1)


def _bspline_row(x, kn, degree, n_basis, tmax):
    """One row of the B-spline design, by the Cox-de Boor recursion."""
    m = len(kn) - 1
    N = [0.0] * m
    for i in range(m):
        if kn[i] <= x < kn[i + 1]:
            N[i] = 1.0
    if x >= tmax:                       # close the right end
        for i in range(m - 1, -1, -1):
            if kn[i] < kn[i + 1]:
                N[i] = 1.0
                break
    for d in range(1, degree + 1):
        for i in range(m - d):
            left = 0.0
            den = kn[i + d] - kn[i]
            if den > _EPS:
                left = (x - kn[i]) / den * N[i]
            right = 0.0
            den = kn[i + d + 1] - kn[i + 1]
            if den > _EPS and i + 1 < m:
                right = (kn[i + d + 1] - x) / den * N[i + 1]
            N[i] = left + right
    return N[:n_basis]


def _first_pc(C, p):
    """First principal direction by power iteration -- deterministic start."""
    n = len(C)
    mean = [sum(C[i][a] for i in range(n)) / n for a in range(p)]
    Z = [[C[i][a] - mean[a] for a in range(p)] for i in range(n)]
    S = [[sum(Z[i][a] * Z[i][b] for i in range(n)) / max(n - 1, 1)
          for b in range(p)] for a in range(p)]
    v = [1.0 / math.sqrt(p)] * p
    for _ in range(200):
        u = [sum(S[a][b] * v[b] for b in range(p)) for a in range(p)]
        nu = math.sqrt(sum(t * t for t in u))
        if nu < 1e-300:
            break
        u = [t / nu for t in u]
        if max(abs(u[a] - v[a]) for a in range(p)) < 1e-13:
            v = u
            break
        v = u
    # sign convention: largest-magnitude entry positive, so the projection
    # -- and therefore the initial labelling -- is not sign-arbitrary
    big = 0
    for a in range(p):
        if abs(v[a]) > abs(v[big]):
            big = a
    if v[big] < 0.0:
        v = [-t for t in v]
    return [sum(Z[i][a] * v[a] for a in range(p)) for i in range(n)], v


def functional_mixture(Y, K, t=None, n_basis=5, degree=3, max_iter=300,
                       tol=1e-10, var_floor=1e-8):
    r"""Model-based clustering of curves through a spline basis.

    Parameters
    ----------
    Y : array-like, shape ``(n, m)``
        One curve per row, all on the same grid.
    K : int
        Number of components.
    t : array-like, optional
        The grid. Defaults to ``m`` equally spaced points on ``[0, 1]``.
    n_basis, degree : int
        B-spline basis size and degree.
    var_floor : float
        Lower bound on a component variance, relative to the total
        coefficient variance. Without it a component that collapses onto
        a single curve drives its variance to zero and the likelihood to
        infinity -- the standard degeneracy of unconstrained Gaussian
        mixtures.

    Returns
    -------
    RichResult
        ``labels``, ``posterior``, ``proportions``, the component
        ``mean_curves`` on the grid, ``loglik`` and its path, and ``bic``.
    """
    M = [[float(v) for v in row] for row in k.mat(Y)]
    n = len(M)
    if n == 0:
        raise ValueError("funmix: no curves")
    m = len(M[0])
    if any(len(r) != m for r in M):
        raise ValueError("funmix: every curve must be on the same grid")
    K = int(K)
    if K < 1:
        raise ValueError("funmix: K must be at least 1")
    if K > n:
        raise ValueError("funmix: %d components for %d curves" % (K, n))
    tv = ([i / (m - 1.0) for i in range(m)] if t is None
          else [float(v) for v in k.vec(t)])
    if len(tv) != m:
        raise ValueError("funmix: %d grid points but curves of length %d"
                         % (len(tv), m))
    p = int(n_basis)
    degree = int(degree)
    if p > m:
        raise ValueError("funmix: %d basis functions for %d time points -- "
                         "the coefficient fit is not identified" % (p, m))

    tmin, tmax = min(tv), max(tv)
    if tmax - tmin <= _EPS:
        raise ValueError("funmix: the grid has no extent")
    kn = _knots(tmin, tmax, p, degree)
    B = [_bspline_row(x, kn, degree, p, tmax) for x in tv]

    # coefficients per curve: ridge-stabilised least squares on the basis.
    # The ridge is scaled to the matrix -- a fixed 1e-10 does nothing when
    # the entries are themselves small.
    BtB = [[sum(B[i][a] * B[i][b] for i in range(m)) for b in range(p)]
           for a in range(p)]
    scale = sum(BtB[a][a] for a in range(p)) / p
    for a in range(p):
        BtB[a][a] += 1e-8 * scale
    C = []
    for i in range(n):
        rhs = [sum(B[u][a] * M[i][u] for u in range(m)) for a in range(p)]
        C.append(k.cholsolve(BtB, rhs))

    # deterministic initialisation: cut the first principal score into K
    # equal groups. No random restart, so two implementations start alike.
    score, _pc = _first_pc(C, p)
    order = sorted(range(n), key=lambda i: (score[i], i))
    lab0 = [0] * n
    for rank, i in enumerate(order):
        lab0[i] = min(rank * K // n, K - 1)

    grand = [sum(C[i][a] for i in range(n)) / n for a in range(p)]
    total_var = sum(sum((C[i][a] - grand[a]) ** 2 for i in range(n)) / n
                    for a in range(p)) / p
    floor = max(var_floor * max(total_var, _EPS), 1e-300)

    pi = [0.0] * K
    mu = [[0.0] * p for _ in range(K)]
    sg = [[0.0] * p for _ in range(K)]
    for j in range(K):
        idx = [i for i in range(n) if lab0[i] == j]
        if not idx:
            idx = [order[min(j, n - 1)]]
        pi[j] = len(idx) / float(n)
        for a in range(p):
            mu[j][a] = sum(C[i][a] for i in idx) / len(idx)
            v = sum((C[i][a] - mu[j][a]) ** 2 for i in idx) / len(idx)
            sg[j][a] = max(v, floor)

    def logdens(i, j):
        s = 0.0
        for a in range(p):
            s += (-0.5 * math.log(2.0 * math.pi * sg[j][a])
                  - 0.5 * (C[i][a] - mu[j][a]) ** 2 / sg[j][a])
        return s

    path = []
    post = [[0.0] * K for _ in range(n)]
    ll = -float("inf")
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        ll_new = 0.0
        for i in range(n):
            lp = [math.log(max(pi[j], 1e-300)) + logdens(i, j)
                  for j in range(K)]
            mx = max(lp)
            ssum = sum(math.exp(v - mx) for v in lp)
            ll_new += mx + math.log(ssum)
            for j in range(K):
                post[i][j] = math.exp(lp[j] - mx) / ssum
        path.append(ll_new)
        if it > 1 and abs(ll_new - ll) <= tol * (abs(ll) + 1.0):
            ll = ll_new
            converged = True
            break
        ll = ll_new
        for j in range(K):
            nk = sum(post[i][j] for i in range(n))
            pi[j] = nk / n
            nk = max(nk, 1e-300)
            for a in range(p):
                mu[j][a] = sum(post[i][j] * C[i][a] for i in range(n)) / nk
                v = sum(post[i][j] * (C[i][a] - mu[j][a]) ** 2
                        for i in range(n)) / nk
                sg[j][a] = max(v, floor)

    mean_curves = [[sum(B[u][a] * mu[j][a] for a in range(p))
                    for u in range(m)] for j in range(K)]

    # canonical component order: a mixture is identified only up to
    # relabelling, so sort by the integral of the mean curve (trapezoid).
    def integral(cv):
        return sum(0.5 * (cv[u] + cv[u + 1]) * (tv[u + 1] - tv[u])
                   for u in range(m - 1))

    ordk = sorted(range(K), key=lambda j: (integral(mean_curves[j]), j))
    pi = [pi[j] for j in ordk]
    mu = [mu[j] for j in ordk]
    sg = [sg[j] for j in ordk]
    mean_curves = [mean_curves[j] for j in ordk]
    post = [[post[i][j] for j in ordk] for i in range(n)]
    labels = [max(range(K), key=lambda j: (post[i][j], -j)) for i in range(n)]

    nfree = K - 1 + K * p + K * p
    bic = -2.0 * ll + nfree * math.log(n)
    aic = -2.0 * ll + 2.0 * nfree
    # entropy of the classification: 0 means every curve is assigned with
    # certainty, and a large value means K is doing no work
    ent = -sum(post[i][j] * math.log(max(post[i][j], 1e-300))
               for i in range(n) for j in range(K))

    return RichResult(payload={
        "estimate": labels, "labels": labels, "posterior": post,
        "proportions": pi, "coefficients": mu, "variances": sg,
        "mean_curves": mean_curves, "basis": B, "knots": kn,
        "curve_coefficients": C, "grid": tv,
        "loglik": ll, "loglik_path": path, "bic": bic, "aic": aic,
        "entropy": ent, "n_parameters": nfree,
        "iterations": it, "converged": converged,
        "K": K, "n": n, "n_basis": p, "degree": degree,
        "method": "functional clustering: a K-component Gaussian mixture on "
                  "B-spline coefficients fitted by EM, deterministic "
                  "principal-score initialisation (James & Sugar 2003)",
        "note": "components are returned sorted by the integral of their "
                "mean curve -- a mixture is identified only up to "
                "relabelling, and a canonical order is what makes two "
                "correct fits comparable",
    })


def cheatsheet():
    return ("funmix: functional_mixture(Y, K) -> EM clustering of curves "
            "through a spline basis, canonically ordered components "
            "(James & Sugar 2003, JASA 98:397-408)")
