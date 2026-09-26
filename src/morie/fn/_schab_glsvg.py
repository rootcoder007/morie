"""Matheron lag classes and Cressie's covariance of the Matheron estimator (Schabenberger & Gotway Sec. 4.5.1)."""

import math

from ._schab_vario import semivariogram


def model_gamma(h, nugget, sill, rng, model):
    """gamma(h) at a list of lags, as floats, gamma(0) = 0."""
    return [float(v) for v in semivariogram(list(h), nugget, sill, rng, model)]


def lag_pairs(coords, breaks):
    """Pairs (i, j), i < j, grouped into lag classes (breaks[m], breaks[m+1]]."""
    pts = [tuple(float(v) for v in p) for p in coords]
    br = [float(v) for v in breaks]
    if len(br) < 2 or any(b2 <= b1 for b1, b2 in zip(br, br[1:])):
        raise ValueError("`breaks` must be increasing, at least two values")
    classes = [[] for _ in range(len(br) - 1)]
    dists = [[] for _ in range(len(br) - 1)]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.dist(pts[i], pts[j])
            for m in range(len(br) - 1):
                if br[m] < d <= br[m + 1]:
                    classes[m].append((i, j))
                    dists[m].append(d)
                    break
    keep = [m for m in range(len(classes)) if classes[m]]
    return pts, [classes[m] for m in keep], [sum(dists[m]) / len(dists[m]) for m in keep]


def matheron(z, classes):
    return [sum((z[i] - z[j]) ** 2 for i, j in c) / (2.0 * len(c)) for c in classes]


def pair_gamma(pts, nugget, sill, rng, model):
    n = len(pts)
    g = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row = model_gamma([math.dist(pts[i], pts[j]) for j in range(n)], nugget, sill, rng, model)
        g[i] = row
    return g


def matheron_covariance(pts, classes, nugget, sill, rng, model):
    """Cov[gamma_hat(h_m), gamma_hat(h_n)] for Gaussian data, from eq (4.32).

    Cov[T_ij^2, T_kl^2] = 2 {gamma(s_i - s_l) + gamma(s_j - s_k) - gamma(s_i - s_k)
    - gamma(s_j - s_l)}^2 (Cressie 1993, eq 2.6.10; the book's (4.32) prints
    gamma(h_ij) for the first term), and gamma_hat = sum T^2 / (2 |N|).
    """
    g = pair_gamma(pts, nugget, sill, rng, model)
    k = len(classes)
    R = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(a, k):
            s = 0.0
            for i, j in classes[a]:
                gi, gj = g[i], g[j]
                for kk, ll in classes[b]:
                    c = gi[ll] + gj[kk] - gi[kk] - gj[ll]
                    s += c * c
            R[a][b] = R[b][a] = s / (2.0 * len(classes[a]) * len(classes[b]))
    return R


def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("covariance matrix of the Matheron estimator is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def chol_quad(L, r):
    """r' (L L')^{-1} r by forward substitution."""
    y = []
    for i in range(len(r)):
        y.append((r[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i])
    return sum(v * v for v in y)
