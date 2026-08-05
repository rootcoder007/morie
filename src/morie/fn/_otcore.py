# morie.fn -- function file (rootcoder007/morie)
"""Deterministic kernels shared by the optimal-transport modules.

Written against Peyre, G. and Cuturi, M. (2019), Computational Optimal
Transport, Foundations and Trends in Machine Learning 11(5-6):355-607.
Equation numbers quoted in the individual modules refer to that text; the
copy consulted was arXiv:1803.00567v4.

Nothing in here draws a random number.  Sliced methods take their
directions from the van der Corput / AS 241 stream in ``_s03core`` so the
Python and R arms land on the same projections rather than merely the
same distribution.
"""

import math

from . import _s03core as core

__all__ = [
    "hist", "costmat", "sinkhorn", "sinkhorn_unbalanced", "emd",
    "partial_plan", "sqrtm_sym", "w2gauss", "wp1d", "quantiles",
    "directions", "project", "frob", "kl", "gw_cost",
]

_NEG_INF = float("-inf")


def _log(x):
    return math.log(x) if x > 0.0 else _NEG_INF


def hist(a, normalise=False):
    """A weight vector as a plain list of floats."""
    v = [float(t) for t in core.vec(a)]
    if any(t < 0.0 for t in v):
        raise ValueError("weights must be non-negative")
    if normalise:
        s = sum(v)
        if s <= 0.0:
            raise ValueError("weights must have positive total mass")
        v = [t / s for t in v]
    return v


def costmat(X, Y, p=2):
    """Ground cost ``C_ij = ||x_i - y_j||^p`` for two point clouds."""
    A = core.mat(X)
    B = core.mat(Y)
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("point clouds must share a dimension")
    out = []
    for xi in A:
        row = []
        for yj in B:
            s = math.sqrt(sum((xi[k] - yj[k]) ** 2 for k in range(d)))
            row.append(s ** p)
        out.append(row)
    return out


def frob(T, C):
    return sum(T[i][j] * C[i][j] for i in range(len(T)) for j in range(len(T[0])))


def kl(T, R):
    """KL(T|R) as in (4.6): sum T log(T/R) - T + R."""
    tot = 0.0
    for i in range(len(T)):
        for j in range(len(T[0])):
            t = T[i][j]
            r = R[i][j]
            if t > 0.0:
                tot += t * (_log(t) - _log(r))
            tot += r - t
    return tot


def sinkhorn(a, b, C, eps, n_iter=200):
    """Log-domain Sinkhorn, (4.15).  Returns ``(T, f, g)``.

    A fixed iteration count, never a tolerance test, so both arms stop in
    the same place.  ``f`` and ``g`` are the eps-scaled dual potentials,
    ``T_ij = exp((f_i + g_j - C_ij)/eps)``.
    """
    n, m = len(a), len(b)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    la = [_log(t) for t in a]
    lb = [_log(t) for t in b]
    f = [0.0] * n
    g = [0.0] * m
    for _ in range(int(n_iter)):
        for i in range(n):
            if la[i] == _NEG_INF:
                f[i] = _NEG_INF
                continue
            f[i] = eps * (la[i] - core.logsumexp(
                [(g[j] - C[i][j]) / eps for j in range(m)]))
        for j in range(m):
            if lb[j] == _NEG_INF:
                g[j] = _NEG_INF
                continue
            g[j] = eps * (lb[j] - core.logsumexp(
                [(f[i] - C[i][j]) / eps for i in range(n)]))
    T = []
    for i in range(n):
        row = []
        for j in range(m):
            z = f[i] + g[j] - C[i][j]
            row.append(math.exp(z / eps) if z > _NEG_INF else 0.0)
        T.append(row)
    return T, f, g


def sinkhorn_unbalanced(a, b, C, eps, lam, n_iter=200):
    """Scaling iterations (10.9) with tau1 = tau2 = lam."""
    n, m = len(a), len(b)
    if eps <= 0.0 or lam <= 0.0:
        raise ValueError("epsilon and lambda must be positive")
    pw = lam / (lam + eps)
    K = [[math.exp(-C[i][j] / eps) for j in range(m)] for i in range(n)]
    u = [1.0] * n
    v = [1.0] * m
    for _ in range(int(n_iter)):
        for i in range(n):
            s = sum(K[i][j] * v[j] for j in range(m))
            u[i] = (a[i] / s) ** pw if s > 0.0 else 0.0
        for j in range(m):
            s = sum(K[i][j] * u[i] for i in range(n))
            v[j] = (b[j] / s) ** pw if s > 0.0 else 0.0
    T = [[u[i] * K[i][j] * v[j] for j in range(m)] for i in range(n)]
    return T


# ------------------------------------------------------- exact transport


def _nwcorner(a, b):
    n, m = len(a), len(b)
    ra = list(a)
    rb = list(b)
    T = [[0.0] * m for _ in range(n)]
    basis = []
    i = j = 0
    while True:
        t = ra[i] if ra[i] < rb[j] else rb[j]
        T[i][j] = t
        basis.append((i, j))
        ra[i] -= t
        rb[j] -= t
        if i == n - 1 and j == m - 1:
            break
        if ra[i] <= 1e-15 and i < n - 1:
            i += 1
        elif j < m - 1:
            j += 1
        else:
            i += 1
    return T, basis


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _complete_tree(basis, n, m):
    parent = list(range(n + m))
    edges = []
    for (i, j) in basis:
        ri, rj = _find(parent, i), _find(parent, n + j)
        if ri != rj:
            parent[ri] = rj
            edges.append((i, j))
    have = set(edges)
    for i in range(n):
        for j in range(m):
            if len(edges) >= n + m - 1:
                break
            if (i, j) in have:
                continue
            ri, rj = _find(parent, i), _find(parent, n + j)
            if ri != rj:
                parent[ri] = rj
                edges.append((i, j))
                have.add((i, j))
    return sorted(edges)


def _adj(basis, n):
    adj = {}
    for (i, j) in basis:
        adj.setdefault(i, []).append((n + j, i, j))
        adj.setdefault(n + j, []).append((i, i, j))
    return adj


def _potentials(basis, C, n, m):
    adj = _adj(basis, n)
    u = [0.0] * n
    v = [0.0] * m
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for (nb, i, j) in adj.get(node, []):
            if nb in seen:
                continue
            seen.add(nb)
            if nb >= n:
                v[nb - n] = C[i][j] - u[i]
            else:
                u[nb] = C[i][j] - v[j]
            stack.append(nb)
    return u, v


def _tree_path(basis, n, si, sj):
    adj = _adj(basis, n)
    goal = n + sj
    stack = [(si, [], frozenset([si]))]
    while stack:
        node, path, seen = stack.pop()
        if node == goal:
            return path
        for (nb, i, j) in adj.get(node, []):
            if nb not in seen:
                stack.append((nb, path + [(i, j)], seen | {nb}))
    return None


def emd(a, b, C, max_pivots=20000):
    """Exact discrete optimal transport by the transportation simplex.

    ponytail: Dantzig entering rule with a pivot cap rather than Bland's
    rule.  Degenerate ties could in principle cycle; the cap turns that
    into a loud error instead of a hang.  Swap in Bland's rule if a real
    instance ever trips it.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("emd: empty marginal")
    if len(C) != n or len(C[0]) != m:
        raise ValueError("emd: cost matrix does not match the marginals")
    sa, sb = sum(a), sum(b)
    if abs(sa - sb) > 1e-9 * max(1.0, abs(sa)):
        raise ValueError("emd: marginals must have equal total mass")
    T, basis = _nwcorner(a, b)
    basis = _complete_tree(basis, n, m)
    for _ in range(int(max_pivots)):
        bset = set(basis)
        u, v = _potentials(basis, C, n, m)
        best = None
        for i in range(n):
            for j in range(m):
                if (i, j) in bset:
                    continue
                d = C[i][j] - u[i] - v[j]
                if d < -1e-11 and (best is None or d < best[0] - 1e-15):
                    best = (d, i, j)
        if best is None:
            break
        _, si, sj = best
        path = _tree_path(basis, n, si, sj)
        minus = path[0::2]
        theta = min(T[i][j] for (i, j) in minus)
        leave = None
        for (i, j) in minus:
            if T[i][j] <= theta + 1e-15:
                leave = (i, j)
                break
        T[si][sj] += theta
        sign = -1.0
        for (i, j) in path:
            T[i][j] += sign * theta
            sign = -sign
        basis = sorted([e for e in basis if e != leave] + [(si, sj)])
    else:
        raise RuntimeError("emd: pivot cap reached")
    return T, frob(T, C)


def partial_plan(a, b, C, m):
    """Partial transport of exactly ``m`` units, via dummy row and column.

    Padding the cost with a zero-price dummy row (supply ``|b| - m``) and
    dummy column (demand ``|a| - m``) turns the inequality-constrained
    partial problem into an ordinary balanced one, so it is solved exactly
    rather than relaxed.  Caffarelli & McCann (2010).
    """
    n, k = len(a), len(b)
    sa, sb = sum(a), sum(b)
    m = float(m)
    if m < 0.0 or m > min(sa, sb) + 1e-12:
        raise ValueError("the transported mass must lie in [0, min(|a|,|b|)]")
    big = 2.0 * max(C[i][j] for i in range(n) for j in range(k)) + 1.0
    sup = list(a) + [sb - m]
    dem = list(b) + [sa - m]
    Ce = [[C[i][j] for j in range(k)] + [0.0] for i in range(n)]
    Ce.append([0.0] * k + [big])
    T, _ = emd(sup, dem, Ce)
    P = [[T[i][j] for j in range(k)] for i in range(n)]
    return P, frob(P, C)


def gw_cost(Cx, Cy, T, a, b):
    """Gromov objective and the linearising product ``Cx T Cy``.

    ``sum_ijkl (Cx_ik - Cy_jl)^2 T_ij T_kl`` expands into two marginal
    constants plus ``-2 <Cx T Cy, T>``; the constants do not depend on
    ``T`` once the marginals are fixed.
    """
    n, m = len(a), len(b)
    t1 = sum(Cx[i][k] ** 2 * a[i] * a[k] for i in range(n) for k in range(n))
    t3 = sum(Cy[j][l] ** 2 * b[j] * b[l] for j in range(m) for l in range(m))
    CT = [[sum(Cx[i][k] * T[k][l] * Cy[l][j] for k in range(n)
               for l in range(m)) for j in range(m)] for i in range(n)]
    val = t1 + t3 - 2.0 * sum(CT[i][j] * T[i][j]
                              for i in range(n) for j in range(m))
    return val, CT


# --------------------------------------------------------------- Gaussian


def sqrtm_sym(S):
    """Symmetric positive semi-definite square root via sign-fixed Jacobi."""
    vals, vecs = core.jacobi(S)
    n = len(vals)
    r = [math.sqrt(t) if t > 0.0 else 0.0 for t in vals]
    return [[sum(vecs[i][k] * r[k] * vecs[j][k] for k in range(n))
             for j in range(n)] for i in range(n)]


def w2gauss(m1, S1, m2, S2):
    """Squared 2-Wasserstein between Gaussians, (2.41) with Bures (2.42)."""
    a = [float(t) for t in core.vec(m1)]
    b = [float(t) for t in core.vec(m2)]
    A = core.mat(S1)
    B = core.mat(S2)
    d = len(a)
    if len(b) != d or len(A) != d or len(B) != d:
        raise ValueError("w2gauss: dimension mismatch")
    R = sqrtm_sym(A)
    M = [[sum(R[i][k] * B[k][l] * R[l][j] for k in range(d) for l in range(d))
          for j in range(d)] for i in range(d)]
    Msq = sqrtm_sym(M)
    bures = sum(A[i][i] + B[i][i] - 2.0 * Msq[i][i] for i in range(d))
    if bures < 0.0:
        bures = 0.0
    return sum((a[i] - b[i]) ** 2 for i in range(d)) + bures


# ------------------------------------------------------------ 1-D and slices


def wp1d(x, y, p=2):
    """Univariate ``W_p`` for two equal-size uniform samples."""
    xs = sorted(float(t) for t in core.vec(x))
    ys = sorted(float(t) for t in core.vec(y))
    if len(xs) != len(ys):
        raise ValueError("wp1d: samples must have equal length")
    if not xs:
        raise ValueError("wp1d: empty sample")
    if p <= 0:
        raise ValueError("wp1d: p must be positive")
    s = sum(abs(xs[i] - ys[i]) ** p for i in range(len(xs))) / len(xs)
    return s ** (1.0 / p)


def quantiles(x, grid):
    """Type-7 quantiles of ``x`` at the probabilities in ``grid``."""
    v = sorted(float(t) for t in core.vec(x))
    return [core.quantile7(v, q) for q in grid]


def directions(d, n_proj):
    """``n_proj`` deterministic unit directions in R^d.

    Van der Corput driven normals, row-major, each row scaled to unit
    length; in one dimension the single direction is +1.
    """
    if d < 1 or n_proj < 1:
        raise ValueError("directions: d and n_proj must be positive")
    if d == 1:
        return [[1.0] for _ in range(n_proj)]
    z = core.normdraws(d * int(n_proj))
    out = []
    for k in range(int(n_proj)):
        row = z[k * d:(k + 1) * d]
        nrm = math.sqrt(sum(t * t for t in row))
        if nrm <= 0.0:
            row = [1.0] + [0.0] * (d - 1)
            nrm = 1.0
        out.append([t / nrm for t in row])
    return out


def project(X, theta):
    return [sum(row[k] * theta[k] for k in range(len(theta))) for row in X]
