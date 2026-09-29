# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels of the spatial machine-learning front ends (``zx*``).

Spatial features (coordinates and a quadratic trend surface), least
squares, CART regression trees, random forests and bagging on Philox
draws, gradient boosting, least-squares SVM, lasso / elastic net by
coordinate descent, spatial block, buffered and leave-one-out
cross-validation and non-negative least squares for stacking. R twin:
``R/SpatialLearning.R`` (helpers ``.sml_*``).
"""

from __future__ import annotations

import math

from ._qpcore import solve
from ._rng import random_uniform

__all__: list = []


def vec(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    return [float(t) for t in v]


def mat(A, ncol=None):
    A = A.tolist() if hasattr(A, "tolist") else A
    rows = [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in A]
    return rows


def features(X, coords, trend=1):
    """Rows ``[x..., s1, s2]`` (trend 1) or with ``s1^2, s1 s2, s2^2`` (trend 2)."""
    C = mat(coords)
    n = len(C)
    Xm = [[] for _ in range(n)] if X is None else mat(X)
    if len(Xm) != n:
        raise ValueError("X and coords must have the same number of rows")
    out = []
    for x, c in zip(Xm, C):
        s1, s2 = c[0], (c[1] if len(c) > 1 else 0.0)
        row = list(x) + [s1, s2]
        if trend == 2:
            row += [s1 * s1, s1 * s2, s2 * s2]
        out.append(row)
    return out


def ols_fit(Z, y):
    """Least squares with an intercept: returns ``[b0, b...]``."""
    D = [[1.0] + r for r in Z]
    p = len(D[0])
    A = [[math.fsum(r[a] * r[b] for r in D) for b in range(p)] for a in range(p)]
    return [float(v) for v in solve(A, [math.fsum(r[a] * t for r, t in zip(D, y)) for a in range(p)])]


def ols_predict(b, Z):
    return [b[0] + math.fsum(u * v for u, v in zip(b[1:], r)) for r in Z]


class Stream:
    """Philox uniforms in blocks of 1024 (stream index = block), consumed in order."""

    def __init__(self, seed):
        self.seed, self.block, self.buf, self.pos = seed, 0, [], 0

    def next(self):
        if self.pos >= len(self.buf):
            u = random_uniform(1024, seed=self.seed, stream=self.block)
            self.buf = [float(v) for v in (u.tolist() if hasattr(u, "tolist") else u)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def ssum(v):
    """Left-to-right double sum (the R twin loops the same way, so tree splits agree exactly)."""
    s = 0.0
    for a in v:
        s += a
    return s


def _best_split(Z, y, idx, feats, min_leaf):
    best = None
    n = len(idx)
    for f in feats:
        order = sorted(idx, key=lambda i: (Z[i][f], i))
        vals = [Z[i][f] for i in order]
        ys = [y[i] for i in order]
        tot, tot2 = ssum(ys), ssum(v * v for v in ys)
        sl = sl2 = 0.0
        for k in range(1, n):
            sl += ys[k - 1]
            sl2 += ys[k - 1] * ys[k - 1]
            if k < min_leaf or n - k < min_leaf or not vals[k - 1] < vals[k]:
                continue
            sr, sr2 = tot - sl, tot2 - sl2
            sse = (sl2 - sl * sl / k) + (sr2 - sr * sr / (n - k))
            if best is None or sse < best[0]:
                best = (sse, f, 0.5 * (vals[k - 1] + vals[k]), order[:k], order[k:])
    return best


def tree_fit(Z, y, idx, max_depth, min_leaf, mtry, stream, depth=0):
    """CART regression tree: exhaustive SSE splits on ``mtry`` Philox-chosen features per node."""
    ys = [y[i] for i in idx]
    mean = ssum(ys) / len(ys)
    if (max_depth is not None and depth >= max_depth) or len(idx) < 2 * min_leaf or max(ys) == min(ys):
        return ("leaf", mean)
    p = len(Z[0])
    if mtry is None or mtry >= p:
        feats = list(range(p))
    else:
        u = [stream.next() for _ in range(p)]
        feats = sorted(sorted(range(p), key=lambda j: (u[j], j))[:mtry])
    b = _best_split(Z, y, idx, feats, min_leaf)
    if b is None:
        return ("leaf", mean)
    _sse, f, thr, left, right = b
    return (
        f,
        thr,
        tree_fit(Z, y, left, max_depth, min_leaf, mtry, stream, depth + 1),
        tree_fit(Z, y, right, max_depth, min_leaf, mtry, stream, depth + 1),
    )


def tree_predict(t, z):
    while t[0] != "leaf":
        t = t[2] if z[t[0]] <= t[1] else t[3]
    return t[1]


def forest(Z, y, n_trees, mtry, min_leaf, max_depth, seed):
    """Random forest (Breiman 2001): bootstrap rows and draw ``mtry`` features per node from one Philox stream."""
    n = len(y)
    st = Stream(seed)
    trees, oob_sum, oob_cnt = [], [0.0] * n, [0] * n
    for _ in range(n_trees):
        boot = [min(int(st.next() * n), n - 1) for _ in range(n)]
        t = tree_fit(Z, y, boot, max_depth, min_leaf, mtry, st)
        trees.append(t)
        inb = set(boot)
        for i in range(n):
            if i not in inb:
                oob_sum[i] += tree_predict(t, Z[i])
                oob_cnt[i] += 1
    oob = [s / c if c else math.nan for s, c in zip(oob_sum, oob_cnt)]
    return trees, oob


def forest_predict(trees, Z):
    return [ssum(tree_predict(t, z) for t in trees) / len(trees) for z in Z]


def boost(Z, y, n_trees, rate, max_depth, min_leaf):
    """Gradient boosting with squared loss (Friedman 2001): trees fitted to the residuals, shrunk by ``rate``."""
    n = len(y)
    f0 = ssum(y) / n
    F = [f0] * n
    trees = []
    for _ in range(n_trees):
        r = [a - b for a, b in zip(y, F)]
        t = tree_fit(Z, r, list(range(n)), max_depth, min_leaf, None, None)
        trees.append(t)
        F = [a + rate * tree_predict(t, z) for a, z in zip(F, Z)]
    return f0, trees, F


def boost_predict(f0, trees, rate, Z):
    return [f0 + rate * ssum(tree_predict(t, z) for t in trees) for z in Z]


def standardise(Z):
    p = len(Z[0])
    cols = [[r[j] for r in Z] for j in range(p)]
    mu = [math.fsum(c) / len(c) for c in cols]
    sd = [math.sqrt(math.fsum((v - m) ** 2 for v in c) / len(c)) for c, m in zip(cols, mu)]
    sd = [s if s > 0 else 1.0 for s in sd]
    return mu, sd, [[(v - m) / s for v, m, s in zip(r, mu, sd)] for r in Z]


def lssvm(Z, y, gamma, C):
    """Least-squares SVM regression (Suykens and Vandewalle 1999) with an RBF kernel on standardised features."""
    mu, sd, S = standardise(Z)
    n = len(y)
    K = [[math.exp(-gamma * math.fsum((a - b) ** 2 for a, b in zip(S[i], S[j]))) for j in range(n)] for i in range(n)]
    A = [[0.0] + [1.0] * n] + [[1.0] + [K[i][j] + (1.0 / C if i == j else 0.0) for j in range(n)] for i in range(n)]
    sol = solve(A, [0.0] + list(y))
    return mu, sd, S, float(sol[0]), [float(v) for v in sol[1:]]


def lssvm_predict(model, gamma, Z):
    mu, sd, S, b, a = model
    out = []
    for r in Z:
        s = [(v - m) / q for v, m, q in zip(r, mu, sd)]
        out.append(
            b
            + math.fsum(ai * math.exp(-gamma * math.fsum((u - w) ** 2 for u, w in zip(s, sj))) for ai, sj in zip(a, S))
        )
    return out


def coord_descent(Z, y, lam, alpha, tol=1e-12, max_iter=100000):
    """Elastic net ``1/(2n) ||y - b0 - Zb||^2 + lam (alpha |b|_1 + (1 - alpha) |b|^2 / 2)`` on standardised columns (glmnet)."""
    mu, sd, S = standardise(Z)
    n, p = len(y), len(S[0])
    ybar = math.fsum(y) / n
    r = [v - ybar for v in y]
    b = [0.0] * p
    for _ in range(max_iter):
        dmax = 0.0
        for j in range(p):
            col = [row[j] for row in S]
            rho = math.fsum(c * (ri + c * b[j]) for c, ri in zip(col, r)) / n
            z = math.copysign(max(abs(rho) - lam * alpha, 0.0), rho) / (1.0 + lam * (1.0 - alpha))
            d = z - b[j]
            if d != 0.0:
                r = [ri - d * c for ri, c in zip(r, col)]
                b[j] = z
                dmax = max(dmax, abs(d))
        if dmax < tol:
            break
    beta = [bj / s for bj, s in zip(b, sd)]
    b0 = ybar - math.fsum(bj * m for bj, m in zip(beta, mu))
    return [b0] + beta


def blocks(coords, n_blocks, k, seed):
    """Spatial block folds (Roberts et al. 2017): ``n_blocks x n_blocks`` grid over the bounding box, blocks
    randomly ordered by Philox uniforms and dealt to the ``k`` folds in turn."""
    C = mat(coords)
    xs, ys = [c[0] for c in C], [c[1] if len(c) > 1 else 0.0 for c in C]
    x0, y0 = min(xs), min(ys)
    wx = (max(xs) - x0) / n_blocks or 1.0
    wy = (max(ys) - y0) / n_blocks or 1.0
    cell = [
        min(int((a - x0) / wx), n_blocks - 1) * n_blocks + min(int((b - y0) / wy), n_blocks - 1) for a, b in zip(xs, ys)
    ]
    used = sorted(set(cell))
    u = random_uniform(len(used), seed=seed)
    u = [float(v) for v in (u.tolist() if hasattr(u, "tolist") else u)]
    order = sorted(range(len(used)), key=lambda j: (u[j], j))
    fold_of_block = {used[j]: r % k for r, j in enumerate(order)}
    return [fold_of_block[c] for c in cell]


def rmse(a, b):
    return math.sqrt(math.fsum((u - v) ** 2 for u, v in zip(a, b)) / len(a))


def nnls(A, b, max_iter=500):
    """Lawson and Hanson (1974) active-set non-negative least squares."""
    m, n = len(A), len(A[0])
    x = [0.0] * n
    P = []
    At = [[A[i][j] for i in range(m)] for j in range(n)]

    def grad(x):
        r = [b[i] - math.fsum(A[i][j] * x[j] for j in range(n)) for i in range(m)]
        return [math.fsum(At[j][i] * r[i] for i in range(m)) for j in range(n)]

    for _ in range(max_iter):
        w = grad(x)
        cand = [j for j in range(n) if j not in P and w[j] > 1e-12]
        if not cand:
            break
        P.append(max(cand, key=lambda j: (w[j], -j)))
        while True:
            Ap = [[A[i][j] for j in P] for i in range(m)]
            G = [[math.fsum(Ap[i][a] * Ap[i][c] for i in range(m)) for c in range(len(P))] for a in range(len(P))]
            z = [float(v) for v in solve(G, [math.fsum(Ap[i][a] * b[i] for i in range(m)) for a in range(len(P))])]
            if min(z) > 0:
                for j in range(n):
                    x[j] = 0.0
                for a, j in enumerate(P):
                    x[j] = z[a]
                break
            neg = [a for a in range(len(P)) if z[a] <= 0]
            t = min(x[P[a]] / (x[P[a]] - z[a]) for a in neg)
            for a, j in enumerate(P):
                x[j] = x[j] + t * (z[a] - x[j])
            P = [j for j in P if x[j] > 1e-12]
            for j in range(n):
                if j not in P:
                    x[j] = 0.0
            if not P:
                break
    return x
