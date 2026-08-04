# morie.fn -- k02 batch shared helpers (rootcoder007/morie)
"""Internal helpers shared by the k02 batch.  Not part of the public API.

Mirrors ``r-package/morie/R/k02util.R`` statement for statement so the three
arms (Python, morie R, rmorie) agree to machine precision.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as _st

__all__ = []


def k02fe(y, v):
    """Inverse-variance fixed-effect summary.

    Returns ``(mu, var, sumw, Q, df)`` where ``mu = sum(w y)/sum(w)``,
    ``w = 1/v``, ``var = 1/sum(w)`` and ``Q = sum(w (y - mu)^2)`` is
    Cochran's homogeneity statistic on ``df = k - 1`` degrees of freedom.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = 1.0 / v
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    q = float(np.sum(w * (y - mu) ** 2))
    return mu, 1.0 / sw, sw, q, len(y) - 1


def k02dl(y, v):
    """DerSimonian-Laird moment estimator of tau^2 and the RE summary.

    Returns ``(tau2, mu, var, Q, df)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    mu, _var, sw, q, df = k02fe(y, v)
    w = 1.0 / v
    c = sw - float(np.sum(w * w)) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0.0 else 0.0
    ws = 1.0 / (v + tau2)
    sws = float(np.sum(ws))
    mur = float(np.sum(ws * y)) / sws
    return tau2, mur, 1.0 / sws, q, df


def k02mm(y, v, tau0):
    """Generalised method-of-moments tau^2 at working weights 1/(v + tau0).

    DerSimonian and Kacker (2007) equation (6): with a_i = 1/(v_i + tau0),

        tau2 = [ sum a_i (y_i - ybar_a)^2 - sum a_i v_i + sum a_i^2 v_i / sum a_i ]
               / [ sum a_i - sum a_i^2 / sum a_i ]

    which collapses exactly to DerSimonian-Laird when tau0 = 0.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    a = 1.0 / (v + tau0)
    sa = float(np.sum(a))
    sa2 = float(np.sum(a * a))
    yb = float(np.sum(a * y)) / sa
    num = float(np.sum(a * (y - yb) ** 2)) - float(np.sum(a * v)) + float(np.sum(a * a * v)) / sa
    den = sa - sa2 / sa
    return max(0.0, num / den) if den > 0.0 else 0.0


def k02z(p):
    return float(_st.norm.ppf(p))


def k02tq(p, df):
    return float(_st.t.ppf(p, df))


def k02p2z(z):
    return 2.0 * float(_st.norm.sf(abs(z)))


def k02p2t(tv, df):
    return 2.0 * float(_st.t.sf(abs(tv), df))


def k02pchi(q, df):
    return float(_st.chi2.sf(q, df))


_K02_INVPHI = 0.6180339887498949


def k02gold(f, lo, hi, iters=80):
    """Golden-section minimiser on [lo, hi] with a fixed iteration count."""
    a = float(lo)
    b = float(hi)
    c = b - _K02_INVPHI * (b - a)
    d = a + _K02_INVPHI * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(int(iters)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _K02_INVPHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _K02_INVPHI * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def k02gh(n):
    """Gauss-Hermite nodes and weights for the weight exp(-x^2).

    Newton iteration on the orthonormal Hermite recurrence, the ``gauher``
    routine of Press et al., *Numerical Recipes* section 4.5 -- written out so
    that the Python and R arms execute the same iteration and agree to the
    last bit.
    """
    n = int(n)
    pim4 = 0.7511255444649425
    eps = 3.0e-14
    x = [0.0] * n
    w = [0.0] * n
    z = 0.0
    pp = 0.0
    m = (n + 1) // 2
    for i in range(1, m + 1):
        if i == 1:
            z = float(np.sqrt(2.0 * n + 1.0)) - 1.85575 * (2.0 * n + 1.0) ** (-0.16667)
        elif i == 2:
            z = z - 1.14 * n**0.426 / z
        elif i == 3:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 4:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2.0 * z - x[i - 3]
        for _ in range(20):
            p1 = pim4
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = z * float(np.sqrt(2.0 / j)) * p2 - float(np.sqrt((j - 1.0) / j)) * p3
            pp = float(np.sqrt(2.0 * n)) * p2
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= eps:
                break
        x[i - 1] = z
        x[n - i] = -z
        w[i - 1] = 2.0 / (pp * pp)
        w[n - i] = w[i - 1]
    return x, w


def k02mod(A, comm):
    """Newman-Girvan modularity of a partition of a weighted undirected graph."""
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    k = [float(t) for t in np.sum(a, axis=1)]
    m2 = float(np.sum(a))
    if m2 <= 0.0:
        return 0.0
    q = 0.0
    for i in range(n):
        for j in range(n):
            if comm[i] == comm[j]:
                q += float(a[i, j]) - k[i] * k[j] / m2
    return q / m2


def k02bfs(A):
    """All-pairs shortest-path lengths by breadth-first search (hop counts)."""
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    nbr = [[j for j in range(n) if a[i, j] != 0.0 and j != i] for i in range(n)]
    out = []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in nbr[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        out.append(dist)
    return out
