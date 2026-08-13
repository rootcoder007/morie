"""Shared helpers for the big2 tranche (28 shelves, 168 public functions).

A new private module: nothing here modifies the shared native cores.
Everything is closed form or a fixed-count recurrence -- no RNG, no
tolerance-driven early exit -- so the Python and R arms agree to
machine precision.
"""

from . import _array_core as np

__all__ = []


def logb(v, base=2.0):
    """Elementwise log to ``base``; ``base=None`` means natural log (nats)."""
    out = np.log(np.asarray(v, dtype=float))
    if base is None:
        return out
    b = float(base)
    if b <= 0.0 or b == 1.0:
        raise ValueError("base must be positive and not 1")
    return out / float(np.log(b))


def pnorm(p):
    """Close a non-negative array to unit total mass."""
    p = np.asarray(p, dtype=float)
    if float(np.min(p)) < 0.0:
        raise ValueError("probabilities must be non-negative")
    tot = float(np.sum(p))
    if not (tot > 0.0):
        raise ValueError("total mass must be positive")
    return p / tot


def xlogx(p, base=2.0):
    """``p * log_base(p)`` with the Shannon convention 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    safe = np.where(p > 0.0, p, 1.0)
    return np.where(p > 0.0, p * logb(safe, base), 0.0)


def entropy(p, base=2.0):
    """Shannon entropy of an already-closed pmf (any shape)."""
    return float(-np.sum(xlogx(p, base)))


def kldiv(p, q, base=2.0):
    """Relative entropy D(p||q); +inf where p > 0 = q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ratio = np.where((p > 0.0) & (q > 0.0), np.where(q > 0.0, q, 1.0), 1.0)
    if float(np.sum(np.where((p > 0.0) & (q <= 0.0), 1.0, 0.0))) > 0.0:
        return float("inf")
    terms = np.where(p > 0.0, p * (logb(np.where(p > 0.0, p, 1.0), base) - logb(ratio, base)), 0.0)
    return float(np.sum(terms))


def sumax(a, axis):
    """``np.sum`` over one axis, kept in one place so both arms agree."""
    return np.sum(np.asarray(a, dtype=float), axis=axis)


def dims3(p):
    """Shape of a 3-D joint pmf supplied as nested lists."""
    nx = len(p)
    ny = len(p[0])
    nz = len(p[0][0])
    for b in p:
        if len(b) != ny:
            raise ValueError("ragged 3-D array")
        for c in b:
            if len(c) != nz:
                raise ValueError("ragged 3-D array")
    return nx, ny, nz


def flat3(p):
    """Flatten a 3-D nested-list pmf to a 1-D list, x fastest-outermost."""
    return [float(c) for b in p for a in b for c in a]


def marg3(p, keep):
    """Marginal of a 3-D nested-list pmf onto the axes in ``keep``.

    ``keep`` is a tuple drawn from (0, 1, 2) in increasing order; the
    result is a flat list in row-major order over the kept axes. Kept in
    one place because both arms must sum in the same order.
    """
    nx, ny, nz = dims3(p)
    sizes = [nx, ny, nz]
    out_shape = [sizes[k] for k in keep]
    total = 1
    for s in out_shape:
        total *= s
    out = [0.0] * total
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = 0
                for a in keep:
                    idx = idx * sizes[a] + (i, j, k)[a]
                out[idx] += float(p[i][j][k])
    return out


def dims2(p):
    """Shape of a 2-D joint pmf supplied as nested lists / rows."""
    rows = [list(r) for r in p]
    ny = len(rows[0])
    for r in rows:
        if len(r) != ny:
            raise ValueError("ragged 2-D array")
    return len(rows), ny


def marg2(p, axis):
    """Marginal of a 2-D nested-list pmf: ``axis=0`` keeps rows (X)."""
    rows = [[float(v) for v in r] for r in p]
    if axis == 0:
        return [sum(r) for r in rows]
    return [sum(r[j] for r in rows) for j in range(len(rows[0]))]


def mat(a):
    """Coerce a nested sequence / matrix to a list-of-lists of floats."""
    rows = [[float(v) for v in r] for r in a]
    if not rows:
        raise ValueError("empty matrix")
    nc = len(rows[0])
    if nc == 0:
        raise ValueError("empty matrix")
    for r in rows:
        if len(r) != nc:
            raise ValueError("ragged matrix")
    return rows


def sinkhorn(a, b, C, epsilon, max_iter=200):
    """Sinkhorn scaling with a FIXED iteration count -- no early exit.

    Returns ``(T, u, v, a, b)``.  K = exp(-C/eps) as in Cuturi, M. (2013)
    "Sinkhorn distances: lightspeed computation of optimal transport",
    *Advances in Neural Information Processing Systems* 26, 2292-2300
    Sec. 4.1 with lambda = 1/eps; the alternating updates are
    u <- a / (K v) and v <- b / (K' u), started from v = 1.
    """
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be positive")
    n_it = int(max_iter)
    if n_it < 1:
        raise ValueError("max_iter must be at least 1")
    Cm = mat(C)
    av = [float(v) for v in pnorm(np.atleast_1d(np.asarray(a, dtype=float)))]
    bv = [float(v) for v in pnorm(np.atleast_1d(np.asarray(b, dtype=float)))]
    nr, nc = len(Cm), len(Cm[0])
    if len(av) != nr or len(bv) != nc:
        raise ValueError("marginals do not match the shape of C")
    K = [[float(np.exp(-Cm[i][j] / eps)) for j in range(nc)] for i in range(nr)]
    u = [1.0] * nr
    v = [1.0] * nc
    for _ in range(n_it):
        for i in range(nr):
            s = 0.0
            for j in range(nc):
                s += K[i][j] * v[j]
            u[i] = 0.0 if s <= 0.0 else av[i] / s
        for j in range(nc):
            s = 0.0
            for i in range(nr):
                s += K[i][j] * u[i]
            v[j] = 0.0 if s <= 0.0 else bv[j] / s
    T = [[u[i] * K[i][j] * v[j] for j in range(nc)] for i in range(nr)]
    return T, u, v, av, bv


def margerr(T, a, b):
    """Sup-norm violation of the prescribed marginals."""
    nr, nc = len(T), len(T[0])
    re = max(abs(sum(T[i]) - float(a[i])) for i in range(nr))
    ce = max(abs(sum(T[i][j] for i in range(nr)) - float(b[j])) for j in range(nc))
    return max(re, ce)


def sinkhorn_trace(a, b, C, epsilon, max_iter=200):
    """Marginal violation after each of a FIXED number of scalings."""
    eps = float(epsilon)
    if not (eps > 0.0):
        raise ValueError("epsilon must be positive")
    n_it = int(max_iter)
    Cm = mat(C)
    av = [float(v) for v in pnorm(np.atleast_1d(np.asarray(a, dtype=float)))]
    bv = [float(v) for v in pnorm(np.atleast_1d(np.asarray(b, dtype=float)))]
    nr, nc = len(Cm), len(Cm[0])
    if len(av) != nr or len(bv) != nc:
        raise ValueError("marginals do not match the shape of C")
    K = [[float(np.exp(-Cm[i][j] / eps)) for j in range(nc)] for i in range(nr)]
    u = [1.0] * nr
    v = [1.0] * nc
    trace = []
    for _ in range(n_it):
        for i in range(nr):
            s = 0.0
            for j in range(nc):
                s += K[i][j] * v[j]
            u[i] = 0.0 if s <= 0.0 else av[i] / s
        for j in range(nc):
            s = 0.0
            for i in range(nr):
                s += K[i][j] * u[i]
            v[j] = 0.0 if s <= 0.0 else bv[j] / s
        T = [[u[i] * K[i][j] * v[j] for j in range(nc)] for i in range(nr)]
        trace.append(margerr(T, av, bv))
    return trace


def sekernel(spec=None):
    """Squared-exponential kernel factory.

    ``spec`` is ``(sf, l)`` (default ``(1, 1)``) or a callable
    ``k(x1, x2)``.  Rasmussen, C. E. & Williams, C. K. I. (2006)
    *Gaussian Processes for Machine Learning*, Adaptive Computation and
    Machine Learning, MIT Press, ISBN 0-262-18253-X,
    doi:10.7551/mitpress/3206.001.0001, p. 19:
    ``k(x_p, x_q) = sf^2 exp(-(1/(2 l^2)) |x_p - x_q|^2)``.
    """
    if spec is None:
        sf, ell = 1.0, 1.0
    elif callable(spec):
        return lambda a, b: float(spec(a, b))
    else:
        pars = [float(t) for t in spec]
        if len(pars) != 2:
            raise ValueError("kernel must be (sf, l) or a callable")
        sf, ell = pars
    if not (ell > 0.0):
        raise ValueError("length-scale must be positive")

    def k(a, b):
        s = 0.0
        for i in range(len(a)):
            t = float(a[i]) - float(b[i])
            s += t * t
        return sf * sf * float(np.exp(-s / (2.0 * ell * ell)))

    return k


def _lu(A):
    """Doolittle LU with partial pivoting; returns (LU, piv, sign)."""
    n = len(A)
    M = [[float(v) for v in row] for row in A]
    piv = list(range(n))
    sign = 1.0
    for c in range(n):
        p = c
        best = abs(M[c][c])
        for r in range(c + 1, n):
            if abs(M[r][c]) > best:
                best = abs(M[r][c])
                p = r
        if best == 0.0:
            raise ValueError("matrix is singular")
        if p != c:
            M[c], M[p] = M[p], M[c]
            piv[c], piv[p] = piv[p], piv[c]
            sign = -sign
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r][c] = f
            for j in range(c + 1, n):
                M[r][j] -= f * M[c][j]
    return M, piv, sign


def solve(A, b):
    """Solve ``A x = b`` by LU with partial pivoting."""
    n = len(A)
    if len(b) != n:
        raise ValueError("A and b do not conform")
    M, piv, _ = _lu(A)
    y = [float(b[piv[i]]) for i in range(n)]
    for i in range(1, n):
        s = y[i]
        for j in range(i):
            s -= M[i][j] * y[j]
        y[i] = s
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def slogdet(A):
    """Sign and log |det A| via the same LU."""
    M, _, sign = _lu(A)
    ld = 0.0
    for i in range(len(A)):
        d = M[i][i]
        if d < 0.0:
            sign = -sign
        ld += float(np.log(abs(d)))
    return sign, ld
