# morie.fn -- internal helper file (rootcoder007/morie)
"""Shared primitives for the Horowitz (2009) inverse-problem and
transformation-model shelf (Chapters 3, 5 and 6).

Internal.  Kept in one place so that the nine ``hrz*`` callables that
use them cannot drift apart, and so the R mirror is a single file.

Reference
---------
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
Econometrics*. Springer.  ISBN 978-0-387-92869-2.
"""

from . import _array_core as np
from . import _s03core as core

SQRT2PI = 2.5066282746310002


def u01(v):
    """Mid-rank transform onto [0, 1].

    Horowitz (2009) p. 156 observes that the support of (X, W) may be
    taken to be [0, 1]^2 with no loss of generality, "because it can
    always be satisfied by, if necessary, carrying out a monotone
    increasing transformation of (X, W)".  The mid-rank map
    (rank - 1/2)/n is such a transformation and is exactly
    reproducible in both language arms, unlike a fitted CDF.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = int(v.size)
    if n == 0:
        raise ValueError("empty input")
    r = core.rank_avg([float(t) for t in v])
    return np.asarray([(float(ri) - 0.5) / n for ri in r], dtype=float)


def grid_w(m):
    """Equispaced grid on [0, 1] with trapezoid quadrature weights."""
    m = int(m)
    if m < 3:
        raise ValueError(f"grid must have at least 3 points, got {m}.")
    step = 1.0 / (m - 1)
    z = np.asarray([i * step for i in range(m)], dtype=float)
    w = [step] * m
    w[0] = step / 2.0
    w[m - 1] = step / 2.0
    return z, np.asarray(w, dtype=float)


def kmat(a, b, h):
    """Gaussian kernel matrix K((a_i - b_j)/h)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    return np.asarray(
        [[np.exp(-0.5 * ((float(ai) - float(bj)) / h) ** 2) / SQRT2PI
          for bj in b] for ai in a], dtype=float)


def ll_smooth(z, y, zq, h):
    """Local-linear regression of y on z evaluated at zq.

    Appendix A.3.  Local linear reproduces an affine function exactly,
    which is what makes an exact-recovery anchor possible for the
    additive fits built on it; Nadaraya-Watson does not.
    """
    z = np.asarray(z, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    zq = np.asarray(zq, dtype=float).ravel()
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if z.size != y.size:
        raise ValueError(f"z has {z.size} points but y has {y.size}.")
    out = []
    for q in zq:
        s0 = s1 = s2 = t0 = t1 = 0.0
        for zi, yi in zip(z, y):
            u = (float(zi) - float(q)) / h
            wt = np.exp(-0.5 * u * u)
            s0 += wt
            s1 += wt * u
            s2 += wt * u * u
            t0 += wt * float(yi)
            t1 += wt * u * float(yi)
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-300:
            out.append(t0 / s0 if s0 > 0 else 0.0)
        else:
            out.append((s2 * t0 - s1 * t1) / det)
    return np.asarray(out, dtype=float)


def wquant(v, w, tau):
    """Weighted tau-quantile.

    The smallest order statistic of ``v`` whose cumulative normalised
    weight reaches ``tau``.  With equal weights this is the usual
    empirical quantile, so it agrees with a plain sort on a degenerate
    kernel -- the anchor used by hrzplrq.
    """
    v = [float(t) for t in np.asarray(v, dtype=float).ravel()]
    w = [float(t) for t in np.asarray(w, dtype=float).ravel()]
    if len(v) != len(w):
        raise ValueError(f"v has {len(v)} points but w has {len(w)}.")
    if not v:
        raise ValueError("empty input")
    tot = 0.0
    for t in w:
        tot += t
    if tot <= 0:
        raise ValueError("weights sum to zero")
    order = sorted(range(len(v)), key=lambda i: v[i])
    acc = 0.0
    for i in order:
        acc += w[i] / tot
        if acc >= float(tau) - 1e-12:
            return float(v[i])
    return float(v[order[len(order) - 1]])


def ade(X, y, h):
    """Density-weighted average derivative, Horowitz Sec. 2.6.1.

    .. math:: \\delta = E[f_X(X)\\,\\partial E(Y|X)/\\partial X]
              = -2\\,E[f_X'(X)\\,Y].

    In a single-index model delta is proportional to beta, so it fixes
    the index DIRECTION without optimising over it.  The leave-one-out
    form is used: the own-observation term of a kernel density
    derivative is identically zero only in the limit, and keeping it
    biases delta toward zero.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    y = np.asarray(y, dtype=float).ravel()
    n, d = int(X.shape[0]), int(X.shape[1])
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size}.")
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    delta = [0.0] * d
    scale = 1.0 / ((n - 1) * h ** (d + 1))
    for i in range(n):
        g = [0.0] * d
        for j in range(n):
            if j == i:
                continue
            prod = 1.0
            us = [0.0] * d
            for k in range(d):
                u = (float(X[i][k]) - float(X[j][k])) / h
                us[k] = u
                prod *= np.exp(-0.5 * u * u) / SQRT2PI
            for k in range(d):
                g[k] += -us[k] * prod
        for k in range(d):
            delta[k] += -2.0 * float(y[i]) * g[k] * scale / n
    return np.asarray(delta, dtype=float)


def index_dir(X, y, h):
    """Index direction with the scale normalisation |beta_1| = 1
    (Horowitz assumption HT2(a), p. 219)."""
    d = ade(X, y, h)
    lead = float(d[0])
    if abs(lead) < 1e-300:
        raise ValueError(
            "the first covariate has a zero average derivative, so the "
            "normalisation |beta_1| = 1 (HT2(a)) is not available.")
    return np.asarray([float(t) / abs(lead) for t in d], dtype=float)


def cheatsheet():
    return "_hrz3: shared grid/kernel/index primitives for the Horowitz shelf"


def bw01(n):
    """Default bandwidth on the mid-rank [0, 1] scale.

    After ``u01`` the marginals are exactly uniform on [0, 1], whose
    standard deviation is 1/sqrt(12), so Silverman's constant gives a
    scale rather than only a rate.  ``n ** (-1/6)`` alone is a rate and
    on the unit interval is far too wide: at n = 40 it puts two thirds
    of the kernel mass outside the support.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    return 1.06 * n ** (-1.0 / 6.0) / (12.0 ** 0.5)


def fxw_grid(u, v, z, wq, h):
    """Bivariate kernel density of (X, W) on the grid, mass-corrected.

    A Gaussian kernel has unbounded support, so on the compact [0, 1]^2
    a fixed share of its mass falls outside and the raw estimate does
    NOT integrate to one.  Horowitz (2009) p. 173 requires a
    compactly-supported kernel (HH5) and notes that boundary effects
    "can be accommodated by replacing the kernel K with a boundary
    kernel".  Renormalising the discretised density to unit mass on
    [0, 1]^2 is such a correction, and it makes ``mass == 1`` an exact
    identity that fails loudly if the 1/(n h^2) constant is mis-wired.

    Returns ``(f, raw_mass)`` where ``f[k][l]`` estimates
    ``f_XW(z_k, z_l)`` and integrates to one against ``wq``.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    n = int(u.size)
    if v.size != n:
        raise ValueError(f"u has {n} points but v has {v.size}.")
    KX = kmat(z, u, h)
    KW = kmat(z, v, h)
    m = int(np.asarray(z).size)
    f = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            s = 0.0
            for i in range(n):
                s += float(KX[k][i]) * float(KW[l][i])
            f[k][l] = s / (n * h * h)
    mass = 0.0
    for k in range(m):
        for l in range(m):
            mass += float(wq[k]) * float(wq[l]) * f[k][l]
    if mass <= 0:
        raise ValueError("the kernel density estimate has non-positive mass.")
    for k in range(m):
        for l in range(m):
            f[k][l] /= mass
    return f, mass


def sieve(z, J, kind="poly"):
    """Series basis on [0, 1], eq. (5.79).

    ``"cos"`` is the orthonormal cosine basis
    :math:`\\{1, \\sqrt2\\cos(\\pi k v)\\}`, for which the coefficients
    in (5.79) are literally the inner products
    :math:`\\beta_j = \\langle g, \\psi_j\\rangle` as the text notes.
    ``"poly"`` is the monomial basis, which spans the same spaces but
    is not orthonormal.
    """
    z = np.asarray(z, dtype=float).ravel()
    J = int(J)
    if J < 1:
        raise ValueError(f"J must be at least 1, got {J}.")
    if kind == "poly":
        return np.asarray([[float(t) ** k for k in range(J)] for t in z],
                          dtype=float)
    if kind == "cos":
        rows = []
        for t in z:
            row = [1.0]
            for k in range(1, J):
                row.append(2.0 ** 0.5 * np.cos(np.pi * k * float(t)))
            rows.append(row)
        return np.asarray(rows, dtype=float)
    raise ValueError(f"kind must be 'poly' or 'cos', got {kind!r}.")
