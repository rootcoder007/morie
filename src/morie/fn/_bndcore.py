# morie.fn -- function file (rootcoder007/morie)
"""Shared pieces for the Manski-style partial-identification bound family.

Every worst-case bound in the ``bnd*``/``bns*`` family comes out of the same
law-of-total-probability decomposition of the counterfactual mean.  Writing
the two or three lines once keeps the twenty modules that use them honest
about which equation they are evaluating.

References
----------
Molinari, F. (2021).  Microeconometrics with partial identification.
Handbook of Econometrics, Volume 7A, 355-486.  North Holland.
Working-paper version arXiv:2004.11751, equations (2.11) and (2.13).
"""

from . import _tail1core as C

__all__ = []


def yd(y, D, name):
    """Validate an outcome / binary-treatment pair and return both vectors."""
    yv = C.vec(y)
    dv = C.vec(D)
    if len(yv) == 0:
        raise ValueError(name + ": y is empty")
    if len(dv) != len(yv):
        raise ValueError(name + ": y and D must have the same length")
    for v in dv:
        if v != 0.0 and v != 1.0:
            raise ValueError(name + ": D must be coded 0/1")
    return yv, dv


def cellmeans(yv, dv):
    """Return ``(p1, m1, p0, m0)`` -- share and mean of each treatment arm.

    An empty arm reports mean 0.0; every caller multiplies that mean by the
    arm's share, which is then also 0, so the empty arm contributes nothing.
    """
    n = len(yv)
    n1 = 0
    s1 = 0.0
    s0 = 0.0
    for i in range(n):
        if dv[i] == 1.0:
            n1 += 1
            s1 += yv[i]
        else:
            s0 += yv[i]
    n0 = n - n1
    m1 = s1 / n1 if n1 else 0.0
    m0 = s0 / n0 if n0 else 0.0
    return (n1 / float(n), m1, n0 / float(n), m0)


def wc_arm(m_t, p_t, lo, hi):
    """Worst-case bounds on ``E[y(t)]``, Molinari (2021) eq. (2.11).

    ``E(y | s = t) P(s = t) + y_0 P(s != t)`` and the same with ``y_1``.
    """
    return (m_t * p_t + lo * (1.0 - p_t), m_t * p_t + hi * (1.0 - p_t))


def wc_ate(yv, dv, lo, hi):
    """Worst-case bounds on ``E[y(1)] - E[y(0)]``.

    The sharp lower bound subtracts the upper bound of arm 0 from the lower
    bound of arm 1, and conversely; Molinari (2021) p. 18.
    """
    p1, m1, p0, m0 = cellmeans(yv, dv)
    a1 = wc_arm(m1, p1, lo, hi)
    a0 = wc_arm(m0, p0, lo, hi)
    return (a1[0] - a0[1], a1[1] - a0[0])


def support(v):
    """``(min, max)`` of a non-empty vector."""
    lo = hi = v[0]
    for x in v:
        if x < lo:
            lo = x
        if x > hi:
            hi = x
    return (lo, hi)


def q1(v, p):
    """Type-1 (inverse-CDF) sample quantile: ``min t : F_n(t) >= p``."""
    s = sorted(v)
    m = len(s)
    i = int(-((-(p * m)) // 1))
    if i < 1:
        i = 1
    if i > m:
        i = m
    return s[i - 1]


def cells(x):
    """Distinct values of ``x`` in order of first appearance."""
    out = []
    for v in x:
        if v not in out:
            out.append(v)
    return out


def wc_intersect(yv, dv, cellv, lo, hi):
    """Arm bounds intersected over the cells of ``cellv``.

    Returns ``(lo1, hi1, lo0, hi0)``: the exclusion-restriction
    intersection bounds of Molinari (2021) eq. (2.15) for each arm.
    """
    n = len(yv)
    lo1 = lo0 = None
    hi1 = hi0 = None
    for g in cells(cellv):
        gy = [yv[i] for i in range(n) if cellv[i] == g]
        gd = [dv[i] for i in range(n) if cellv[i] == g]
        p1, m1, p0, m0 = cellmeans(gy, gd)
        a1 = wc_arm(m1, p1, lo, hi)
        a0 = wc_arm(m0, p0, lo, hi)
        if lo1 is None or a1[0] > lo1:
            lo1 = a1[0]
        if hi1 is None or a1[1] < hi1:
            hi1 = a1[1]
        if lo0 is None or a0[0] > lo0:
            lo0 = a0[0]
        if hi0 is None or a0[1] < hi0:
            hi0 = a0[1]
    return (lo1, hi1, lo0, hi0)
