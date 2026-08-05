# morie.fn -- function file (rootcoder007/morie)
"""Moment-inequality criterion shared by the interval-identification modules.

The criterion is Molinari (2021) equation (4.2) with the two inequalities
of an interval-identified scalar, ``E[yL] - theta <= 0`` and
``theta - E[yU] <= 0``.  Only violated inequalities contribute, so the
criterion is exactly zero on ``[E yL, E yU]`` and positive off it, which
is equation (4.4).
"""

from . import _tail1core as C

__all__ = []


def interval_data(moments, name):
    """Split an ``(n, 2)`` matrix into ``(yL, yU)`` with ``yL <= yU``."""
    M = C.mat(moments)
    n = len(M)
    if n < 2:
        raise ValueError(name + ": need at least two observations")
    if len(M[0]) != 2:
        raise ValueError(name + ": moments must have two columns, yL and yU")
    yl = [r[0] for r in M]
    yu = [r[1] for r in M]
    for i in range(n):
        if yu[i] < yl[i]:
            raise ValueError(name + ": yU is below yL at some observation")
    return yl, yu


def stats(yl, yu):
    """``(n, mL, sL, mU, sU)`` with sample standard deviations (ddof 1)."""
    n = len(yl)
    mL = C.mean(yl)
    mU = C.mean(yu)
    sL = C.sd(yl)
    sU = C.sd(yu)
    if sL <= 0.0:
        sL = 1e-12
    if sU <= 0.0:
        sU = 1e-12
    return (n, mL, sL, mU, sU)


def crit(theta, n, mL, sL, mU, sU):
    """``Q_n(theta)``, the sum of squared positive parts, eq. (4.2)."""
    rn = n ** 0.5
    a = rn * (mL - theta) / sL
    b = rn * (theta - mU) / sU
    if a < 0.0:
        a = 0.0
    if b < 0.0:
        b = 0.0
    return a * a + b * b


def critmax(theta, n, mL, sL, mU, sU):
    """``q_max(theta)``, the max form of eq. (4.3)."""
    rn = n ** 0.5
    a = rn * (mL - theta) / sL
    b = rn * (theta - mU) / sU
    m = a if a > b else b
    return m if m > 0.0 else 0.0
