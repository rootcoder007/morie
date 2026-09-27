"""Shared pieces for the quasi-Newton minimisers: numerical gradient and a strong-Wolfe line search.

Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, 2nd ed., Algorithms 3.5 and 3.6.
"""

import math


def num_grad(f, x):
    """Central differences with h_i = 1e-6 max(1, |x_i|)."""
    g = []
    for i in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[i]))
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


def dot(a, b):
    s = 0.0
    for u, v in zip(a, b):
        s += u * v
    return s


def _cubic_min(a, fa, da, b, fb, db):
    # minimiser of the cubic interpolating (a, fa, da), (b, fb, db) (N&W eq. 3.59); bisection if undefined
    d1 = da + db - 3 * (fa - fb) / (a - b)
    rad = d1 * d1 - da * db
    if rad < 0:
        return 0.5 * (a + b)
    d2 = math.copysign(math.sqrt(rad), b - a)
    den = db - da + 2 * d2
    if den == 0:
        return 0.5 * (a + b)
    t = b - (b - a) * (db + d2 - d1) / den
    lo, hi = min(a, b), max(a, b)
    if not (lo + 0.1 * (hi - lo) <= t <= hi - 0.1 * (hi - lo)):
        return 0.5 * (a + b)
    return t


def wolfe(phi, a1=1.0, a_max=None, c1=1e-4, c2=0.9, max_eval=40):
    """Strong-Wolfe step: phi(a) -> (value, slope). Returns (a, value, slope, evaluations)."""
    f0, d0 = phi(0.0)
    if d0 >= 0:
        return 0.0, f0, d0, 1
    a_prev, f_prev, d_prev = 0.0, f0, d0
    a = a1 if a_max is None else min(a1, a_max)
    n = 1

    def zoom(lo, flo, dlo, hi, fhi, dhi, n):
        while n < max_eval:
            a = _cubic_min(lo, flo, dlo, hi, fhi, dhi)
            fa, da = phi(a)
            n += 1
            if fa > f0 + c1 * a * d0 or fa >= flo:
                hi, fhi, dhi = a, fa, da
            else:
                if abs(da) <= -c2 * d0:
                    return a, fa, da, n
                if da * (hi - lo) >= 0:
                    hi, fhi, dhi = lo, flo, dlo
                lo, flo, dlo = a, fa, da
            if abs(hi - lo) < 1e-16 * max(1.0, abs(lo)):
                break
        return lo, flo, dlo, n

    while n < max_eval:
        fa, da = phi(a)
        n += 1
        if fa > f0 + c1 * a * d0 or (n > 2 and fa >= f_prev):
            return zoom(a_prev, f_prev, d_prev, a, fa, da, n)
        if abs(da) <= -c2 * d0:
            return a, fa, da, n
        if da >= 0:
            return zoom(a, fa, da, a_prev, f_prev, d_prev, n)
        if a_max is not None and a >= a_max:
            return a, fa, da, n
        a_prev, f_prev, d_prev = a, fa, da
        a = 2 * a if a_max is None else min(2 * a, a_max)
    return a_prev, f_prev, d_prev, n
