"""Shared helpers for the closed-form distribution modules: vectorisation, bisection quantiles, Philox draws."""

import math

from ._rng import random_uniform


def vec(x):
    if isinstance(x, (int, float)):
        return [float(x)], True
    return [float(v) for v in x], False


def out(vals, scalar):
    return vals[0] if scalar else vals


def bisect_q(cdf, p, lo, hi, tol=1e-14):
    """Quantile by bisection on a continuous increasing cdf, expanding infinite brackets."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a, b = lo, hi
    if math.isinf(a):
        a = -1.0
        while cdf(a) > p:
            a *= 2
    if math.isinf(b):
        b = 1.0 if a < 1 else 2 * a
        while cdf(b) < p:
            b = b * 2 if b > 0 else b + 1
    for _ in range(300):
        m = (a + b) / 2
        if cdf(m) < p:
            a = m
        else:
            b = m
        if b - a <= tol * (1 + abs(m)):
            break
    return (a + b) / 2


def draws(quantile, n, seed):
    if n <= 0:
        return []
    return [quantile(u) for u in random_uniform(int(n), seed=seed, stream=0)]


def log_i0(z):
    """log of the modified Bessel function I_0(z), z >= 0."""
    z = abs(z)
    if z < 50:
        s, t, k = 1.0, 1.0, 0
        while True:
            k += 1
            t *= (z / 2) ** 2 / (k * k)
            s += t
            if t < 1e-17 * s:
                break
        return math.log(s)
    # asymptotic expansion: I0(z) ~ e^z / sqrt(2 pi z) * sum_k ((2k-1)!!)^2 / (k! (8z)^k)
    s, t = 1.0, 1.0
    for k in range(1, 30):
        t *= (2 * k - 1) ** 2 / (k * 8 * z)
        if t < 1e-17:
            break
        s += t
    return z - 0.5 * math.log(2 * math.pi * z) + math.log(s)


def gauss_legendre(f, a, b, n=64):
    """Composite 16-point Gauss-Legendre over n panels."""
    xs = (
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    )
    ws = (
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    )
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        c = a + (i + 0.5) * h
        for x, w in zip(xs, ws):
            s += w * (f(c + x * h / 2) + f(c - x * h / 2))
    return s * h / 2
