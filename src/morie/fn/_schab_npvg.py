"""Shared engine for the nonparametric semivariograms of Schabenberger & Gotway Sec. 4.6.1."""

import math

from ._schab_st import bessel_j0, gauss_legendre

_GL = {}


def omega(d, x):
    """Basis function Omega_d(t) of eq (4.8): cos(t), J0(t), sin(t)/t for d = 1, 2, 3."""
    x = [float(v) for v in x]
    if d == 1:
        return [math.cos(v) for v in x]
    if d == 3:
        return [1.0 if v == 0.0 else math.sin(v) / v for v in x]
    if d == 2:
        if not x:
            return []
        nq = max(200, int(1.5 * max(abs(v) for v in x)) + 100)
        return [float(v) for v in bessel_j0(x, n_quad=nq)]
    raise ValueError("`d` must be 1, 2 or 3")


def _gl(n):
    if n not in _GL:
        nodes, weights = gauss_legendre(n)
        _GL[n] = ([float(v) for v in nodes], [float(v) for v in weights])
    return _GL[n]


def omega_integral(d, h, lo, hi, n_per_panel=20):
    """Integral of Omega_d(h w) over w in [lo, hi], composite Gauss-Legendre.

    Panels are at most a quarter period of cos(h w) wide, so every panel
    sees a smooth, non-oscillating stretch of the integrand.
    """
    if hi <= lo:
        return 0.0
    if h == 0.0:
        return hi - lo
    panels = max(1, math.ceil((hi - lo) * abs(h) / (math.pi / 2)))
    t, wt = _gl(n_per_panel)
    step = (hi - lo) / panels
    pts, wts = [], []
    for p in range(panels):
        a = lo + p * step
        pts.extend(a + 0.5 * step * (v + 1.0) for v in t)
        wts.extend(0.5 * step * v for v in wt)
    vals = omega(d, [h * v for v in pts])
    return sum(w * v for w, v in zip(wts, vals))


def kernel_correlation(h, theta_l, theta_u, d, b):
    """K(h) = C(theta, h) / sigma^2 for F(theta, w) of eq (4.49) on [0, b].

    F is the U(theta_l, theta_u) cdf G for 0 <= w <= b, zero below 0 and one
    above b, so besides its density 1/(theta_u - theta_l) on
    [max(theta_l, 0), min(theta_u, b)] it has an atom G(0) at w = 0 when
    theta_l < 0 and an atom 1 - G(b) at w = b when theta_u > b.
    """
    width = theta_u - theta_l
    g0 = min(max(-theta_l / width, 0.0), 1.0)
    gb = min(max((b - theta_l) / width, 0.0), 1.0)
    lo, hi = max(theta_l, 0.0), min(theta_u, b)
    out = []
    for hv in h:
        k = g0 + (1.0 - gb) * omega(d, [hv * b])[0]
        k += omega_integral(d, hv, lo, hi) / width
        out.append(k)
    return out


def nelder_mead(fn, x0, step=0.35, max_iter=4000, tol=1e-14):
    """Plain Nelder-Mead, the same simplex moves as `_schaben._nelder_mead`."""
    n = len(x0)
    sim = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    f = [fn(p) for p in sim]
    for _ in range(int(max_iter)):
        order = sorted(range(n + 1), key=lambda i: f[i])
        sim, f = [sim[i] for i in order], [f[i] for i in order]
        if abs(f[-1] - f[0]) <= tol * (abs(f[0]) + tol):
            break
        cen = [sum(sim[i][j] for i in range(n)) / n for j in range(n)]
        xr = [2.0 * cen[j] - sim[-1][j] for j in range(n)]
        fr = fn(xr)
        if fr < f[0]:
            xe = [3.0 * cen[j] - 2.0 * sim[-1][j] for j in range(n)]
            fe = fn(xe)
            sim[-1], f[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < f[-2]:
            sim[-1], f[-1] = xr, fr
        else:
            xc = [0.5 * (cen[j] + sim[-1][j]) for j in range(n)]
            fc = fn(xc)
            if fc < f[-1]:
                sim[-1], f[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = [sim[0][j] + 0.5 * (sim[i][j] - sim[0][j]) for j in range(n)]
                    f[i] = fn(sim[i])
    i = min(range(n + 1), key=lambda k: f[k])
    return sim[i], f[i]
