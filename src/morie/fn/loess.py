# morie.fn -- k02 batch (rootcoder007/morie)
"""LOWESS: locally weighted scatterplot smoothing with robustness iterations.

Source consulted: Cleveland, W.S. (1979), Robust locally weighted regression
and smoothing scatterplots, *JASA* 74(368), 829-836.  At each x_i the ``r =
floor(f n)`` nearest points get tricube weights

    w(u) = (1 - |u|^3)^3,  u = (x_j - x_i) / d_i,  d_i the largest distance used

a weighted line (degree 1) is fitted and its value at x_i taken as the fit.
The paper's robustness step then recomputes the fit ``iter`` more times with
each point additionally weighted by the bisquare of its residual over six
times the median absolute residual, which is what makes the smoother resist
outliers.  This is the algorithm behind ``stats::lowess``; the implementation
follows Cleveland's own ``lowest``/``clowess`` step for step, including the
h9/h1 and c9/c1 guards, so it reproduces R's output exactly at ``delta = 0``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["loess"]


def _lowest(x, y, xs, nleft, nright, w, userw, rw, n):
    rng = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    j = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if userw:
                w[j] = w[j] * rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
    nrt = j - 1
    if a <= 0.0:
        return None
    for j in range(nleft, nrt + 1):
        w[j] = w[j] / a
    if h > 0.0:
        a = 0.0
        for j in range(nleft, nrt + 1):
            a += w[j] * x[j]
        b = xs - a
        c = 0.0
        for j in range(nleft, nrt + 1):
            c += w[j] * (x[j] - a) ** 2
        if float(np.sqrt(c)) > 0.001 * rng:
            b = b / c
            for j in range(nleft, nrt + 1):
                w[j] = w[j] * (b * (x[j] - a) + 1.0)
    ys = 0.0
    for j in range(nleft, nrt + 1):
        ys += w[j] * y[j]
    return ys


def loess(x, y, span=2.0 / 3.0, iterations=3):
    """Cleveland's LOWESS smoother.

    Parameters
    ----------
    x, y : array-like
        Predictor and response; ``x`` need not be sorted.
    span : float, default 2/3
        Fraction of points in each local neighbourhood.
    iterations : int, default 3
        Number of robustness iterations after the first fit.

    Returns
    -------
    RichResult
        estimate (fitted values in the sorted-x order), x, fitted, residuals,
        robustness_weights, span, iterations, n, method.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    ordr = np.argsort(xa).tolist()
    xv = [float(xa[i]) for i in ordr]
    yv = [float(ya[i]) for i in ordr]
    n = len(xv)
    ys = [0.0] * n
    rw = [1.0] * n
    res = [0.0] * n
    w = [0.0] * n
    ns = max(2, min(n, int(n * float(span) + 1e-7)))
    for it in range(int(iterations) + 1):
        nleft = 0
        nright = ns - 1
        for i in range(n):
            while nright < n - 1:
                d1 = xv[i] - xv[nleft]
                d2 = xv[nright + 1] - xv[i]
                if d1 <= d2:
                    break
                nleft += 1
                nright += 1
            val = _lowest(xv, yv, xv[i], nleft, nright, w, it > 0, rw, n)
            ys[i] = yv[i] if val is None else val
        for i in range(n):
            res[i] = yv[i] - ys[i]
        if it == int(iterations):
            break
        absr = sorted(abs(t) for t in res)
        m1 = n // 2
        m2 = n - m1 - 1
        cmad = 3.0 * (absr[m1] + absr[m2])
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for i in range(n):
            r = abs(res[i])
            if r <= c1:
                rw[i] = 1.0
            elif r <= c9:
                rw[i] = (1.0 - (r / cmad) ** 2) ** 2
            else:
                rw[i] = 0.0
    return RichResult(
        payload={
            "estimate": ys,
            "x": xv,
            "fitted": ys,
            "residuals": res,
            "robustness_weights": rw,
            "span": float(span),
            "iterations": int(iterations),
            "n": int(n),
            "method": "LOWESS locally weighted regression (Cleveland 1979)",
        }
    )


# CANONICAL TEST
# >>> y = [1.2, 2.3, 2.9, 4.1, 5.2, 5.8, 7.3, 8.1, 8.9, 10.2]
# >>> r = loess(list(range(1, 11)), y, span=0.5, iterations=3)
# >>> # stats::lowess(1:10, y, f = 0.5, iter = 3, delta = 0)
# >>> assert abs(r["fitted"][0] - 1.25178975416042) < 1e-10
# >>> assert abs(r["fitted"][7] - 8.1) < 1e-10
# >>> assert abs(r["fitted"][9] - 10.0965394786483) < 1e-10


def cheatsheet():
    return "loess(x, y, span): Cleveland LOWESS smoother."


lowessfit = loess
