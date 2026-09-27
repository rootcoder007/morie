# morie.fn -- function file (rootcoder007/morie)
"""Mark correlation function and mark variogram of a marked point pattern (Stoyan and Stoyan 1994)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from .ripk import _rect, isotropic_weight

__all__ = ["mark_correlation", "mark_variogram"]


def _quantile7(s, p):
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _bw_nrd0(x):
    """``stats::bw.nrd0``: 0.9 min(sd, IQR/1.34) n^(-1/5)."""
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    s = sorted(x)
    lo = min(sd, (_quantile7(s, 0.75) - _quantile7(s, 0.25)) / 1.34)
    if not lo > 0:
        lo = sd or abs(x[0]) or 1.0
    return 0.9 * lo * n ** (-0.2)


def _density(x, w, bw, lo_x, hi_x, n_user):
    """``stats::density`` (Gaussian kernel, unnormalised weights): linear
    binning on ``n`` points of ``[from - 4 bw, to + 4 bw]``, circular
    convolution with the kernel on ``2n`` points, linear interpolation."""
    n = max(n_user, 512)
    if n > 512:
        n = 2 ** math.ceil(math.log2(n))
    lo = lo_x - 4.0 * bw
    up = hi_x + 4.0 * bw
    dx = (up - lo) / (n - 1)
    y = [0.0] * (2 * n)
    for xi, wi in zip(x, w):
        pos = (xi - lo) / dx
        ix = math.floor(pos)
        fx = pos - ix
        if 0 <= ix <= n - 2:
            y[ix] += wi * (1.0 - fx)
            y[ix + 1] += wi * fx
        elif ix == -1:
            y[0] += wi * fx
        elif ix == n - 1:
            y[ix] += wi * (1.0 - fx)
    kern = [0.0] * (2 * n)
    c = 1.0 / (bw * math.sqrt(2.0 * math.pi))
    for k in range(2 * n):
        t = k * dx if k <= n else -(2 * n - k) * dx
        kern[k] = c * math.exp(-0.5 * (t / bw) ** 2)
    nz = [(j, v) for j, v in enumerate(y) if v != 0.0]
    dens = []
    for k in range(n):
        s = 0.0
        for j, v in nz:
            s += v * kern[(j - k) % (2 * n)]
        dens.append(max(0.0, s))
    out = []
    for i in range(n_user):
        xv = lo_x + (hi_x - lo_x) * i / (n_user - 1)
        p = (xv - lo) / dx
        a = min(int(math.floor(p)), n - 2)
        out.append(dens[a] + (p - a) * (dens[a + 1] - dens[a]))
    return out


def _pairs(P, x0, x1, y0, y1, rmax, correction):
    area = (x1 - x0) * (y1 - y0)
    out = []
    for i in range(len(P)):
        for j in range(len(P)):
            if i == j:
                continue
            d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1])
            if d > rmax:
                continue
            if correction == "iso":
                e = min(1.0 / isotropic_weight(P[i][0], P[i][1], d, x0, x1, y0, y1), 100.0)
            else:
                e = area / ((x1 - x0 - abs(P[i][0] - P[j][0])) * (y1 - y0 - abs(P[i][1] - P[j][1])))
            out.append((i, j, d, e))
    return out


def _smooth(points, marks, window, f, ef, r, rmax, correction):
    P = [(float(a), float(b)) for a, b in np.asarray(points, dtype=float).tolist()]
    m = [float(v) for v in np.asarray(marks, dtype=float).tolist()]
    n = len(P)
    if len(m) != n or n < 2:
        raise ValueError("points and marks must have the same length >= 2")
    if correction not in ("iso", "trans"):
        raise ValueError("correction must be 'iso' or 'trans'")
    x0, x1, y0, y1 = _rect(window)
    if rmax is None:
        rmax = min(math.sqrt(1000.0 / (math.pi * n / ((x1 - x0) * (y1 - y0)))), min(x1 - x0, y1 - y0) / 4.0)
    if r is None:
        r = [rmax * k / 512 for k in range(513)]
    r = [float(v) for v in r]
    pr = _pairs(P, x0, x1, y0, y1, max(r), correction)
    d = [p[2] for p in pr]
    wt = [p[3] for p in pr]
    ff = [f(m[p[0]], m[p[1]]) for p in pr]
    bw = _bw_nrd0(d)
    k1 = _density(d, wt, bw, min(r), max(r), len(r))
    kf = _density(d, [a * b for a, b in zip(ff, wt)], bw, min(r), max(r), len(r))
    # the ratio is undefined where the smoothed pair density is numerically zero
    # (stats::density leaves FFT rounding of order 1e-17 there)
    floor = 1e-6 * max(k1)
    est = [kf[i] / (ef * k1[i]) if k1[i] > floor else float("nan") for i in range(len(r))]
    return r, est, bw


def mark_correlation(
    points, marks, window, *, r=None, rmax=None, correction: str = "iso", normalise: bool = True
) -> RichResult:
    r"""Stoyan's mark correlation function ``k_mm(r)``.

    .. math::

        k_{mm}(r) = \frac{E[m_i m_j \mid d_{ij} = r]}{E[m]^2}

    estimated by kernel smoothing of the edge-corrected pair
    contributions over pair distance, exactly as
    ``spatstat.explore::markcorr`` with ``method = "density"``: a
    Gaussian kernel with the ``bw.nrd0`` bandwidth of the pair distances,
    evaluated by ``stats::density`` (linear binning and FFT convolution,
    reproduced here), numerator weights ``m_i m_j e_ij`` over denominator
    weights ``e_ij``.  ``e_ij`` is Ripley's isotropic weight (capped at
    100) or the translation weight.  Values above 1 mean pairs at that
    distance carry larger marks than random pairs.  Where the smoothed pair
    density is below ``1e-6`` of its maximum (no pairs near ``r``) the ratio
    is undefined and returned as NaN; ``markcorr`` returns rounding noise there.

    :param points: (n, 2) coordinates.
    :param marks: Non-negative numeric marks (n,).
    :param window: ``(xmin, xmax, ymin, ymax)``.
    :param r: Distances (default 513 values up to the ``rmax`` rule of Kest).
    :param rmax: Largest distance for the default ``r``.
    :param correction: ``iso`` or ``trans``.
    :param normalise: Divide by ``E[m]^2`` (else return ``c_mm(r)``).
    :return: :class:`RichResult` with ``r``, ``k``, ``bw``, ``theo``.

    References
    ----------
    Stoyan, D. and Stoyan, H. (1994). *Fractals, Random Shapes and Point
    Fields*. Wiley, Chichester.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> mk = [1.0, 2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 2.0, 0.8, 1.2]
    >>> r = mark_correlation(pts, mk, (0, 1, 0, 1))
    >>> round(r.k[256], 6)
    0.998959
    """
    m = [float(v) for v in np.asarray(marks, dtype=float).tolist()]
    if any(v < 0 for v in m) and normalise:
        raise ValueError("negative marks are not permitted when normalise=True")
    ef = (sum(m) / len(m)) ** 2
    rr, est, bw = _smooth(points, m, window, lambda a, b: a * b, ef if normalise else 1.0, r, rmax, correction)
    return RichResult(payload={"r": rr, "k": est, "bw": bw, "theo": 1.0 if normalise else ef})


def mark_variogram(points, marks, window, *, r=None, rmax=None, correction: str = "iso") -> RichResult:
    r"""Mark variogram ``gamma(r) = E[(m_i - m_j)^2 / 2 | d_ij = r]``.

    As ``spatstat.explore::markvario``: :func:`mark_correlation`'s kernel
    smoother with ``f = (m_1 - m_2)^2 / 2`` and no normalisation; under
    independent marks ``gamma(r)`` equals the mark variance (``theo``).

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> mk = [1.0, 2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 2.0, 0.8, 1.2]
    >>> round(mark_variogram(pts, mk, (0, 1, 0, 1)).gamma[256], 6)
    0.32
    """
    m = [float(v) for v in np.asarray(marks, dtype=float).tolist()]
    mu = sum(m) / len(m)
    var = sum((v - mu) ** 2 for v in m) / (len(m) - 1)
    rr, est, bw = _smooth(points, m, window, lambda a, b: 0.5 * (a - b) ** 2, 1.0, r, rmax, correction)
    return RichResult(payload={"r": rr, "gamma": est, "bw": bw, "theo": var})


def cheatsheet() -> str:
    return "mark_correlation / mark_variogram -> Stoyan mark correlation and mark variogram (spatstat markcorr, markvario)."
