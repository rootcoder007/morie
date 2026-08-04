# morie.fn -- function file (rootcoder007/morie)
"""Fractal dimension of the EMG against force (Rangayyan Sect. 5.13).

Book page read as a rendered image (pdftoppm, 130 dpi), not from an
extracted text layer: Rangayyan and Krishnan, *Biomedical Signal
Analysis*, 3rd ed., Wiley-IEEE Press, 2024, Section 5.13.2 "Fractal
dimension", "Higuchi's method", p. 304, equations (5.39)-(5.41), and
Section 5.13.4 "Fractal analysis of EMG signals", p. 305.

Equation (5.39), the reconstructed signals:

    x_k(m) = x(m), x(m+k), x(m+2k), ..., x(m + floor((N-m)/k) k),
    m = 1, 2, ..., k

Equation (5.40), the length of each reconstruction:

    L(m,k) = (1/k) * (N-1) / (k floor((N-m)/k))
             * sum_{i=1}^{floor((N-m)/k)} |x(m+ik) - x[m+(i-1)k]|

Equation (5.41):

    L(k) = (1/k) sum_{m=1}^{k} L(m,k)

"The slope of a straight-line fit to a log-log plot of L(k) against 1/k
gives the FD of the original signal."

Note the LEADING 1/k of equation (5.40), which is Higuchi (1988) eq. (1).
Dropping it multiplies L(k) by k and reduces the fitted slope by exactly
one, so a straight line -- whose fractal dimension is 1 -- comes out as
0.  That is the failure the straight-line anchor below is there to catch.

Section 5.13.4, p. 305: "Segments of duration 1 s were cut for each level
of contraction to estimate FD.  It is evident that FD increases with the
level of contraction (except for the last level of contraction) with high
correlation."  Figure 5.25 reports r^2 = 0.95 for that record.  The
goodness of fit is equation (5.28), p. 292.

Higuchi (1988) is the primary source for the estimator itself: Higuchi,
T., "Approach to an irregular time series on the basis of the fractal
theory", *Physica D* 31:277-283, 1988.
"""

from __future__ import annotations

from math import log

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["rangayyan_emg_fractal_dim"]


def _aslist(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _r2(x, y):
    """Equation (5.28), p. 292."""
    n = len(x)
    sxy = sxx = syy = sx = sy = 0.0
    for i in range(n):
        sxy += x[i] * y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sx += x[i]
        sy += y[i]
    xbar = sx / n
    ybar = sy / n
    num = sxy - n * xbar * ybar
    den = (sxx - n * xbar * xbar) * (syy - n * ybar * ybar)
    if den <= 0.0:
        return float("nan")
    return (num * num) / den


def _linfit(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    den = n * sxx - sx * sx
    if den == 0.0:
        return float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / n


def higuchi_fd(x, kmax):
    """Equations (5.39)-(5.41): Higuchi's fractal dimension of x."""
    xs = _aslist(x)
    N = len(xs)
    kk = int(kmax)
    if N < 4:
        raise ValueError("higuchi_fd: need at least four samples")
    if kk < 2:
        raise ValueError("higuchi_fd: kmax must be at least two")
    kk = min(kk, N // 2)
    Lk = []
    ks = []
    for k in range(1, kk + 1):
        acc = 0.0
        used = 0
        for m in range(1, k + 1):
            # eq (5.39), 1-based m; index m-1 into the 0-based list
            idx = list(range(m - 1, N, k))
            if len(idx) < 2:
                continue
            terms = len(idx) - 1          # = floor((N-m)/k)
            s = 0.0
            for i in range(terms):
                s += abs(xs[idx[i + 1]] - xs[idx[i]])
            # eq (5.40): the leading 1/k, then (N-1)/(k * terms)
            acc += (s / k) * ((N - 1) / (k * terms))
            used += 1
        if used == 0:
            continue
        Lk.append(acc / used)             # eq (5.41)
        ks.append(k)
    pts = [(log(1.0 / ks[i]), log(Lk[i])) for i in range(len(ks)) if Lk[i] > 0.0]
    if len(pts) < 2:
        raise ValueError("higuchi_fd: the signal has no measurable length")
    slope, _ = _linfit([p[0] for p in pts], [p[1] for p in pts])
    return slope, ks, Lk


def _runs(f):
    out = []
    i = 0
    n = len(f)
    while i < n:
        j = i
        while j < n and f[j] == f[i]:
            j += 1
        out.append((i, j))
        i = j
    return out


def rangayyan_emg_fractal_dim(emg, force, fs, kmax=10, rest_level=0.0):
    """Higuchi FD of a 1 s segment at each level of contraction.

    Parameters
    ----------
    emg : array-like
        The EMG signal.
    force : array-like
        The commanded contraction level, one value per EMG sample, held
        constant within each trial.
    fs : float
        Sampling rate in Hz.  One second, that is round(fs) samples, is
        cut from the start of each level, as Section 5.13.4 describes;
        shorter levels are used whole.
    kmax : int
        The largest lag k of equations (5.39)-(5.41).
    rest_level : float
        Levels at or below this are rest and are excluded.

    Returns
    -------
    estimate : r2 of FD against the contraction level, eq. (5.28)
    levels   : the held levels
    fd       : Higuchi FD of the 1 s segment at each level
    slope, intercept : the straight-line fit of Figure 5.25
    """
    e = _aslist(emg)
    f = _aslist(force)
    n = len(e)
    if n == 0:
        raise ValueError("rangayyan_emg_fractal_dim: emg is empty")
    if len(f) != n:
        raise ValueError("rangayyan_emg_fractal_dim: emg and force must have the same length")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("rangayyan_emg_fractal_dim: fs must be positive")
    seglen = int(round(fs))
    if seglen < 4:
        raise ValueError("rangayyan_emg_fractal_dim: fs is too low for a one-second segment")

    lev = []
    fd = []
    ivs = []
    for (a, b) in _runs(f):
        if f[a] <= rest_level:
            continue
        stop = min(b, a + seglen)
        if stop - a < 4:
            continue
        lev.append(f[a])
        fd.append(higuchi_fd(e[a:stop], kmax)[0])
        ivs.append([a, stop])
    if len(lev) < 2:
        raise ValueError("rangayyan_emg_fractal_dim: need at least two usable contraction levels")
    r2 = _r2(lev, fd)
    slope, intercept = _linfit(lev, fd)
    return RichResult(
        title="Fractal dimension of the EMG versus force",
        summary_lines=[("levels", len(lev)), ("slope", slope), ("r2", r2)],
        payload={
            "estimate": r2,
            "levels": lev,
            "fd": fd,
            "intervals": ivs,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "kmax": int(kmax),
            "segment_samples": seglen,
            "n_levels": len(lev),
            "n": n,
            "fs": fs,
            "method": "Rangayyan (2024) eqs. (5.39)-(5.41) p.304 Higuchi FD per 1 s segment, Sect. 5.13.4 p.305; eq. (5.28) for r^2",
        },
    )


def cheatsheet():
    return "rgemgfd: Higuchi fractal dimension of the EMG against force, Rangayyan Sect. 5.13"
