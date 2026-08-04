# morie.fn -- function file (rootcoder007/morie)
"""EMG correlates of isometric muscular contraction (Rangayyan Sect. 5.11).

Book pages read from the typeset PDF: Rangayyan and Krishnan,
*Biomedical Signal Analysis*, 3rd ed., Wiley-IEEE Press, 2024,
Section 5.11 "Application: Electrical and Mechanical Correlates of
Muscular Contraction", pp. 294-296, with Figures 5.20 and 5.21.

  "the subjects performed isometric contraction (that is, with no
  movement of the associated leg) of the rectus femoris (thigh) muscle
  to different levels of torque ... Four levels of contraction were
  performed from 20% to 80% of the MVC level of the individual subject
  ... Each contraction was held for a duration of about 6 s ...
  RMS values were computed for each contraction level over a duration
  of 5 s ... The almost-linear trends of the RMS values ... with
  muscular contraction indicate the usefulness of the derived parameter
  in the analysis of muscular activity.  It should, however, be noted
  that the relationship between RMS values and contraction may not
  follow the same (linear) pattern for different muscles."

Because the contraction is ISOMETRIC the commanded level is held
constant for the duration of each trial, so the levels are read off the
force channel as maximal runs of a CONSTANT commanded value.  This is
what separates this function from rgemgf, which delineates intervals
from a continuously varying force trace by the 10% MVC rule of
Section 5.9.

The linear trend is quantified by equation (5.28), p. 292.  Figure 5.21
is the reason r^2 is returned rather than only the slope: for the biceps
and the deltoid the relationship is NOT linear, and only r^2 shows it.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["rangayyan_isometric_contraction"]


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


def _runs(f):
    """Maximal runs of an identical commanded level."""
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


def rangayyan_isometric_contraction(emg, force, fs, rest_level=0.0):
    """RMS of the EMG at each held level of isometric contraction.

    Parameters
    ----------
    emg : array-like
        The EMG signal.
    force : array-like
        The commanded contraction level, one value per EMG sample, held
        constant within each trial (%MVC in the book's experiment).
    fs : float
        Sampling rate in Hz.
    rest_level : float
        Levels at or below this value are rest, not contraction, and are
        excluded from the fit.  Zero by default.

    Returns
    -------
    estimate : r2 of the RMS against the contraction level, eq. (5.28)
    levels   : the held levels, in the order they occur
    rms      : the RMS value of the EMG at each level, eq. (3.9)
    durations : the duration in seconds of each held level
    slope, intercept : the straight-line fit of Figure 5.20
    r2       : the same as estimate
    """
    from .bsastat import rms as _rms

    e = _aslist(emg)
    f = _aslist(force)
    n = len(e)
    if n == 0:
        raise ValueError("rangayyan_isometric_contraction: emg is empty")
    if len(f) != n:
        raise ValueError("rangayyan_isometric_contraction: emg and force must have the same length")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("rangayyan_isometric_contraction: fs must be positive")

    lev = []
    rv = []
    dur = []
    ivs = []
    for (a, b) in _runs(f):
        if f[a] <= rest_level:
            continue
        lev.append(f[a])
        rv.append(_rms(e[a:b])["rms"])
        dur.append((b - a) / fs)
        ivs.append([a, b])
    if len(lev) < 2:
        raise ValueError("rangayyan_isometric_contraction: need at least two held contraction levels")
    r2 = _r2(lev, rv)
    slope, intercept = _linfit(lev, rv)
    return RichResult(
        title="Isometric contraction: EMG RMS versus level",
        summary_lines=[("levels", len(lev)), ("slope", slope), ("r2", r2)],
        payload={
            "estimate": r2,
            "levels": lev,
            "rms": rv,
            "durations": dur,
            "intervals": ivs,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "n_levels": len(lev),
            "n": n,
            "fs": fs,
            "method": "Rangayyan (2024) Sect. 5.11 pp.294-296, RMS per held isometric level; eq. (5.28) for r^2",
        },
    )


def cheatsheet():
    return "rgisint: EMG RMS at each held level of isometric contraction, Rangayyan Sect. 5.11"
