# morie.fn -- function file (rootcoder007/morie)
"""EMG parameters in relation to the force exerted (Rangayyan Sect. 5.9).

Book pages read as rendered images (pdftoppm, 130 dpi), not from an
extracted text layer: Rangayyan and Krishnan, *Biomedical Signal
Analysis*, 3rd ed., Wiley-IEEE Press, 2024, Section 5.9 "Application:
Quantitative Analysis of the EMG in Relation to Force Exerted",
pp. 290-292, with equation (5.28) on p. 292.

The procedure on p. 290 is followed literally:

  "starting from the first sample, the point where the force signal
  increased beyond 10% MVC was identified.  Then, the next point where
  the signal dropped below 10% MVC was identified.  This process was
  repeated until the end of the signal.  To refine the definition of
  each interval of contraction, a threshold was defined as 0.7 times
  the maximum level of contraction within the interval.  Then, the
  smaller extent of each interval previously identified, within which
  the force remained above the threshold, was detected."

  "Within each interval of the force signal identified as above as well
  as the corresponding interval in the EMG signal, the average force
  exerted, the RMS value, ZCR, and the turns count divided by the time
  duration of the interval (referred to as the turns count rate or TCR)
  were computed.  The threshold to detect significant turns was set at
  100 microV."

The goodness of fit between each parameter and force is equation (5.28),
p. 292, read from the rendered page:

    r^2 = [ sum_{n=1}^{N} x(n) y(n) - N xbar ybar ]^2
          / ( [ sum x^2(n) - N xbar^2 ] [ sum y^2(n) - N ybar^2 ] )

The book's own figures for the record of Figure 5.13 give r^2 = 0.98 for
RMS, 0.78 for ZCR and 0.97 for TCR (captions of Figures 5.15, 5.16 and
5.17), which is the pattern this function reproduces: RMS and TCR track
force closely, ZCR does not.

The RMS, ZCR and turns count are NOT reimplemented here.  Sections 5.6.1,
5.6.2 and 5.6.3 already have one implementation each in this package and
they are called: duplicating them is how the two copies drift apart.
"""

from __future__ import annotations

from math import sqrt

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["rangayyan_emg_force"]

# fraction of the maximum voluntary contraction that opens an interval
_MVC_FRACTION = 0.10
# "a threshold was defined as 0.7 times the maximum level of contraction
# within the interval"
_REFINE_FRACTION = 0.70


def _aslist(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _r2(x, y):
    """Equation (5.28), p. 292, written exactly as the book prints it."""
    n = len(x)
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    sx = 0.0
    sy = 0.0
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
    """Ordinary least-squares straight line, the book's "linear model"."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    den = n * sxx - sx * sx
    if den == 0.0:
        return float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _intervals(force):
    """The p. 290 two-stage delineation of the contraction intervals."""
    n = len(force)
    peak = max(force)
    thr = _MVC_FRACTION * peak
    coarse = []
    i = 0
    while i < n:
        if force[i] > thr:
            j = i
            while j < n and force[j] > thr:
                j += 1
            coarse.append((i, j))
            i = j
        else:
            i += 1
    fine = []
    for (a, b) in coarse:
        seg = force[a:b]
        t2 = _REFINE_FRACTION * max(seg)
        # the LONGEST run inside the interval over which the force stays
        # above the refined threshold -- "the smaller extent of each
        # interval previously identified, within which the force remained
        # above the threshold"
        best = None
        k = 0
        m = len(seg)
        while k < m:
            if seg[k] >= t2:
                q = k
                while q < m and seg[q] >= t2:
                    q += 1
                if best is None or (q - k) > (best[1] - best[0]):
                    best = (k, q)
                k = q
            else:
                k += 1
        if best is not None:
            fine.append((a + best[0], a + best[1]))
    return fine


def rangayyan_emg_force(emg, force, fs, window=None, turn_threshold=100.0):
    """RMS, ZCR and TCR of the EMG against force, with equation (5.28).

    Parameters
    ----------
    emg : array-like
        The EMG signal.
    force : array-like
        The simultaneously recorded force signal, same length and
        sampling rate.  Its own maximum is taken as the MVC level, so it
        may be in %MVC or in any consistent unit.
    fs : float
        Sampling rate in Hz.
    window : int, optional
        Length in samples of the causal short-time window of Section
        5.6; when given, the short-time RMS and turns-count SERIES of
        Figure 5.10 are returned alongside the per-interval parameters.
        It plays no part in the per-interval parameters, which the book
        computes over whole intervals.
    turn_threshold : float
        The significant-turn threshold of Section 5.6.3; 100 microV in
        the book's own experiment.

    Returns
    -------
    estimate : r2 of RMS against force, the book's headline quantity
    intervals : the detected contraction intervals, as sample ranges
    force_levels, rms, zcr, tcr : one value per interval
    r2_rms, r2_zcr, r2_tcr : equation (5.28) for each parameter
    slope_rms, intercept_rms, ... : the straight-line fits
    """
    from .bsastat import rms as _rms
    from .bsastat import rangayyan_zero_crossing as _zcr
    from .bsastat import turnscount as _turns

    e = _aslist(emg)
    f = _aslist(force)
    n = len(e)
    if n == 0:
        raise ValueError("rangayyan_emg_force: emg is empty")
    if len(f) != n:
        raise ValueError("rangayyan_emg_force: emg and force must have the same length")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("rangayyan_emg_force: fs must be positive")
    if max(f) <= 0.0:
        raise ValueError("rangayyan_emg_force: the force signal has no positive excursion")
    if turn_threshold < 0.0:
        raise ValueError("rangayyan_emg_force: turn_threshold must be nonnegative")

    ivs = _intervals(f)
    lev = []
    rv = []
    zv = []
    tv = []
    for (a, b) in ivs:
        seg_f = f[a:b]
        seg_e = e[a:b]
        lev.append(sum(seg_f) / len(seg_f))
        rv.append(_rms(seg_e)["rms"])
        # ZCR per second, as the book plots it against time
        zv.append(_zcr(seg_e, fs)["zcr_per_second"] if len(seg_e) >= 2 else float("nan"))
        # "the turns count divided by the time duration of the interval"
        if len(seg_e) >= 3:
            tv.append(_turns(seg_e, turn_threshold)["turns"] / (len(seg_e) / fs))
        else:
            tv.append(float("nan"))

    if len(ivs) >= 2:
        r2r = _r2(lev, rv)
        r2z = _r2(lev, zv)
        r2t = _r2(lev, tv)
        sr, ir = _linfit(lev, rv)
        sz, iz = _linfit(lev, zv)
        st, it = _linfit(lev, tv)
    else:
        r2r = r2z = r2t = float("nan")
        sr = ir = sz = iz = st = it = float("nan")

    payload = {
        "estimate": r2r,
        "intervals": [list(iv) for iv in ivs],
        "n_intervals": len(ivs),
        "force_levels": lev,
        "rms": rv,
        "zcr": zv,
        "tcr": tv,
        "r2_rms": r2r,
        "r2_zcr": r2z,
        "r2_tcr": r2t,
        "slope_rms": sr,
        "intercept_rms": ir,
        "slope_zcr": sz,
        "intercept_zcr": iz,
        "slope_tcr": st,
        "intercept_tcr": it,
        "mvc": max(f),
        "n": n,
        "fs": fs,
        "method": "Rangayyan (2024) Sect. 5.9 pp.290-292, interval delineation at 10% MVC refined at 0.7 of the interval peak; eq. (5.28) for r^2",
    }
    if window is not None:
        w = int(window)
        payload["short_time_rms"] = _rms(e, window=w)["short_time"]
        payload["short_time_turns"] = _turns(e, turn_threshold, window=w)["short_time"]
        payload["window"] = w
    return RichResult(
        title="EMG parameters versus force",
        summary_lines=[("intervals", len(ivs)), ("r2 RMS", r2r), ("r2 ZCR", r2z), ("r2 TCR", r2t)],
        payload=payload,
    )


def cheatsheet():
    return "rgemgf: RMS, ZCR and TCR of the EMG against force, Rangayyan Sect. 5.9"
