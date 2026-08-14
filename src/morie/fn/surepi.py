# morie.fn -- function file (rootcoder007/morie)
r"""EARS: aberration detection on a short, moving baseline.

Syndromic surveillance rarely has years of clean history for the thing
being watched. EARS is built for that case: every method uses only the
last few days, so it can start flagging almost immediately after a data
feed is switched on.

**All three long-term methods are the same CUSUM with different
baselines.** The underlying recursion is

.. math:: S_t = \max\!\left(0,\; S_{t-1}
          + \frac{X_t - (\mu_0 + k\sigma_t)}{\sigma_t}\right),

and the variants differ in two choices only:

``C1-MILD``
    mean and standard deviation from days :math:`t-7 \dots t-1`, with
    :math:`S_{t-1} = 0`. Because the previous CUSUM is discarded, this
    reduces **exactly** to a z-score against the trailing week, and a
    flag is raised when the count exceeds the baseline mean by three
    standard deviations.
``C2-MEDIUM``
    the same, but the baseline is days :math:`t-9 \dots t-3`. The
    two-day gap is the point: an outbreak that has already begun
    contaminates a baseline that runs right up to the present, and a
    contaminated baseline raises :math:`\sigma` and hides the signal.
    C2 buys insensitivity to that at the cost of two days' staleness.
``C3-ULTRA``
    accumulates, using C2's baseline and summing the current and two
    preceding C2 statistics -- an average run length of three days. It
    catches a sustained small elevation that no single day would flag,
    and correspondingly it is the one that will keep firing after the
    event has passed.

**The zero standard deviation problem is real and is handled
explicitly.** On a quiet feed the trailing week is often constant --
seven zeros, or seven ones. Then :math:`\sigma = 0` and the z-score is
undefined; a single case the next day is either "infinitely
significant" or silently dropped, depending on whose implementation
you use. Neither is acceptable, so ``sigma_floor`` sets the smallest
usable standard deviation and the choice is visible in the result
rather than buried.

**Two other published EARS methods, kept because the paper carries
them.** ``salmonella_cusum`` is eq. (4) with the five-year weekly mean
and week-specific standard deviation, flagging at :math:`S_t \ge 0.5`
for counts of five or more -- the serotype-specific algorithm. And
``compound_smoothing`` is the 4253H method of eq. (5), flagging when
:math:`x_0 > \beta + 2\sigma_x` against a repeatedly-smoothed
baseline.

**What a flag is and is not.** These are aberration detectors, not
tests of an epidemiological hypothesis. A flag says the count is
unusual against its own recent history; it does not say an outbreak is
occurring, and the paper is explicit that the thresholds exist to be
tuned by local health departments to the sensitivity and specificity
they are willing to pay for.

References
----------
Hutwagner, L., Thompson, W., Seeman, G. M. & Treadwell, T. (2003)
"The Bioterrorism Preparedness and Response Early Aberration
Reporting System (EARS)", *Journal of Urban Health: Bulletin of the
New York Academy of Medicine* 80(2, Supplement 1), i89-i96. The
article prints no DOI. Section "Long-term implementation methods with
limited baseline data" (C1-MILD, C2-MEDIUM, C3-ULTRA, their baselines
and the three-standard-deviation flag), eq. (4) (the CUSUM recursion
and the Salmonella algorithm) and eq. (5) (the 4253H compound
smoothing threshold).

Hutwagner, L. C., Maloney, E. K., Bean, N. H., Slutsker, L. & Martin,
S. M. (1997) "Using laboratory-based surveillance data for
prevention: an algorithm for detecting Salmonella outbreaks",
*Emerging Infectious Diseases* 3(3), 395-400. Reference 7 of the EARS
paper; the source of the serotype CUSUM reproduced here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["c1_mild", "c2_medium", "c3_ultra", "ears_detect",
           "salmonella_cusum", "compound_smoothing"]

_EPS = 1e-12
_METHODS = ("C1", "C2", "C3")
# (lag, width): the baseline is the `width` days ending `lag`
# days before t. C1 is t-7..t-1 and C2/C3 are t-9..t-3 --
# both SEVEN days, differing only in how far back they stop.
_WINDOWS = {"C1": (1, 7), "C2": (3, 7), "C3": (3, 7)}


def _baseline(counts, t, lag, width, sigma_floor):
    """Mean and sd of the `width` days ending `lag` days before t."""
    lo = t - lag - width + 1
    hi = t - lag
    if lo < 0:
        return None, None, 0
    win = [float(counts[q]) for q in range(lo, hi + 1)]
    m = sum(win) / len(win)
    if len(win) < 2:
        return m, sigma_floor, len(win)
    v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
    return m, max(math.sqrt(max(v, 0.0)), float(sigma_floor)), len(win)


def _stat(counts, method, sigma_floor):
    lag, width = _WINDOWS[method]
    n = len(counts)
    out = []
    for t in range(n):
        m, s, used = _baseline(counts, t, lag, width, sigma_floor)
        if m is None:
            out.append(None)
            continue
        out.append((float(counts[t]) - m) / s)
    return out


def c1_mild(counts, threshold=3.0, sigma_floor=1.0):
    r"""C1: z-score against days :math:`t-7 \dots t-1`.

    ``S_{t-1}`` is zero for C1, so the CUSUM recursion collapses to a
    plain standardised deviation -- which is why this is the mildest
    of the three.
    """
    return ears_detect(counts, method="C1", threshold=threshold,
                       sigma_floor=sigma_floor)


def c2_medium(counts, threshold=3.0, sigma_floor=1.0):
    r"""C2: as C1, but the baseline is days :math:`t-9 \dots t-3`.

    The two-day gap keeps the start of an outbreak out of its own
    baseline.
    """
    return ears_detect(counts, method="C2", threshold=threshold,
                       sigma_floor=sigma_floor)


def c3_ultra(counts, threshold=2.0, sigma_floor=1.0):
    r"""C3: the sum of the current and two preceding C2 statistics.

    Accumulating over three days detects a sustained small excess that
    no single day would flag. The published default threshold is 2.
    """
    return ears_detect(counts, method="C3", threshold=threshold,
                       sigma_floor=sigma_floor)


def ears_detect(counts, method="C2", threshold=3.0, sigma_floor=1.0):
    r"""Run one of the three EARS long-term detectors.

    Returns the statistic and the flag for every day. Days without a
    complete baseline return ``None`` rather than a fabricated value --
    an unformed statistic is not a zero.
    """
    if method not in _METHODS:
        raise ValueError("surepi: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    cv = [float(v) for v in k.vec(counts)]
    if any(v < 0.0 for v in cv):
        raise ValueError("surepi: counts must be non-negative")
    if float(sigma_floor) <= 0.0:
        raise ValueError("surepi: sigma_floor must be positive -- a "
                         "flat baseline gives sigma = 0 and an "
                         "undefined statistic")
    lag, width = _WINDOWS[method]
    need = lag + width - 1
    if len(cv) <= need:
        raise ValueError("surepi: %s needs more than %d days of "
                         "history, got %d" % (method, need, len(cv)))
    base = _stat(cv, "C2" if method == "C3" else method, sigma_floor)
    if method == "C3":
        stat = []
        for t in range(len(cv)):
            trio = [base[q] for q in (t, t - 1, t - 2)
                    if q >= 0 and base[q] is not None]
            # C3 accumulates only over days that HAVE a C2 statistic
            stat.append(sum(trio) if len(trio) == 3 else None)
    else:
        stat = base
    flags = [None if s is None else bool(s > float(threshold))
             for s in stat]
    return RichResult(payload={
        "estimate": stat, "statistic": stat, "flag": flags,
        "n_flagged": sum(1 for f in flags if f),
        "method": method, "threshold": float(threshold),
        "baseline_lag": lag, "baseline_width": width,
        "sigma_floor": float(sigma_floor),
        "n": len(cv), "n_evaluable": sum(1 for s in stat
                                         if s is not None),
        "reference": "Hutwagner, Thompson, Seeman & Treadwell (2003), "
                     "EARS long-term methods",
        "caveat": "a flag marks a count unusual against its own recent "
                  "history; it is not a test that an outbreak is "
                  "occurring",
    })


def salmonella_cusum(counts, mu0, sigma, k_shift=1.0, decision=0.5,
                     min_count=5):
    r"""Eq. (4): the serotype-specific Salmonella CUSUM.

    .. math:: S_t = \max\!\left(0,\; S_{t-1}
              + \frac{X_t - (\mu_0 + k\sigma_t)}{\sigma_t}\right)

    with the five-year weekly mean :math:`\mu_0` and the week-specific
    standard deviation. A flag needs both :math:`S_t \ge` ``decision``
    and a count of at least ``min_count``; the count floor is in the
    published algorithm because a CUSUM on tiny counts flags noise.
    """
    cv = [float(v) for v in k.vec(counts)]
    n = len(cv)
    mv = ([float(mu0)] * n if isinstance(mu0, (int, float))
          else [float(v) for v in k.vec(mu0)])
    sv = ([float(sigma)] * n if isinstance(sigma, (int, float))
          else [float(v) for v in k.vec(sigma)])
    if not (len(mv) == len(sv) == n):
        raise ValueError("surepi: mu0 and sigma must be scalars or "
                         "match the series length (%d, %d, %d)"
                         % (n, len(mv), len(sv)))
    if any(v <= 0.0 for v in sv):
        raise ValueError("surepi: sigma must be positive everywhere")
    S, out, flags = 0.0, [], []
    for t in range(n):
        S = max(0.0, S + (cv[t] - (mv[t] + float(k_shift) * sv[t]))
                / sv[t])
        out.append(S)
        flags.append(bool(S >= float(decision)
                          and cv[t] >= float(min_count)))
    return {"cusum": out, "flag": flags, "estimate": out,
            "n_flagged": sum(flags), "decision": float(decision),
            "k": float(k_shift), "min_count": float(min_count),
            "method": "Hutwagner et al. (2003) eq. (4); the Salmonella "
                      "Outbreak Detection Algorithm"}


def compound_smoothing(values, current, passes=(4, 2, 5, 3),
                       multiplier=2.0):
    r"""Eq. (5): the 4253H compound smoothing baseline and threshold.

    Running medians of the stated widths, then the hanning step that
    replaces each value by
    :math:`\tfrac14 x_{t-1} + \tfrac12 x_t + \tfrac14 x_{t+1}`. A flag
    is raised when :math:`x_0 > \beta + 2\sigma_x`, with
    :math:`\sigma_x` the standard deviation of the differences between
    the smoothed and raw series.
    """
    v = [float(x) for x in k.vec(values)]
    if len(v) < max(passes) + 2:
        raise ValueError("surepi: the series is too short for the "
                         "smoothing passes %s (have %d)"
                         % (list(passes), len(v)))
    s = list(v)
    for w in passes:
        s = _runmed(s, int(w))
    # the H of 4253H
    s = [s[0]] + [0.25 * s[i - 1] + 0.5 * s[i] + 0.25 * s[i + 1]
                  for i in range(1, len(s) - 1)] + [s[-1]]
    resid = [v[i] - s[i] for i in range(len(v))]
    m = sum(resid) / len(resid)
    sd = math.sqrt(sum((r - m) ** 2 for r in resid)
                   / max(len(resid) - 1, 1))
    base = s[-1]
    return {"smoothed": s, "baseline": base, "sigma": sd,
            "threshold": base + float(multiplier) * sd,
            "flag": bool(float(current) > base
                         + float(multiplier) * sd),
            "current": float(current),
            "method": "Hutwagner et al. (2003) eq. (5), 4253H compound "
                      "smoothing after Stern & Lightfoot"}


def _runmed(x, width):
    """Running median of odd or even width, endpoints carried."""
    n = len(x)
    if width <= 1 or width > n:
        return list(x)
    half = width // 2
    out = list(x)
    for i in range(n):
        lo, hi = i - half, i + half
        if width % 2 == 0:
            hi = i + half - 1
        if lo < 0 or hi >= n:
            continue
        out[i] = k.median(x[lo:hi + 1])
    return out


def cheatsheet():
    return ("surepi: EARS. C1 baseline = days t-7..t-1, C2 = t-9..t-3, "
            "C3 = sum of three consecutive C2s. S_{t-1} = 0 for C1 and "
            "C2, so both are just z-scores against a MOVING baseline; "
            "flag at mean + 3 sd. C2's two-day gap keeps a starting "
            "outbreak out of its own baseline. A flat baseline gives "
            "sigma = 0, so sigma_floor is explicit, not silent. A flag "
            "is an aberration, not an outbreak.")


# compact alias per ledger/NAMING.md
earssignal = ears_detect
