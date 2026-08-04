# morie.fn -- function file (rootcoder007/morie)
"""Daniell-smoothed periodogram."""

from math import fsum

from ._richresult import RichResult
from .specdn import spectral_density

__all__ = [
    "spectral_smoothed",
    "smpgram",
]


def spectral_smoothed(y, span=3):
    """Periodogram smoothed by a Daniell (equal-weight) window.

    The raw periodogram of Schabenberger & Gotway Sec. 4.7.1 is an
    INCONSISTENT estimator of the spectral density: its variance does not
    fall as the record lengthens, only its bias does. The standard repair
    is to average adjacent Fourier ordinates, trading resolution for
    variance; with equal weights that is the Daniell smoother.

    NOT IN SCHABENBERGER & GOTWAY. A fixed-string search of the book for
    "Daniell" and for "smoothed periodogram" returns nothing; Sec. 4.7.2
    fits a PARAMETRIC spectral density model by weighted least squares on
    the periodogram instead, which is a different repair for the same
    problem. The periodogram this function smooths is the book's
    (eq (4.57), Sec. 4.7.1.1); the smoothing step is the conventional one
    described in the time-series literature under Daniell's name -- see
    Bloomfield, P. (2000), *Fourier Analysis of Time Series*, 2nd edn,
    Wiley, Ch. 8. That reference is named from the general literature and
    was NOT verified against a PDF in this corpus.

    The window is applied CIRCULARLY over the Fourier ordinates, because
    the frequency axis of a real record is symmetric and wraps; a
    truncating window would bias the two ends.

    Parameters
    ----------
    y : (r,) array-like
        Values on a one-dimensional lattice, in order.
    span : int
        Window width in ordinates; must be odd and at least 1.

    Returns
    -------
    RichResult
        ``omega``, ``smoothed``, ``raw``, ``span``, ``equivalent_df``,
        ``n``, ``method``.
    """
    span = int(span)
    if span < 1 or span % 2 == 0:
        raise ValueError("`span` must be an odd positive integer")
    base = spectral_density(y)
    omega = base["omega"]
    raw = base["periodogram"]
    m = len(raw)
    if span > m:
        raise ValueError("`span` (%d) exceeds the number of Fourier "
                         "ordinates (%d)" % (span, m))
    half = span // 2
    sm = []
    for k in range(m):
        sm.append(fsum([raw[(k + t) % m] for t in range(-half, half + 1)])
                  / span)

    return RichResult(payload={
        "omega": omega,
        "smoothed": sm,
        "raw": raw,
        "span": span,
        "equivalent_df": 2.0 * span,
        "circular_window": True,
        "n": base["n"],
        "method": ("Daniell-smoothed periodogram; periodogram from "
                   "Schabenberger & Gotway (2005) eq (4.57), the smoother "
                   "is NOT in that book (see Bloomfield 2000, Ch. 8)"),
    })


def cheatsheet():
    return "specsm: Daniell-smoothed periodogram"


# compact alias per ledger/NAMING.md
smpgram = spectral_smoothed
