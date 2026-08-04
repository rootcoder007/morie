# morie.fn -- function file (rootcoder007/morie)
"""Cross-spectrum between two series."""

from math import atan2, fsum, pi, sqrt

from ._richresult import RichResult
from ._spx import dft, mean, vec

__all__ = [
    "cross_spectrum",
    "crossspec",
    "crossspectrum",
]


def cross_spectrum(x, y):
    """Raw cross-periodogram of two equal-length records.

    NOT IN SCHABENBERGER & GOTWAY. A fixed-string search for
    "cross-spectr" in the book finds one bibliography entry (Renshaw's
    Environmetrics paper) and no method. The definition used is the
    standard one of Brillinger, D. R. (2001), *Time Series: Data Analysis
    and Theory*, SIAM Classics edn, Ch. 7 -- named from the general
    literature and NOT verified against a PDF in this corpus.

    At the Fourier frequencies w_j = 2 pi j / n,

        S_xy(w_j) = X(w_j) conj(Y(w_j)) / (2 pi n),

    with X and Y the discrete transforms of the MEAN-REMOVED records. The
    real part is the co-spectrum, the negated imaginary part the
    quadrature spectrum; ``amplitude`` is |S_xy| and ``phase`` is
    arg(S_xy) in radians, positive phase meaning x leads y.

    The mean is removed first. Leaving it in puts the entire record mean
    into the w = 0 ordinate and, worse, leaks it across the whole spectrum
    once the record is not an exact number of periods long. The zero
    frequency is dropped for the same reason it is dropped from the
    periodogram in ``spectral_density``.

    This is a RAW cross-periodogram: it is not consistent, exactly as the
    raw periodogram is not. Smooth it before interpreting a single
    ordinate.

    Parameters
    ----------
    x, y : (n,) array-like
        Equal-length records.

    Returns
    -------
    RichResult
        ``omega``, ``cospectrum``, ``quadrature``, ``amplitude``,
        ``phase``, ``n``, ``method``.
    """
    xv = vec(x, "x")
    yv = vec(y, "y")
    n = len(xv)
    if len(yv) != n:
        raise ValueError("`x` and `y` must have the same length")
    if n < 4:
        raise ValueError("at least 4 observations are needed")
    mx = mean(xv)
    my = mean(yv)
    dx = [t - mx for t in xv]
    dy = [t - my for t in yv]
    if fsum([t * t for t in dx]) <= 0 or fsum([t * t for t in dy]) <= 0:
        raise ValueError("`x` and `y` must not be constant")

    xr, xi = dft(dx)
    yr, yi = dft(dy)
    scale = 2.0 * pi * n
    ks = [k for k in range(1, n // 2 + 1)]
    omega = [2.0 * pi * k / n for k in ks]
    co = []
    qu = []
    amp = []
    ph = []
    for k in ks:
        re = (xr[k] * yr[k] + xi[k] * yi[k]) / scale
        im = (xi[k] * yr[k] - xr[k] * yi[k]) / scale
        co.append(re)
        qu.append(-im)
        amp.append(sqrt(re * re + im * im))
        ph.append(atan2(im, re))

    return RichResult(payload={
        "omega": omega,
        "cospectrum": co,
        "quadrature": qu,
        "amplitude": amp,
        "phase": ph,
        "means_removed": True,
        "raw_not_consistent": True,
        "n": n,
        "method": ("Raw cross-periodogram (Brillinger 2001, Ch. 7); NOT "
                   "in Schabenberger & Gotway"),
    })


def cheatsheet():
    return "speccs: raw cross-periodogram of two records"


# compact alias per ledger/NAMING.md
crossspec = cross_spectrum
crossspectrum = cross_spectrum
