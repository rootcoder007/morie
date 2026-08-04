# morie.fn -- function file (rootcoder007/morie)
"""Magnitude-squared coherence by Welch averaging."""

from math import cos, fsum, pi, sqrt

from ._richresult import RichResult
from ._spx import dft, mean, vec

__all__ = [
    "coherence",
    "mscoh",
]


def coherence(x, y, nperseg=None, overlap=0.5):
    """Magnitude-squared coherence between two records.

    NOT IN SCHABENBERGER & GOTWAY -- a fixed-string search of the book for
    "coherence" returns nothing. The definition is the standard

        C_xy(w) = |S_xy(w)|^2 / {S_xx(w) S_yy(w)},

    from Bendat, J. S. & Piersol, A. G. (2010), *Random Data: Analysis and
    Measurement Procedures*, 4th edn, Wiley, Ch. 5 -- named from the
    general literature and NOT verified against a PDF in this corpus.

    AVERAGING IS NOT OPTIONAL. On a single segment |S_xy|^2 = S_xx S_yy
    identically, so the coherence of any two records whatsoever is exactly
    1 at every frequency. That is the classic way to produce a beautiful
    and completely meaningless coherence plot. The estimate here therefore
    averages the three spectra over Welch segments, and a run that yields
    fewer than two segments raises rather than returning ones.

    Each segment is mean-removed and Hann-windowed; the mean is removed
    per segment, not once for the whole record, because a slow trend
    otherwise reappears as a large low-frequency term in every segment.

    Parameters
    ----------
    x, y : (n,) array-like
        Equal-length records.
    nperseg : int, optional
        Segment length; defaults to n // 4 rounded down, minimum 8.
    overlap : float
        Fractional overlap between segments, in [0, 1).

    Returns
    -------
    RichResult
        ``omega``, ``coherence``, ``sxx``, ``syy``, ``n_segments``,
        ``nperseg``, ``n``, ``method``.
    """
    xv = vec(x, "x")
    yv = vec(y, "y")
    n = len(xv)
    if len(yv) != n:
        raise ValueError("`x` and `y` must have the same length")
    if nperseg is None:
        m = max(8, n // 4)
    else:
        m = int(nperseg)
    if m < 8:
        raise ValueError("`nperseg` must be at least 8")
    if m > n:
        raise ValueError("`nperseg` (%d) exceeds the record length (%d)"
                         % (m, n))
    overlap = float(overlap)
    if not 0.0 <= overlap < 1.0:
        raise ValueError("`overlap` must lie in [0, 1)")
    step = max(1, int(round(m * (1.0 - overlap))))
    starts = list(range(0, n - m + 1, step))
    if len(starts) < 2:
        raise ValueError("fewer than 2 segments: coherence would be "
                         "identically 1 and would mean nothing; shorten "
                         "`nperseg` or lengthen the records")

    win = [0.5 - 0.5 * cos(2.0 * pi * t / (m - 1.0)) for t in range(m)]
    ks = [k for k in range(1, m // 2 + 1)]
    sxx = [0.0] * len(ks)
    syy = [0.0] * len(ks)
    cre = [0.0] * len(ks)
    cim = [0.0] * len(ks)
    for s in starts:
        sx = xv[s:s + m]
        sy = yv[s:s + m]
        mx = mean(sx)
        my = mean(sy)
        wx = [(sx[t] - mx) * win[t] for t in range(m)]
        wy = [(sy[t] - my) * win[t] for t in range(m)]
        xr, xi = dft(wx)
        yr, yi = dft(wy)
        for idx in range(len(ks)):
            k = ks[idx]
            sxx[idx] = sxx[idx] + xr[k] * xr[k] + xi[k] * xi[k]
            syy[idx] = syy[idx] + yr[k] * yr[k] + yi[k] * yi[k]
            cre[idx] = cre[idx] + xr[k] * yr[k] + xi[k] * yi[k]
            cim[idx] = cim[idx] + xi[k] * yr[k] - xr[k] * yi[k]

    nseg = float(len(starts))
    coh = []
    for idx in range(len(ks)):
        den = sxx[idx] * syy[idx]
        if den <= 0:
            coh.append(float("nan"))
        else:
            coh.append((cre[idx] ** 2 + cim[idx] ** 2) / den)

    return RichResult(payload={
        "omega": [2.0 * pi * k / m for k in ks],
        "coherence": coh,
        "sxx": [t / nseg for t in sxx],
        "syy": [t / nseg for t in syy],
        "n_segments": nseg,
        "nperseg": float(m),
        "step": float(step),
        "single_segment_coherence_is_identically_one": True,
        "n": n,
        "method": ("Magnitude-squared coherence by Welch averaging "
                   "(Bendat & Piersol 2010, Ch. 5); NOT in Schabenberger "
                   "& Gotway"),
    })


def cheatsheet():
    return "speccoh: magnitude-squared coherence (Welch)"


# compact alias per ledger/NAMING.md
mscoh = coherence
