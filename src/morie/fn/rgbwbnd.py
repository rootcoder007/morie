# morie.fn -- function file (rootcoder007/morie)
"""Spectral bandwidth."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_bandwidth"]


def rangayyan_bandwidth(psd, freqs, criterion="3dB"):
    r"""Spectral bandwidth by two criteria (Rangayyan Ch. 3):

    - ``"3dB"``: the span of frequencies where
      :math:`S(f) \ge S_{\max}/2` (half power, i.e. -3 dB);
    - ``"99"``: the narrowest band from the peak containing 99% of the
      total power.

    The two answer different questions and can differ by an order of
    magnitude on a peaky spectrum, so the criterion is explicit rather
    than defaulted silently.

    Parameters
    ----------
    psd : array-like
        Power spectral density.
    freqs : array-like
        Matching frequencies.
    criterion : {"3dB", "99"}
        Which bandwidth to report.

    Returns
    -------
    RichResult
        keys: ``bandwidth``, ``f_low``, ``f_high``, ``f_peak``,
        ``criterion``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (bandwidth measures).
    """
    S = np.asarray(psd, dtype=float).ravel()
    f = np.asarray(freqs, dtype=float).ravel()
    if S.size != f.size:
        raise ValueError("psd and freqs must have the same length.")
    if S.size < 2:
        raise ValueError("need at least 2 spectral points.")
    if np.any(S < 0):
        raise ValueError("a power spectral density cannot be negative.")
    ipk = int(np.argmax(S))
    if criterion == "3dB":
        thr = S[ipk] / 2.0
        above = np.flatnonzero(S >= thr)
        lo, hi = float(f[above[0]]), float(f[above[-1]])
    elif criterion == "99":
        total = float(S.sum())
        if total <= 0:
            raise ValueError("spectrum has zero total power.")
        lo_i = hi_i = ipk
        acc = S[ipk]
        while acc < 0.99 * total and (lo_i > 0 or hi_i < S.size - 1):
            left = S[lo_i - 1] if lo_i > 0 else -np.inf
            right = S[hi_i + 1] if hi_i < S.size - 1 else -np.inf
            if left >= right:
                lo_i -= 1
                acc += S[lo_i]
            else:
                hi_i += 1
                acc += S[hi_i]
        lo, hi = float(f[lo_i]), float(f[hi_i])
    else:
        raise ValueError("criterion must be '3dB' or '99'.")
    return RichResult(payload={"bandwidth": hi - lo, "f_low": lo, "f_high": hi,
                               "f_peak": float(f[ipk]), "criterion": criterion,
                               "method": f"{criterion} bandwidth about the spectral peak"})


def cheatsheet():
    return "rgbwbnd: 3dB and 99% answer different questions; criterion is explicit"
