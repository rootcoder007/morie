# morie.fn -- function file (rootcoder007/morie)
"""Bartlett's averaged periodogram."""

import math as _math

from ._richresult import RichResult

__all__ = ["rangayyan_bartlett_psd", "bartlettpsd"]


def _dft_power(seg):
    """|DFT|^2 at each bin, direct evaluation.

    O(M^2) rather than an FFT: correct at any M, no padding to a power of
    two, and these segments are short.  ponytail: swap in the radix-2
    path in _signal_core if a caller ever needs long segments.
    """
    m = len(seg)
    out = []
    for k in range(m // 2 + 1):
        re = im = 0.0
        for n, v in enumerate(seg):
            ang = -2.0 * _math.pi * k * n / m
            re += v * _math.cos(ang)
            im += v * _math.sin(ang)
        out.append(re * re + im * im)
    return out


def rangayyan_bartlett_psd(x, fs=1.0, n_segments=None, segment_length=None):
    r"""Bartlett's method: average the periodograms of disjoint segments.

    Rangayyan eqs. (6.14)-(6.16).  The record is split into :math:`K`
    non-overlapping segments of :math:`M` samples,

    .. math:: S_i(\omega) = \frac{1}{M}
              \left| \sum_{n=0}^{M-1} x_i(n) e^{-j\omega n} \right|^2

    and the estimate is their sample mean,
    :math:`S_B(\omega) = \frac1K \sum_i S_i(\omega)`.

    Averaging :math:`K` independent periodograms divides the variance by
    :math:`K` while multiplying the resolution bandwidth by the same
    factor -- the trade the method exists to make.  Segments are DISJOINT
    here, as the book specifies; Welch's overlapping variant is a
    different estimator and is not what this citation promises.

    Parameters
    ----------
    x : sequence
        The signal.
    fs : float
        Sampling rate, for the returned frequency axis.
    n_segments, segment_length : int, optional
        Give exactly one.  Trailing samples that do not fill a whole
        segment are dropped, as the segmentation in eq. (6.14) requires.

    Returns
    -------
    RichResult
        ``psd``, ``freqs``, ``n_segments``, ``segment_length``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, eqs. (6.14)-(6.16), after Oppenheim & Schafer.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two samples")
    if (n_segments is None) == (segment_length is None):
        raise ValueError("give exactly one of n_segments, segment_length")
    if n_segments is not None:
        k = int(n_segments)
        if k < 1:
            raise ValueError("n_segments must be positive")
        m = n // k
    else:
        m = int(segment_length)
        if m < 2:
            raise ValueError("segment_length must be at least 2")
        k = n // m
    if k < 1 or m < 2:
        raise ValueError("segmentation leaves no usable segment")

    acc = None
    for i in range(k):
        seg = xs[i * m:(i + 1) * m]
        p = [v / m for v in _dft_power(seg)]
        acc = p if acc is None else [a + b for a, b in zip(acc, p)]
    psd = [v / k for v in acc]
    freqs = [j * fs / m for j in range(len(psd))]
    return RichResult(
        title="Bartlett averaged periodogram (Rangayyan eq. 6.16)",
        summary_lines=[("segments", k), ("segment length", m)],
        payload={"psd": psd, "freqs": freqs, "n_segments": k,
                 "segment_length": m,
                 "method": "Rangayyan (2024) eqs. (6.14)-(6.16)"},
    )


bartlettpsd = rangayyan_bartlett_psd


def cheatsheet():
    return "bartlettpsd: mean of K disjoint-segment periodograms, eq 6.16"
