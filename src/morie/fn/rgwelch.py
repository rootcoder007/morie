# morie.fn -- function file (rootcoder007/morie)
"""Welch power spectral density."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_welch_psd"]


def rangayyan_welch_psd(x, fs=1.0, nperseg=None, noverlap=None, window="hann"):
    r"""Welch's averaged periodogram (Rangayyan Ch. 3):

    .. math:: P_W(f) = \frac{1}{KU}\sum_{k=1}^{K} |W_k(f)|^2,
              \qquad U = \frac1N \sum_n w^2[n],

    with U the window power normalisation, WITHOUT which the estimate
    is biased low by the window's energy loss. Averaging K segments
    cuts the variance by roughly K at the cost of resolution -- the
    bias-variance trade the periodogram cannot make.

    Parameters
    ----------
    x : array-like
        Signal.
    fs : float, default 1.0
        Sampling frequency.
    nperseg : int, optional
        Segment length; N // 8 by default.
    noverlap : int, optional
        Overlap; half the segment by default.
    window : {"hann", "hamming", "boxcar"}
        Window type.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``psd``, ``n_segments``, ``U``, ``nperseg``,
        ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Welch's method).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    seg = max(8, N // 8) if nperseg is None else int(nperseg)
    if not 2 <= seg <= N:
        raise ValueError(f"nperseg must lie in 2..{N}, got {seg}.")
    ov = seg // 2 if noverlap is None else int(noverlap)
    if not 0 <= ov < seg:
        raise ValueError(f"noverlap must lie in 0..{seg - 1}, got {ov}.")
    if window == "hann":
        w = np.hanning(seg)
    elif window == "hamming":
        w = np.hamming(seg)
    elif window == "boxcar":
        w = np.ones(seg)
    else:
        raise ValueError("window must be 'hann', 'hamming' or 'boxcar'.")
    U = float(np.mean(w**2))
    step = seg - ov
    starts = range(0, N - seg + 1, step)
    acc, K = None, 0
    for s in starts:
        W = np.fft.rfft(x[s : s + seg] * w)
        p = np.abs(W) ** 2
        acc = p if acc is None else acc + p
        K += 1
    if K == 0:
        raise ValueError("no complete segments; reduce nperseg.")
    psd = acc / (K * U * seg)
    return RichResult(payload={"freqs": np.fft.rfftfreq(seg, d=1.0 / fs), "psd": psd,
                               "n_segments": K, "U": U, "nperseg": seg, "fs": fs,
                               "method": "Welch averaged periodogram with window power U"})


def cheatsheet():
    return "rgwelch: U normalisation is mandatory or the PSD is biased low"
