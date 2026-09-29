"""Spectrogram (STFT magnitude).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 5.
"""

from __future__ import annotations

import cmath
import math

from ._containers import DescriptiveResult

__all__ = ["spcgm"]


def _window(name, n):
    if name in ("hann", "hanning"):
        return [0.5 - 0.5 * math.cos(2.0 * math.pi * k / n) for k in range(n)]
    if name == "hamming":
        return [0.54 - 0.46 * math.cos(2.0 * math.pi * k / n) for k in range(n)]
    if name in ("boxcar", "rectangular"):
        return [1.0] * n
    raise ValueError("window must be 'hann', 'hamming' or 'boxcar'")


def spcgm(
    x,
    fs: float = 1.0,
    *,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
) -> DescriptiveResult:
    r"""Spectrogram: one-sided power spectral density of windowed, overlapping STFT segments.

    Segments of ``nperseg`` samples advance by ``nperseg - noverlap``
    (``noverlap`` defaults to ``nperseg // 8``); each is mean-detrended,
    multiplied by the periodic window ``w`` and transformed with an
    ``nfft``-point DFT (zero-padded; default ``nperseg``). The density is
    ``|X_k|^2 / (fs sum w^2)``, doubled at the interior frequencies of the
    one-sided spectrum, at frequencies ``k fs / nfft`` and segment-centre
    times ``(start + nperseg/2) / fs``: the conventions of
    ``scipy.signal.spectrogram(scaling="density", mode="psd")`` (Rangayyan
    and Krishnan 2024, ch. 5; Allen and Rabiner 1977). ``value`` is the
    ``Sxx`` matrix (frequency by time).

    References
    ----------
    Allen, J. B. and Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. *Proceedings of the IEEE* 65, 1558-1564.

    Examples
    --------
    >>> import math
    >>> x = [math.sin(2 * math.pi * 0.25 * n) for n in range(64)]
    >>> r = spcgm(x, nperseg=16, noverlap=8)
    >>> max(range(9), key=lambda k: r.value[k][0])
    4
    """
    xs = [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]
    n = len(xs)
    seg = min(int(nperseg), n)
    ov = seg // 8 if noverlap is None else int(noverlap)
    L = seg if nfft is None else int(nfft)
    if seg > L:
        raise ValueError("nfft must be at least nperseg")
    w = _window(window, seg)
    scale = 1.0 / (fs * math.fsum(t * t for t in w))
    nf = L // 2 + 1
    cols, times = [], []
    start = 0
    while start + seg <= n:
        s = xs[start : start + seg]
        m = math.fsum(s) / seg
        s = [(v - m) * t for v, t in zip(s, w)]
        p = []
        for k in range(nf):
            z = sum(s[j] * cmath.exp(-2j * math.pi * k * j / L) for j in range(seg))
            p.append(abs(z) ** 2 * scale)
        for k in range(1, nf - (1 if L % 2 == 0 else 0)):
            p[k] *= 2.0
        cols.append(p)
        times.append((start + seg / 2.0) / fs)
        start += seg - ov
    freqs = [k * fs / L for k in range(nf)]
    Sxx = [[c[k] for c in cols] for k in range(nf)]
    return DescriptiveResult(name="spcgm", value=Sxx, extra={"frequencies": freqs, "times": times, "Sxx": Sxx})


def cheatsheet() -> str:
    return "spcgm(x, fs, nperseg=256) -> one-sided PSD spectrogram of windowed STFT segments (scipy conventions)."
