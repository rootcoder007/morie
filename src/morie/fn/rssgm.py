# morie.fn -- function file (rootcoder007/morie)
"""Reassigned spectrogram (time-frequency reassignment)."""

from __future__ import annotations

import cmath
import math

from . import _array_core as np
from ._containers import DescriptiveResult

_QUOTE = "Knowledge itself is power. -- Francis Bacon"


def _dft(seg, nf):
    L = len(seg)
    out = []
    for m in range(nf):
        s = 0j
        for k in range(L):
            s += seg[k] * cmath.exp(-2j * math.pi * m * k / L)
        out.append(s)
    return out


def reassigned_spectrogram(
    x,
    fs: float = 1.0,
    window: int = 256,
    hop: int = 128,
) -> DescriptiveResult:
    """Reassigned spectrogram: each STFT cell moved to its centre of gravity.

    With ``X_h`` the short-time Fourier transform under the periodic Hann
    window ``h``, ``X_th`` the transform under the time-weighted window
    ``tau h(tau)`` (``tau`` measured in seconds from the window centre) and
    ``X_dh`` under the window derivative ``h'(tau)``, Auger and Flandrin
    (1995) reassign the cell ``(t, f)`` to

        t_hat = t + Re(X_th / X_h),   f_hat = f - Im(X_dh / X_h) / (2 pi),

    the local group delay and instantaneous frequency. In continuous time a
    complex tone at ``f0`` is reassigned exactly to ``f0``; with the sampled
    window derivative used here the error is of order 1e-5 of ``fs``. (Finite differences of the STFT phase between neighbouring
    frames and bins -- what this function used to compute, with the time
    and frequency derivatives interchanged -- are not the reassignment
    operators.) Frames start at sample 0 and advance by ``hop`` without
    padding; cells with ``|X_h|`` below ``1e-10`` of the frame peak stay
    where they are.

    Parameters
    ----------
    x : array-like
        1-D input signal (real or complex).
    fs : float
        Sampling frequency.
    window : int
        Window length in samples (capped at the signal length).
    hop : int
        Hop size in samples.

    Returns
    -------
    DescriptiveResult
        ``value`` = largest magnitude; ``extra``: ``magnitude`` (|X_h|,
        frequencies x frames), ``frequencies``, ``times`` (frame centres),
        ``t_reassigned``, ``f_reassigned``.

    References
    ----------
    Auger, F. and Flandrin, P. (1995). Improving the readability of
    time-frequency and time-scale representations by the reassignment
    method. IEEE Transactions on Signal Processing 43(5), 1068-1089.
    Kodera, K., Gendrin, R. and de Villedary, C. (1978). Analysis of
    time-varying signals with small BT values. IEEE Trans. ASSP 26, 64-76.

    Examples
    --------
    >>> import math
    >>> tone = [math.cos(2 * math.pi * 0.25 * k) for k in range(64)]
    >>> r = reassigned_spectrogram(tone, fs=1.0, window=16, hop=8)
    >>> round(float(r.extra["f_reassigned"][4][0]), 12)
    0.25
    """
    xs = [complex(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]
    n = len(xs)
    L = int(min(int(window), n))
    hop = int(hop)
    if L < 2 or hop < 1:
        raise ValueError("need window >= 2 samples and hop >= 1")
    fs = float(fs)
    h = [0.5 - 0.5 * math.cos(2.0 * math.pi * k / L) for k in range(L)]
    th = [(k - L / 2.0) / fs * h[k] for k in range(L)]
    dh = [math.pi * fs / L * math.sin(2.0 * math.pi * k / L) for k in range(L)]
    nf = L // 2 + 1
    freqs = [m * fs / L for m in range(nf)]
    times, M, RT, RF = [], [], [], []
    start = 0
    while start + L <= n:
        seg = xs[start : start + L]
        Xh = _dft([v * w for v, w in zip(seg, h)], nf)
        Xt = _dft([v * w for v, w in zip(seg, th)], nf)
        Xd = _dft([v * w for v, w in zip(seg, dh)], nf)
        tc = (start + L / 2.0) / fs
        times.append(tc)
        peak = max(abs(z) for z in Xh) or 1.0
        cm, ct, cf = [], [], []
        for m in range(nf):
            z = Xh[m]
            cm.append(abs(z))
            if abs(z) > 1e-10 * peak:
                ct.append(tc + (Xt[m] / z).real)
                cf.append(freqs[m] - (Xd[m] / z).imag / (2.0 * math.pi))
            else:
                ct.append(tc)
                cf.append(freqs[m])
        M.append(cm)
        RT.append(ct)
        RF.append(cf)
        start += hop

    def T(A):
        return np.array([[A[j][i] for j in range(len(A))] for i in range(nf)])

    magnitude = T(M)
    return DescriptiveResult(
        name="reassigned_spectrogram",
        value=float(max(max(c) for c in M)),
        extra={
            "magnitude": magnitude,
            "frequencies": np.array(freqs),
            "times": np.array(times),
            "t_reassigned": T(RT),
            "f_reassigned": T(RF),
        },
    )


rssgm = reassigned_spectrogram


def cheatsheet() -> str:
    return "reassigned_spectrogram({}) -> Auger-Flandrin reassigned spectrogram (t + Re(Xth/Xh), f - Im(Xdh/Xh)/2pi)."
