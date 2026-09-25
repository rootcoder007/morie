# morie.fn -- function file (rootcoder007/morie)
"""Reassigned spectrogram for sharper TF representation."""

from __future__ import annotations

from . import _array_core as np
from ._signal_core import stft as _stft

from ._containers import DescriptiveResult

_QUOTE = "Let the past die. Kill it, if you have to."


def reassigned_stft(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> DescriptiveResult:
    """Reassigned Short-Time Fourier Transform spectrogram.

    Sharpens time-frequency localization by reassigning each spectrogram
    coefficient to its center of gravity in the TF plane.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).
    nperseg : int
        Segment length (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``spectrogram``, ``frequencies``, ``times``,
        ``reassigned_times``, ``reassigned_freqs``.

    Notes
    -----
    Auger and Flandrin (1995): with X_h the STFT under the (periodic
    Hann) window h, X_th under the time-weighted window tau h(tau) (tau
    measured from the window centre) and X_dh under h'(tau),

        t_hat = t + Re(X_th / X_h),   f_hat = f - Im(X_dh / X_h) / (2 pi).

    Frames are the same as the plain STFT here (half overlap, no
    padding). An impulse at t0 is reassigned to t0 in every bin and a
    complex tone at f0 to f0.

    References
    ----------
    Auger, F. & Flandrin, P. (1995). Improving the readability of
    time-frequency and time-scale representations by the reassignment
    method. IEEE Trans. Signal Processing 43(5), 1068-1089.
    """
    import cmath
    import math

    xs = [complex(v) for v in np.asarray(x).ravel().tolist()]
    n = len(xs)
    L = int(min(nperseg, n))
    step = L - L // 2
    fs = float(fs)
    h = [0.5 - 0.5 * math.cos(2.0 * math.pi * k / L) for k in range(L)]
    tau = [(k - L / 2.0) / fs for k in range(L)]
    th = [t_ * w for t_, w in zip(tau, h)]
    dh = [math.pi * fs / L * math.sin(2.0 * math.pi * k / L) for k in range(L)]
    nf = L // 2 + 1

    def dft(seg):
        return [sum(seg[k] * cmath.exp(-2j * math.pi * m * k / L) for k in range(L))
                for m in range(nf)]
    freqs = [m * fs / L for m in range(nf)]
    times, P, RT, RF = [], [], [], []
    start = 0
    while start + L <= n:
        seg = xs[start:start + L]
        Xh = dft([v * w for v, w in zip(seg, h)])
        Xt = dft([v * w for v, w in zip(seg, th)])
        Xd = dft([v * w for v, w in zip(seg, dh)])
        tc = (start + L / 2.0) / fs
        times.append(tc)
        peak = max(abs(z) for z in Xh) or 1.0
        colp, colt, colf = [], [], []
        for m in range(nf):
            z = Xh[m]
            colp.append(abs(z) ** 2 / sum(h) ** 2)
            if abs(z) > 1e-10 * peak:
                colt.append(tc + (Xt[m] / z).real)
                colf.append(freqs[m] - (Xd[m] / z).imag / (2.0 * math.pi))
            else:
                colt.append(tc)
                colf.append(freqs[m])
        P.append(colp)
        RT.append(colt)
        RF.append(colf)
        start += step
    T = lambda M: np.array([[M[j][i] for j in range(len(M))] for i in range(nf)])
    power = T(P)
    return DescriptiveResult(
        name="reassigned_stft",
        value=float(nf),
        extra={
            "spectrogram": power,
            "frequencies": np.array(freqs),
            "times": np.array(times),
            "reassigned_times": T(RT),
            "reassigned_freqs": T(RF),
        },
    )


reasn = reassigned_stft


def cheatsheet() -> str:
    return "reassigned_stft({}) -> Reassigned spectrogram for sharper TF representation."


# compact alias per ledger/NAMING.md
reassignedstft = reassigned_stft
