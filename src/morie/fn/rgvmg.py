# morie.fn -- function file (rootcoder007/morie)
"""Vibromyogram (VMG) signal characterization (Rangayyan Sects. 1.2.15, 2.2.6, 5.11).

Book pages read from the typeset PDF: Rangayyan and Krishnan,
*Biomedical Signal Analysis*, 3rd ed., Wiley-IEEE Press, 2024.

Section 1.2.15 "The vibromyogram (VMG)", p. 52: "The VMG is the direct
mechanical manifestation of contraction of a skeletal muscle and is a
vibration signal ... The frequency and intensity of the VMG have been
shown to vary in direct proportion to the contraction level."

Section 2.2.6 "The EMG and VMG", p. 77: "it has been shown that the
RMS and mean frequency parameters of the VMG signal increase with muscle
force output, in patterns that parallel those of the EMG."

Section 5.11, p. 295: "The VMG signals were filtered to the bandwidth
3 - 100 Hz ... The VMG and EMG signals were sampled at 250 Hz and
1,000 Hz, respectively."

The two parameters the book names -- intensity and frequency -- are
therefore what this function reports: the RMS value of equation (3.9)
and the mean frequency of the power spectrum, together with the median
frequency, the zero-crossing rate, and the fraction of the power that
falls in the 3-100 Hz VMG band of Section 5.11.

The power spectrum is the periodogram computed by a direct discrete
Fourier transform rather than by a library FFT, so that both language
arms evaluate literally the same sums in the same order.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["rangayyan_vmg"]

# Section 5.11, p. 295: the VMG passband used in the Zhang et al. study
_VMG_LOW = 3.0
_VMG_HIGH = 100.0


def _aslist(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def rangayyan_vmg(vmg, fs, band=None):
    """Intensity and frequency parameters of a VMG record.

    Parameters
    ----------
    vmg : array-like
        The vibromyogram.
    fs : float
        Sampling rate in Hz; 250 Hz in the book's experiment.
    band : (float, float), optional
        The VMG passband whose power fraction is reported; the 3-100 Hz
        band of Section 5.11 by default.

    Returns
    -------
    estimate : the RMS value, the book's intensity parameter
    rms      : the same, equation (3.9)
    mean_frequency : the power-weighted mean frequency in Hz
    median_frequency : the frequency splitting the power in half
    zcr      : zero crossings per second
    band_power_fraction : power in the band divided by total power
    freqs, psd : the one-sided periodogram
    """
    x = _aslist(vmg)
    N = len(x)
    if N < 2:
        raise ValueError("rangayyan_vmg: need at least two samples")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("rangayyan_vmg: fs must be positive")
    lo, hi = (_VMG_LOW, _VMG_HIGH) if band is None else (float(band[0]), float(band[1]))
    if not hi > lo or lo < 0.0:
        raise ValueError("rangayyan_vmg: the band must be an increasing nonnegative pair")

    # eq (3.9): RMS = sqrt((1/N) sum x^2), divisor N
    ms = 0.0
    for v in x:
        ms += v * v
    ms /= N
    rms = sqrt(ms)

    # zero-crossing rate, Section 5.6.2
    crossings = 0
    prev = 1.0 if x[0] >= 0.0 else -1.0
    for i in range(1, N):
        s = 1.0 if x[i] >= 0.0 else -1.0
        if s != prev:
            crossings += 1
        prev = s
    zcr = crossings / (N - 1) * fs

    # one-sided periodogram by a direct DFT
    half = N // 2
    freqs = []
    psd = []
    for j in range(half + 1):
        re = 0.0
        im = 0.0
        w = 2.0 * pi * j / N
        for n in range(N):
            a = w * n
            re += x[n] * cos(a)
            im -= x[n] * sin(a)
        freqs.append(j * fs / N)
        psd.append((re * re + im * im) / N)

    total = 0.0
    for p in psd:
        total += p
    if total <= 0.0:
        mean_f = float("nan")
        med_f = float("nan")
        frac = float("nan")
    else:
        acc = 0.0
        for j in range(len(psd)):
            acc += freqs[j] * psd[j]
        mean_f = acc / total
        run = 0.0
        med_f = freqs[-1]
        for j in range(len(psd)):
            run += psd[j]
            if run >= 0.5 * total:
                med_f = freqs[j]
                break
        inband = 0.0
        for j in range(len(psd)):
            if lo <= freqs[j] <= hi:
                inband += psd[j]
        frac = inband / total

    return RichResult(
        title="Vibromyogram characterization",
        summary_lines=[("samples", N), ("RMS", rms), ("mean frequency (Hz)", mean_f)],
        payload={
            "estimate": rms,
            "rms": rms,
            "ms": ms,
            "mean_frequency": mean_f,
            "median_frequency": med_f,
            "zcr": zcr,
            "crossings": crossings,
            "band_power_fraction": frac,
            "band": [lo, hi],
            "freqs": freqs,
            "psd": psd,
            "total_power": total,
            "n": N,
            "fs": fs,
            "method": "Rangayyan (2024) Sects. 1.2.15 p.52, 2.2.6 p.77 and 5.11 p.295: RMS eq. (3.9) and mean frequency of the periodogram",
        },
    )


def cheatsheet():
    return "rgvmg: VMG intensity and frequency parameters, Rangayyan Sects. 1.2.15/2.2.6/5.11"


# compact alias per ledger/NAMING.md
rangayyanvmg = rangayyan_vmg
