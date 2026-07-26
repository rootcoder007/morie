# morie.fn -- function file (rootcoder007/morie)
"""IIR Butterworth filter -- Rangayyan & Krishnan Sec 3.7.1 / 3.7.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_iir_filter"]


def rangayyan_iir_filter(x, cutoff, order=4, fs=1.0, btype="low"):
    """Butterworth IIR filter via SOS + zero-phase ``filtfilt``.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float or (float, float)
        Cutoff(s) in Hz.
    order : int
        Filter order (default 4).
    fs : float
        Sampling rate (Hz).
    btype : {"low","high","bandpass","bandstop"}

    Returns
    -------
    RichResult with keys ``signal``, ``sos``, ``order``, ``cutoff``, ``fs``, ``btype``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 3.7.1 "Removal of high-frequency
        noise: Butterworth lowpass filters", p.154, and Sec 3.7.2 "Removal
        of low-frequency noise: Butterworth highpass filters", p.161
        (Sec 3.7 "Frequency-domain Filters", p.153).
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    # scipy.signal.butter raises "Digital filter critical frequencies must be
    # 0 < Wn < 1" from deep inside iirfilter, naming neither `cutoff` nor `fs`.
    # With the default fs=1.0 any cutoff in Hz above 0.5 trips it, so the
    # commonest caller mistake -- passing Hz while leaving fs at its default --
    # surfaced as an opaque message about a variable the caller never set.
    # Validate in the caller's own units, as rgfir does.
    cuts = list(cutoff) if isinstance(cutoff, (list, tuple, np.ndarray)) else [cutoff]
    for c in cuts:
        if not (0.0 < float(c) < nyq):
            raise ValueError(
                f"cutoff must satisfy 0 < cutoff < fs/2 (Nyquist); "
                f"got cutoff={cutoff!r} with fs={fs!r} (Nyquist={nyq!r})"
            )
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        wn = [float(c) / nyq for c in cutoff]
        if not wn[0] < wn[1]:
            raise ValueError(
                f"band cutoffs must be increasing, got cutoff={cutoff!r}"
            )
    else:
        wn = float(cutoff) / nyq
    sos = butter(int(order), wn, btype=btype, output="sos")
    y = sosfiltfilt(sos, x)
    res = RichResult(
        title="Butterworth IIR filter",
        summary_lines=[
            ("Order", int(order)),
            ("Type", btype),
            ("Cutoff (Hz)", cutoff),
            ("Fs (Hz)", float(fs)),
        ],
        interpretation=f"Zero-phase Butterworth {btype} filter, order {order}.",
        payload={
            "signal": y,
            "sos": sos,
            "order": int(order),
            "cutoff": cutoff,
            "fs": float(fs),
            "btype": btype,
        },
    )
    return with_describe_pointer(res, "rgiir")


# CANONICAL TEST
# >>> fs=100.0; t=np.arange(100)/fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*40*t)
# >>> r = rangayyan_iir_filter(x, cutoff=10, order=4, fs=fs, btype="low")
# >>> r["signal"].shape == x.shape
# True


def cheatsheet():
    return "rgiir: Butterworth IIR filter (zero-phase) -- Rangayyan & Krishnan Sec 3.7.1"
