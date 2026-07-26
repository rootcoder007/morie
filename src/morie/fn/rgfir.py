# morie.fn -- function file (rootcoder007/morie)
"""FIR filter design (windowed sinc) -- see rangayyan_fir_filter for sources."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_fir_filter"]


def rangayyan_fir_filter(x, cutoff, order=51, fs=1.0, window="hamming"):
    """Windowed-sinc FIR lowpass filter.

    Designs a linear-phase FIR lowpass filter of length ``order`` using
    the windowed-sinc method::

        h[n] = w[n] * 2*fc * sinc(2*fc * (n - M/2))

    with ``fc = cutoff / (fs/2)`` -- normalised to Nyquist, matching
    ``scipy.signal.firwin`` -- and applies it to ``x`` via zero-phase
    forward-backward convolution (``filtfilt``).

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float
        Cutoff frequency in the same units as ``fs``. Must satisfy
        ``0 < cutoff < fs/2``.
    order : int
        Number of taps (odd recommended). Default 51.
    fs : float
        Sampling frequency (Hz). Default 1.0.
    window : str
        Window function name (``hamming``, ``hann``, ``blackman``, ``rect``).

    Returns
    -------
    RichResult with keys ``signal``, ``taps``, ``order``, ``cutoff``, ``fs``.

    Raises
    ------
    ValueError
        If ``cutoff`` is not strictly between 0 and the Nyquist frequency
        ``fs/2``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. *Biomedical Signal Analysis*,
        3rd ed. (IEEE Press / Wiley, 2024),
        Ch. 3 "Filtering for Removal of Artifacts" -- pp. 106-208, for the
        artifact-removal filtering context this function serves.
    SciPy developers. ``scipy.signal.firwin`` reference documentation.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
        -- the authoritative specification for the windowed-sinc design and
        the ``scale=True`` normalisation used here.

    Note: the windowed-sinc FIR design this function implements is NOT
    covered by Rangayyan Ch. 3, whose frequency-domain treatment is built on
    Butterworth (IIR) filters (Sec. 3.7.1-3.7.3); FIR design by truncation
    and windowing is mentioned only in passing among "optimal" filters. The
    chapter is therefore cited for context, not as the design specification.
    Per the SciPy documentation, ``firwin`` "raises ValueError if any value
    in cutoff is ... greater than or equal to fs/2", and with ``scale=True``
    (the default) it normalises "the coefficients so that the frequency
    response is exactly unity" at DC for a lowpass -- hence ``sum(taps) == 1``,
    which is the identity pinned in the tests.
    """
    from scipy.signal import filtfilt, firwin

    x = np.asarray(x, dtype=float)
    order = int(order)
    if order < 3:
        order = 3
    if order % 2 == 0:
        order += 1  # ensure odd (linear-phase Type I)
    nyq = 0.5 * fs
    # Reject an out-of-band cutoff instead of clamping it. The previous code
    # clipped fc into (0, 1), so cutoff=10 Hz at fs=1 Hz -- twenty times the
    # Nyquist rate -- silently returned a near-Nyquist filter rather than
    # telling the caller their cutoff was meaningless. scipy.signal.firwin
    # itself raises ValueError when cutoff >= fs/2; masking that turns a
    # caller error into a plausible-looking wrong answer.
    if not (0.0 < cutoff < nyq):
        raise ValueError(
            f"cutoff must satisfy 0 < cutoff < fs/2 (Nyquist); "
            f"got cutoff={cutoff!r} with fs={fs!r} (Nyquist={nyq!r})"
        )
    fc = cutoff / nyq
    taps = firwin(order, fc, window=window)
    # filtfilt needs len(x) > 3 * order; fall back to single-pass for shorts.
    padlen = 3 * order
    if x.size > padlen:
        y = filtfilt(taps, [1.0], x)
    else:
        from scipy.signal import lfilter

        y = lfilter(taps, [1.0], x)
    res = RichResult(
        title="FIR lowpass filter (windowed sinc)",
        summary_lines=[
            ("Order", order),
            ("Cutoff (Hz)", float(cutoff)),
            ("Fs (Hz)", float(fs)),
            ("Window", window),
            ("Output length", int(y.size)),
        ],
        interpretation=(
            f"Zero-phase FIR lowpass of order {order} with cutoff {cutoff:.4g} Hz applied to {x.size} samples."
        ),
        payload={
            "signal": y,
            "taps": taps,
            "order": order,
            "cutoff": float(cutoff),
            "fs": float(fs),
            "window": window,
        },
    )
    return with_describe_pointer(res, "rgfir")


# CANONICAL TEST
# >>> import numpy as np
# >>> fs = 100.0
# >>> t = np.arange(100) / fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*30*t)
# >>> r = rangayyan_fir_filter(x, cutoff=10, order=51, fs=fs)
# >>> r["signal"].shape == x.shape
# True


def cheatsheet():
    return "rgfir: FIR lowpass filter (windowed sinc), scipy.signal.firwin"
