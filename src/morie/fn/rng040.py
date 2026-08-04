# morie.fn -- function file (rootcoder007/morie)
"""Linear-ramp smoothing filter (Rangayyan eq. 3.42)."""


from math import fsum

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["rampfilt", "rangayyan_ch3_linear_ramp_filter"]


def rampfilt(x=None, fs=2000.0, duration=0.25, slope=10.0):
    """Linearly decreasing ramp impulse response, and the filtering it does.

    Rangayyan (2024) eq. (3.42):
        h(t) = 10 (0.25 - t),   0 <= t <= 0.25 s,

    used in the book with fs = 2 kHz.  The text immediately after the
    equation states that the output was divided by the sum of all the
    values of h(n), making the result a weighted average of the input --
    so the normalization is part of the method, not an embellishment.
    Without it the filter would apply a gain of sum(h), which for the
    book's constants is 626.25.

    Parameters
    ----------
    x : array-like, optional
        Signal to filter.  With no signal only the taps are returned.
    fs : float
        Sampling frequency, 2000 Hz in the book.
    duration : float
        Ramp length in seconds, 0.25 s in the book.
    slope : float
        Leading coefficient, 10 in the book.
    """
    if fs <= 0 or duration <= 0:
        raise ValueError("fs and duration must be positive")
    n_taps = int(round(duration * fs)) + 1
    h = [slope * (duration - i / fs) for i in range(n_taps)]
    gain = fsum(h)
    if gain <= 0:
        raise ValueError("ramp has nonpositive total weight")
    hn = [v / gain for v in h]
    out = {"h": h, "h_normalized": hn, "gain": gain, "n_taps": n_taps,
           "fs": float(fs), "duration": float(duration),
           "method": "Rangayyan (2024) eq. (3.42)"}
    if x is not None:
        xs = aslist(x)
        y = []
        for n in range(len(xs)):
            lo = max(0, n - n_taps + 1)
            y.append(fsum(xs[i] * hn[n - i] for i in range(lo, n + 1)))
        out["y"] = y
        out["n"] = len(xs)
    return RichResult(payload=out)


rangayyan_ch3_linear_ramp_filter = rampfilt  # pre-policy spelling


def cheatsheet():
    return "rng040: linear-ramp smoothing filter, Rangayyan eq. (3.42)"
