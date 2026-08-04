# morie.fn -- function file (rootcoder007/morie)
"""Continuous-time convolution (Rangayyan eq. 3.30)."""


from ._rgcore import aslist, gridint
from ._richresult import RichResult

__all__ = ["contconv", "rangayyan_ch3_continuous_convolution"]


def contconv(x, h, dt=1.0, t=None):
    """Convolution of an input with an impulse response, tabulated form.

    Rangayyan (2024) eq. (3.30):
        y(t) = integral x(tau) h(t - tau) d tau.

    Tabulated on a uniform grid of spacing dt this becomes the discrete
    convolution scaled by dt -- the dt is what makes it an integral
    rather than eq. (3.36)'s sum, and dropping it is the usual way a
    continuous-time convolution comes out wrong by a factor of the
    sampling interval.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    step = float(dt)
    if t is not None:
        ts = aslist(t)
        if len(ts) != len(xs):
            raise ValueError("t must match x in length")
        if len(ts) > 1:
            step = ts[1] - ts[0]
    if step <= 0:
        raise ValueError("dt must be positive")
    n, m = len(xs), len(hs)
    y = []
    for k in range(n + m - 1):
        lo = max(0, k - m + 1)
        hi = min(k, n - 1)
        y.append(sum(xs[i] * hs[k - i] for i in range(lo, hi + 1)) * step)
    t_out = [i * step for i in range(len(y))]
    if t is not None and len(ts):
        t_out = [ts[0] + i * step for i in range(len(y))]
    return RichResult(payload={
        "y": y, "t": t_out, "dt": step, "n": n, "m": m,
        "integral": gridint(y, t_out) if len(y) > 1 else 0.0,
        "method": "Rangayyan (2024) eq. (3.30)"})


rangayyan_ch3_continuous_convolution = contconv  # pre-policy spelling


def cheatsheet():
    return "rng030: continuous-time convolution, Rangayyan eq. (3.30)"
