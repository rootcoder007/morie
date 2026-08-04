# morie.fn -- function file (rootcoder007/morie)
"""Commuted form of continuous-time convolution (Rangayyan eq. 3.31)."""


from ._rgcore import aslist
from ._richresult import RichResult
from .rng030 import contconv

__all__ = ["contconvalt", "rangayyan_ch3_continuous_convolution_alt"]


def contconvalt(x, h, dt=1.0, t=None):
    """Convolution with the roles of x and h swapped.

    Rangayyan (2024) eq. (3.31):
        y(t) = integral h(tau) x(t - tau) d tau,

    which the book gives as an equivalent result to eq. (3.30).  It is
    computed here the other way round and compared against eq. (3.30)
    rather than merely asserted equivalent: ``max_difference`` is the
    largest discrepancy between the two orders, and is zero up to
    rounding for any pair of finite sequences.
    """
    xs, hs = aslist(x), aslist(h)
    swapped = contconv(hs, xs, dt=dt, t=t)
    direct = contconv(xs, hs, dt=dt, t=t)
    a, b = swapped["y"], direct["y"]
    diff = max((abs(p - q) for p, q in zip(a, b)), default=0.0)
    out = dict(swapped)
    out["max_difference"] = diff
    out["commutes"] = diff <= 1e-12 * max(
        1.0, max((abs(v) for v in b), default=1.0))
    out["method"] = "Rangayyan (2024) eq. (3.31)"
    return RichResult(payload=out)


rangayyan_ch3_continuous_convolution_alt = contconvalt  # pre-policy spelling


def cheatsheet():
    return "rng031: commuted continuous convolution, Rangayyan eq. (3.31)"
