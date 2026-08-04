# morie.fn -- function file (rootcoder007/morie)
"""Continuous-time unit step function (Rangayyan eq. 3.27)."""


from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["ustep", "rangayyan_ch3_unit_step_continuous"]


def ustep(t, shift=0.0):
    """Continuous-time unit step u(t).

    Rangayyan (2024) eq. (3.27):
        u(t) = 1 for t > 0, 0 otherwise.

    Note the strict inequality: u(0) = 0 in this book, not 0.5 and not 1.
    The discrete step of eq. (3.35) uses n >= 0 instead, so the two
    disagree at the origin -- they are separate definitions, not one
    sampled from the other.  The book also notes the delta is the
    derivative of u.
    """
    ts = aslist(t)
    s = float(shift)
    return RichResult(payload={
        "u": [1.0 if v - s > 0.0 else 0.0 for v in ts], "t": ts,
        "shift": s, "value_at_origin": 0.0,
        "method": "Rangayyan (2024) eq. (3.27)"})


rangayyan_ch3_unit_step_continuous = ustep  # pre-policy spelling


def cheatsheet():
    return "rng027: continuous unit step, Rangayyan eq. (3.27)"
