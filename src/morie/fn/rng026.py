# morie.fn -- function file (rootcoder007/morie)
"""Dirac delta as a limit of a power function (Rangayyan eq. 3.26)."""


from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["deltalim", "rangayyan_ch3_dirac_delta_limit_form"]


def deltalim(t, a):
    """The power-function family whose limit is the delta.

    Rangayyan (2024) eq. (3.26):
        delta(t) = 0.5 * lim_{a->0} a |t|^(a-1).

    Figure 3.11 plots this for a = 0.8, 0.4, 0.2.  The exponent a - 1 is
    negative for every a in (0, 1), so the function diverges at t = 0 --
    returned as None there, not as a large finite number.  Its integral
    over any symmetric interval [-L, L] is L^a, which tends to 1 as
    a -> 0 for any fixed L: that is why the limit is the unit-area delta,
    and it is reported so the caller can see the convergence.
    """
    ts = aslist(t)
    av = float(a)
    if av <= 0:
        raise ValueError("a must be positive")
    vals = [None if v == 0.0 else 0.5 * av * abs(v) ** (av - 1.0)
            for v in ts]
    lim = max((abs(v) for v in ts), default=1.0) or 1.0
    return RichResult(payload={
        "values": vals, "t": ts, "a": av,
        "area_symmetric": lim ** av,
        "half_width": lim,
        "method": "Rangayyan (2024) eq. (3.26)"})


rangayyan_ch3_dirac_delta_limit_form = deltalim  # pre-policy spelling


def cheatsheet():
    return "rng026: delta as a power-function limit, Rangayyan eq. (3.26)"
