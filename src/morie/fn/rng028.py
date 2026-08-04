# morie.fn -- function file (rootcoder007/morie)
"""Sifting property of the Dirac delta (Rangayyan eq. 3.28)."""


from ._richresult import RichResult

__all__ = ["sifting", "rangayyan_ch3_sifting_property"]


def sifting(x, t0, lower, upper):
    """Sift the value of x at t0 out of an interval.

    Rangayyan (2024) eq. (3.28):
        integral_{T1}^{T2} x(t) delta(t - to) dt
            = x(to)  if T1 < to < T2,
            = 0      otherwise.

    The inequalities are strict at both ends: an impulse sitting exactly
    on a limit of integration contributes nothing under this definition,
    which is why ``inside`` is reported alongside the value.

    Parameters
    ----------
    x : callable
        The function being sifted; must be continuous at t0.
    t0 : float
        Location of the impulse.
    lower, upper : float
        Interval of integration.
    """
    if not callable(x):
        raise TypeError("x must be a callable continuous at t0")
    lo, hi, t = float(lower), float(upper), float(t0)
    if hi <= lo:
        raise ValueError("upper must exceed lower")
    inside = lo < t < hi
    return RichResult(payload={
        "value": float(x(t)) if inside else 0.0, "inside": inside,
        "t0": t, "lower": lo, "upper": hi,
        "method": "Rangayyan (2024) eq. (3.28)"})


rangayyan_ch3_sifting_property = sifting  # pre-policy spelling


def cheatsheet():
    return "rng028: sifting property, Rangayyan eq. (3.28)"
