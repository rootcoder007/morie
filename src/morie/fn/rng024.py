# morie.fn -- function file (rootcoder007/morie)
"""Continuous-time Dirac delta function (Rangayyan eq. 3.24)."""


from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["diracdelta", "rangayyan_ch3_dirac_delta_definition"]


def diracdelta(t, width=None):
    """Dirac delta evaluated on a time grid.

    Rangayyan (2024) eq. (3.24):
        delta(t) = undefined at t = 0, 0 otherwise.

    A generalized function has no pointwise value at the origin, so the
    honest return is the definition itself: 0 everywhere and None at
    t = 0.  Passing ``width`` instead returns the unit-area rectangular
    pulse of that duration -- the approximating family of Figure 3.10,
    whose limit is the delta -- which is what a numerical caller actually
    needs.  The two are kept in one function so that no caller silently
    treats the rectangle as if it were the delta.
    """
    ts = aslist(t)
    if width is None:
        vals = [None if v == 0.0 else 0.0 for v in ts]
        return RichResult(payload={
            "delta": vals, "t": ts, "undefined_at_zero": True,
            "method": "Rangayyan (2024) eq. (3.24)"})
    w = float(width)
    if w <= 0:
        raise ValueError("width must be positive")
    h = 1.0 / w
    vals = [h if abs(v) <= w / 2.0 else 0.0 for v in ts]
    return RichResult(payload={
        "delta": vals, "t": ts, "width": w, "height": h,
        "undefined_at_zero": False,
        "method": "Rangayyan (2024) eq. (3.24), rectangular approximation"})


rangayyan_ch3_dirac_delta_definition = diracdelta  # pre-policy spelling


def cheatsheet():
    return "rng024: Dirac delta definition, Rangayyan eq. (3.24)"
