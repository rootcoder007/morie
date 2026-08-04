# morie.fn -- function file (rootcoder007/morie)
"""Unit-area property of the Dirac delta (Rangayyan eq. 3.25)."""


from ._rgcore import gridint
from ._richresult import RichResult

__all__ = ["deltaarea", "rangayyan_ch3_dirac_delta_unit_area"]


def deltaarea(t=None, values=None, width=None):
    """Verify the unit-area property of a delta approximation.

    Rangayyan (2024) eq. (3.25):
        integral_{-inf}^{inf} delta(t) dt = 1.

    The property is what defines the delta, so the useful computation is
    the check: integrate a candidate approximation and report how far its
    mass is from 1.  With no arguments the exact area 1.0 is returned;
    with ``width`` the rectangular pulse of Figure 3.10 is integrated on
    a fine grid, which must give 1 at any width -- that invariance under
    compression is the point of the figure.
    """
    if values is not None:
        if t is None:
            raise ValueError("give the grid t alongside values")
        area = gridint(values, t)
        return RichResult(payload={
            "area": float(area), "unit_area": abs(area - 1.0) <= 1e-6,
            "method": "Rangayyan (2024) eq. (3.25)"})
    if width is not None:
        w = float(width)
        if w <= 0:
            raise ValueError("width must be positive")
        # panel edges placed on +/- w/2 so no panel straddles the jump;
        # the midpoint rule is then exact for a piecewise-constant pulse.
        n_panels = 800
        span = 2.0 * w
        h = 2.0 * span / n_panels
        edges = [-span + i * h for i in range(n_panels + 1)]
        edges = sorted(set(edges + [-w / 2.0, w / 2.0]))
        area = 0.0
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (lo_e + hi_e)
            area += (hi_e - lo_e) * ((1.0 / w) if abs(mid) <= w / 2.0 else 0.0)
        return RichResult(payload={
            "area": float(area), "width": w,
            "unit_area": abs(area - 1.0) <= 1e-9,
            "method": "Rangayyan (2024) eq. (3.25)"})
    return RichResult(payload={
        "area": 1.0, "unit_area": True,
        "method": "Rangayyan (2024) eq. (3.25)"})


rangayyan_ch3_dirac_delta_unit_area = deltaarea  # pre-policy spelling


def cheatsheet():
    return "rng025: unit area of the delta, Rangayyan eq. (3.25)"
