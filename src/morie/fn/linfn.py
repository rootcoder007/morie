# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""The linear function y = a + b x, its slope angle and line intersections.

Source READ FROM THE CORPUS PDF, page rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 2.5.1, printed page 52 (PDF page 86), equation (2.54)::

    y = a + b x                                                    (2.54)

The book's own reading of the parts, same page: the graph is a straight
line intersecting the ordinate at ``a`` (the **intercept**); ``b`` is the
**slope**; the line falls for ``b < 0``, rises for ``b > 0`` and is
parallel to the abscissa for ``b = 0``; and ``b = tan(alpha)`` where
``alpha`` is the angle at which the line meets the abscissa.  The text
immediately following adds that the intersection point of two lines
``(xS, yS)``, or the intersection with the abscissa, is obtained from the
solution of the corresponding linear equations, which is what the
optional second line supplies.
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["linfn"]


def linfn(a, b, x=None, a2=None, b2=None):
    """Evaluate ``y = a + b x`` and describe the line.

    Parameters
    ----------
    a : float
        Intercept on the ordinate.
    b : float
        Slope; ``b = tan(alpha)``.
    x : float or array-like, optional
        Abscissa value(s) at which to evaluate the function.
    a2, b2 : float, optional
        Intercept and slope of a second line.  Both must be given
        together; the intersection point ``(xS, yS)`` is then returned.

    Returns
    -------
    RichResult
        Keys: ``intercept``, ``slope``, ``angle_rad``, ``angle_deg``,
        ``direction`` (``"rising"`` / ``"falling"`` / ``"parallel"``),
        ``root`` (intersection with the abscissa, ``None`` when
        ``b = 0``), and optionally ``y`` and ``x_int`` / ``y_int``.
    """
    a = float(a)
    b = float(b)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be finite")
    payload = {
        "intercept": a,
        "slope": b,
        "angle_rad": math.atan(b),
        "angle_deg": math.degrees(math.atan(b)),
        "direction": "rising" if b > 0.0 else ("falling" if b < 0.0 else "parallel"),
        "root": (-a / b) if b != 0.0 else None,
    }
    summary = [("y = a + b x", (a, b)), ("angle (deg)", payload["angle_deg"]),
               ("direction", payload["direction"])]
    if x is not None:
        if hasattr(x, "__len__"):
            xs = [float(v) for v in x]
            if not xs:
                raise ValueError("x must not be empty")
            payload["y"] = [a + b * v for v in xs]
        else:
            payload["y"] = a + b * float(x)
        summary.append(("y", payload["y"]))
    if (a2 is None) != (b2 is None):
        raise ValueError("a2 and b2 must be supplied together")
    if a2 is not None:
        a2 = float(a2)
        b2 = float(b2)
        if not math.isfinite(a2) or not math.isfinite(b2):
            raise ValueError("a2 and b2 must be finite")
        if b == b2:
            raise ValueError("the two lines are parallel and do not intersect")
        xs_ = (a2 - a) / (b - b2)
        payload["x_int"] = xs_
        payload["y_int"] = a + b * xs_
        summary.append(("intersection", (payload["x_int"], payload["y_int"])))
    return RichResult(
        title="Linear function y = a + b x (Hedderich eq. 2.54)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet() -> str:
    return "linfn(a, b, x): the line y = a + b x, its angle and intersections -- Hedderich eq. (2.54)."
