# morie.fn -- function file (rootcoder007/morie)
"""Perspective function -- Boyd & Vandenberghe Sec. 3.2.6."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_perspective"]


def boyd_perspective(f, x, t):
    r"""The perspective :math:`g(x, t) = t\,f(x/t)` for :math:`t > 0`.

    Convexity is PRESERVED, and jointly in (x, t) -- which is not obvious,
    since :math:`t f(x/t)` is a product of a variable with a nonlinear
    function of a ratio. That closure is why the perspective shows up
    wherever a convex problem needs a scale variable: the relative
    entropy, the quadratic-over-linear function, and every homogeneous
    reformulation of a fractional program.

    The domain is :math:`t > 0` strictly. At :math:`t = 0` the expression
    is undefined, and the closure of the perspective takes a value there
    only in a limiting sense that depends on f -- so the function raises
    rather than picking one of those limits silently.

    Parameters
    ----------
    f : callable
        The base function.
    x : array-like
        Points.
    t : array-like
        Scales, strictly positive.

    Returns
    -------
    RichResult
        ``value``, ``ratio`` (:math:`x/t`), ``f_ratio``,
        ``homogeneous`` (whether g is degree-1 homogeneous here).

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    The perspective of the squared norm is the quadratic-over-linear
    function :math:`x^2/t`.

    >>> r = boyd_perspective(lambda z: z ** 2, [2.0], [4.0])
    >>> round(float(r["value"][0]), 6)
    1.0

    Degree-1 homogeneity: scaling BOTH arguments scales the value by the
    same factor, which is the defining property.

    >>> a = boyd_perspective(lambda z: z ** 2, [2.0], [4.0])["value"][0]
    >>> b = boyd_perspective(lambda z: z ** 2, [6.0], [12.0])["value"][0]
    >>> bool(abs(b - 3 * a) < 1e-12)
    True

    The perspective of -log is the relative-entropy building block
    :math:`-t\log(x/t)`.

    >>> import numpy as np
    >>> round(float(boyd_perspective(lambda z: -np.log(z), [1.0], [2.0])["value"][0]), 6)
    1.386294

    t = 0 is outside the domain and is refused rather than assigned a
    limit that depends on f.

    >>> boyd_perspective(lambda z: z ** 2, [1.0], [0.0])
    Traceback (most recent call last):
        ...
    ValueError: the perspective needs t > 0 strictly; entry 0 is 0
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    tv = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if xv.size != tv.size:
        raise ValueError(f"x has {xv.size} entries but t has {tv.size}")
    bad = np.flatnonzero(tv <= 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"the perspective needs t > 0 strictly; entry {i} is {tv[i]:g}")
    ratio = xv / tv
    fr = np.asarray([float(f(v)) for v in ratio], dtype=float)
    val = tv * fr
    # Degree-1 homogeneity is definitional, so verify it rather than
    # assert it: g(2x, 2t) should be 2 g(x, t).
    fr2 = np.asarray([float(f(v)) for v in (2 * xv) / (2 * tv)], dtype=float)
    homo = bool(np.allclose(2 * tv * fr2, 2 * val, rtol=1e-10, atol=1e-12))
    return RichResult(
        title="Perspective function",
        summary_lines=[("points", int(xv.size)),
                       ("mean value", float(val.mean())),
                       ("homogeneous", homo)],
        payload={
            "value": val, "ratio": ratio, "f_ratio": fr,
            "homogeneous": homo, "method": "boyd_perspective",
        },
    )


def cheatsheet():
    return "cvxprl: preserves convexity JOINTLY in (x,t); t>0 strictly -- the t=0 limit depends on f"
