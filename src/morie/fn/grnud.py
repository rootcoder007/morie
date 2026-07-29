# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Central-difference numerical gradient approximation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_numerical_differentiation"]

_METHOD = "Central-difference numerical derivative"


def geron_numerical_differentiation(f, x, h=1e-5):
    r"""Approximate the derivative (or gradient) by central differences.

    .. math::
        \frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}

    The central form has error :math:`O(h^2)` where the forward
    difference is :math:`O(h)`, so at ``h = 1e-5`` it is roughly five
    orders of magnitude more accurate for the same two evaluations.  It
    still costs ``2n`` evaluations for an ``n``-dimensional gradient,
    which is the reason Géron reaches for reverse-mode autodiff instead
    and keeps this as the *check* on it.  A Richardson estimate at
    ``2h`` is returned so you can see whether ``h`` was chosen sanely:
    the two should agree to several digits.

    Parameters
    ----------
    f : callable
        ``f(x) -> float``. Must accept the same shape as ``x``.
    x : float or array-like
        Point of evaluation.
    h : float, optional
        Step; must be positive.

    Returns
    -------
    RichResult
        Payload keys ``derivative``, ``derivative_2h``,
        ``richardson`` (extrapolated ``(4 D_h - D_2h)/3``),
        ``step_error`` (``|D_h - D_2h|``), ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Appendix A, Numerical Differentiation section.

    Examples
    --------
    ``d/dx x^2 = 2x``; at ``x = 3`` central differences are exact for a
    quadratic:

    >>> r = geron_numerical_differentiation(lambda t: t ** 2, 3.0)
    >>> round(r["derivative"], 9)
    6.0
    >>> round(r["step_error"], 9)
    0.0

    Gradient of ``x0^2 + 3 x1`` at ``(1, 1)`` is ``(2, 3)``:

    >>> g = geron_numerical_differentiation(lambda v: v[0] ** 2 + 3 * v[1], [1.0, 1.0])
    >>> [round(v, 7) for v in g["derivative"]]
    [2.0, 3.0]
    """
    if not callable(f):
        raise ValueError(f"f must be callable, got {type(f).__name__}.")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite float, got {h}.")
    xa = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(xa)):
        raise ValueError("x contains non-finite values.")
    scalar = xa.ndim == 0

    def _diff(step):
        if scalar:
            up = float(f(float(xa) + step))
            dn = float(f(float(xa) - step))
            if not (np.isfinite(up) and np.isfinite(dn)):
                raise ValueError(f"f returned a non-finite value near x={float(xa)}.")
            return np.asarray((up - dn) / (2 * step))
        out = np.empty(xa.shape, dtype=float)
        for i in np.ndindex(xa.shape):
            xu = xa.copy()
            xd = xa.copy()
            xu[i] += step
            xd[i] -= step
            up = float(f(xu))
            dn = float(f(xd))
            if not (np.isfinite(up) and np.isfinite(dn)):
                raise ValueError(f"f returned a non-finite value near component {i}.")
            out[i] = (up - dn) / (2 * step)
        return out

    d1 = _diff(h)
    d2 = _diff(2 * h)
    rich = (4.0 * d1 - d2) / 3.0
    err = float(np.max(np.abs(d1 - d2)))
    est = float(d1) if scalar else d1.tolist()

    return RichResult(
        title="Numerical differentiation",
        summary_lines=[("h", h), ("|D_h - D_2h|", err)],
        payload={
            "derivative": est,
            "derivative_2h": float(d2) if scalar else d2.tolist(),
            "richardson": float(rich) if scalar else rich.tolist(),
            "step_error": err,
            "estimate": est,
            "n": 1 if scalar else int(xa.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grnud: (f(x+h) - f(x-h)) / 2h, O(h^2) error; 2n evaluations, Richardson check included"
