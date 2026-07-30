# morie.fn -- function file (rootcoder007/morie)
"""Proximal operator -- Parikh & Boyd (2014)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_proximal"]

_CLOSED_FORM = ("l1", "l2", "l2sq", "nonneg", "box", "zero")


def boyd_proximal(h, v, t=1.0, lo=None, hi=None):
    r"""Evaluate the proximal operator of ``h`` at ``v``.

    .. math::
        \operatorname{prox}_{t h}(v) = \arg\min_x \;
            h(x) + \tfrac{1}{2t}\lVert x - v \rVert_2^2 .

    Closed forms are used where they exist, since they are exact and cheap:

    ==========  ==========================================================
    ``h``       :math:`\operatorname{prox}_{th}(v)`
    ==========  ==========================================================
    ``"l1"``    soft threshold, :math:`\operatorname{sign}(v)(|v|-t)_+`
    ``"l2"``    block soft threshold, :math:`(1 - t/\lVert v\rVert)_+ v`
    ``"l2sq"``  shrinkage, :math:`v / (1 + 2t)`
    ``"nonneg"``  projection, :math:`\max(v, 0)`
    ``"box"``   projection onto ``[lo, hi]``
    ``"zero"``  identity
    ==========  ==========================================================

    Note the difference between ``"l1"`` (soft-thresholds each coordinate,
    giving sparsity) and ``"l2"`` (shrinks the whole vector toward zero,
    setting it to exactly zero only as a block) -- picking the wrong one is
    the usual cause of a group-lasso that will not select groups.

    Parameters
    ----------
    h : str or callable
        One of the names above, or a callable ``h(x) -> float``, in which
        case the minimisation is done numerically.
    v : array-like
        Point at which to evaluate.
    t : float
        Step size, positive.
    lo, hi : float, optional
        Box bounds, required when ``h="box"``.

    Returns
    -------
    RichResult
        ``prox`` (the point), ``moreau`` (the Moreau envelope value), and
        ``h_name``.

    References
    ----------
    Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations and
        Trends in Optimization*, 1(3), 127-239.

    Examples
    --------
    Soft thresholding is the L1 prox -- entries smaller than ``t`` go to
    exactly zero.

    >>> [float(v) for v in boyd_proximal("l1", [3.0, -0.5, 0.2], t=1.0)["prox"]]
    [2.0, -0.0, 0.0]

    The L2 prox shrinks the vector as a block; nothing is individually
    zeroed unless the whole block is.

    >>> [float(round(v, 4)) for v in boyd_proximal("l2", [3.0, 4.0], t=1.0)["prox"]]
    [2.4, 3.2]

    >>> [float(v) for v in boyd_proximal("nonneg", [-1.0, 2.0])["prox"]]
    [0.0, 2.0]

    >>> boyd_proximal("l1", [1.0], t=0.0)
    Traceback (most recent call last):
        ...
    ValueError: t must be positive
    """
    if t <= 0:
        raise ValueError("t must be positive")
    v = np.atleast_1d(np.asarray(v, dtype=float)).ravel()

    if callable(h):
        from scipy.optimize import minimize

        obj = lambda x: float(h(x)) + np.sum((x - v) ** 2) / (2 * t)  # noqa: E731
        x = minimize(obj, v, method="Nelder-Mead").x
        name = getattr(h, "__name__", "callable")
    elif h == "l1":
        x = np.sign(v) * np.maximum(np.abs(v) - t, 0.0)
        name = "l1"
    elif h == "l2":
        nv = float(np.linalg.norm(v))
        x = np.zeros_like(v) if nv <= t else (1.0 - t / nv) * v
        name = "l2"
    elif h == "l2sq":
        x = v / (1.0 + 2.0 * t)
        name = "l2sq"
    elif h == "nonneg":
        x = np.maximum(v, 0.0)
        name = "nonneg"
    elif h == "box":
        if lo is None or hi is None:
            raise ValueError('h="box" requires both lo and hi')
        if lo > hi:
            raise ValueError("lo must not exceed hi")
        x = np.clip(v, lo, hi)
        name = "box"
    elif h == "zero":
        x = v.copy()
        name = "zero"
    else:
        raise ValueError(f"unknown h {h!r}; expected a callable or one of {_CLOSED_FORM}")

    hx = float(h(x)) if callable(h) else _h_value(name, x, lo, hi)
    return RichResult(
        title=f"prox_{{t*{name}}}",
        summary_lines=[("h", name), ("t", float(t))],
        payload={
            "prox": x,
            "moreau": hx + float(np.sum((x - v) ** 2)) / (2 * t),
            "h_value": hx,
            "h_name": name,
            "t": float(t),
            "method": "boyd_proximal",
        },
    )


def _h_value(name, x, lo, hi):
    if name == "l1":
        return float(np.sum(np.abs(x)))
    if name == "l2":
        return float(np.linalg.norm(x))
    if name == "l2sq":
        return float(np.sum(x**2))
    if name == "nonneg":
        return 0.0 if np.all(x >= 0) else float("inf")
    if name == "box":
        return 0.0 if np.all((x >= lo) & (x <= hi)) else float("inf")
    return 0.0


def cheatsheet():
    return 'cvxprx: prox of l1/l2/l2sq/nonneg/box in closed form; "l1" sparsifies, "l2" shrinks as a block'
