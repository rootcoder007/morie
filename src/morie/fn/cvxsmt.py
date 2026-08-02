# morie.fn -- function file (rootcoder007/morie)
"""Log-sum-exp -- Boyd & Vandenberghe Sec. 3.1.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_smooth_min"]


def boyd_smooth_min(x, axis=None):
    r"""The log-sum-exp :math:`\operatorname{lse}(x) = \log\sum_i e^{x_i}`.

    A smooth approximation of the MAXIMUM, not the minimum, and the
    approximation is two-sided:

    .. math::
        \max_i x_i \le \operatorname{lse}(x) \le \max_i x_i + \log n.

    So the error is at most :math:`\log n` regardless of the values -- it
    depends only on how many terms there are. That bound is what makes lse
    usable as a differentiable surrogate rather than merely suggestive.

    Its gradient is the softmax, which is why the same function underlies
    both smooth-max approximations and multinomial logistic models.

    Computed with the max subtracted out; the naive form overflows at
    :math:`x \approx 710`.

    Parameters
    ----------
    x : array-like
        Values.
    axis : int, optional
        Axis to reduce.

    Returns
    -------
    RichResult
        ``value``, ``gradient`` (the softmax), ``max``, ``gap``,
        ``bound`` (log n).

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> import numpy as np
    >>> r = boyd_smooth_min([0.0, 0.0, 0.0])
    >>> round(r["value"], 6)
    1.098612

    Exactly the log n upper bound when every entry is equal, which is the
    worst case for the approximation.

    >>> bool(abs(r["gap"] - np.log(3)) < 1e-9)
    True

    The gradient is the softmax: non-negative and summing to one.

    >>> g = boyd_smooth_min([1.0, 2.0, 3.0])["gradient"]
    >>> bool(np.all(g >= 0) and abs(g.sum() - 1) < 1e-12)
    True

    No overflow where a naive exp would return inf.

    >>> bool(np.isfinite(boyd_smooth_min([800.0, 799.0])["value"]))
    True
    """
    xv = np.asarray(x, dtype=float)
    if xv.size == 0:
        raise ValueError("x must be non-empty")
    mx = np.max(xv, axis=axis, keepdims=axis is not None)
    if axis is None:
        val = float(mx + np.log(np.sum(np.exp(xv - mx))))
        grad = np.exp(xv - mx)
        grad = grad / grad.sum()
        n = xv.size
        gap = val - float(mx)
    else:
        val = mx + np.log(np.sum(np.exp(xv - mx), axis=axis, keepdims=True))
        grad = np.exp(xv - mx)
        grad = grad / np.sum(grad, axis=axis, keepdims=True)
        n = xv.shape[axis]
        val = np.squeeze(val, axis=axis)
        gap = None
    return RichResult(
        title="Log-sum-exp",
        summary_lines=[("value", val if np.isscalar(val) else float(np.mean(val))),
                       ("terms", int(n)), ("bound log n", float(np.log(n)))],
        payload={
            "value": val, "gradient": grad,
            "max": float(np.max(xv)) if axis is None else np.max(xv, axis=axis),
            "gap": gap, "bound": float(np.log(n)),
            "method": "boyd_smooth_min",
        },
    )


def cheatsheet():
    return "cvxsmt: smooth MAX (not min); error <= log n whatever the values; gradient IS softmax"
