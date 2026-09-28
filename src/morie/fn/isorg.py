# morie.fn -- function file (rootcoder007/morie)
"""Pool-adjacent-violators (PAV) isotonic regression."""

from __future__ import annotations

from ._containers import DescriptiveResult


def isotonic_regression(x, w=None):
    """Pool-adjacent-violators (PAV) isotonic regression.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like or None
        Weights. Defaults to uniform.

    Returns
    -------
    DescriptiveResult
        value = monotonically fitted values (ndarray): the weighted
        least-squares non-decreasing fit, each pooled block at its weighted
        mean (Barlow, Bartholomew, Bremner and Brunk 1972), as
        stats::isoreg for unit weights.

    References
    ----------
    Barlow, R. E., Bartholomew, D. J., Bremner, J. M. and Brunk, H. D.
    (1972). *Statistical Inference under Order Restrictions*. Wiley, London.

    Examples
    --------
    >>> [float(v) for v in isotonic_regression([3.0, 1.0, 2.0, 5.0, 4.0, 4.5, 2.0, 6.0]).value]
    [2.0, 2.0, 2.0, 3.875, 3.875, 3.875, 3.875, 6.0]
    """
    from morie.fn import _array_core as np

    y = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    n = len(y)
    wt = [1.0] * n if w is None else [float(v) for v in np.asarray(w, dtype=float).tolist()]
    if len(wt) != n or any(v <= 0 for v in wt):
        raise ValueError("w must be positive and match x")
    # pool adjacent violators on a stack of blocks (value, weight, size) (Barlow et al. 1972)
    vals, wts, sizes = [], [], []
    for v, u in zip(y, wt):
        vals.append(v)
        wts.append(u)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            wts[-2] = tw
            sizes[-2] += sizes[-1]
            del vals[-1], wts[-1], sizes[-1]
    result = np.asarray([v for v, k in zip(vals, sizes) for _ in range(k)], dtype=float)
    return DescriptiveResult(name="isotonic_regression", value=result, extra={"n": n})


isorg = isotonic_regression


def cheatsheet() -> str:
    return "isotonic_regression({}) -> Isotonic regression."
