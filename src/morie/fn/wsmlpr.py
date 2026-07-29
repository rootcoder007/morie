# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local polynomial regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_local_polynomial"]


def wasserman_local_polynomial(x, x_data, y_data, h, p=1):
    """
    Local polynomial regression (degree p) with a Gaussian kernel.

    Formula: at each x solve the weighted least squares
    min sum_i K_h(x - X_i) (Y_i - sum_j b_j (X_i - x)^j)^2 and
    report b_0 as m_hat(x); b_1 estimates the derivative. p = 0
    recovers Nadaraya-Watson exactly; p = 1 is the local linear
    smoother with its boundary-bias advantage.

    Parameters
    ----------
    x : array-like
        Evaluation point(s).
    x_data, y_data : array-like
        Paired training sample.
    h : float
        Bandwidth, > 0.
    p : int
        Polynomial degree, >= 0, with enough data (n > p).

    Returns
    -------
    result : dict
        Keys: estimate (m at first x), values, derivatives (b_1 per
        x, nan when p = 0), h, p, n, method.

    References
    ----------
    Wasserman (2004), Ch 20, section 20.4 (local polynomials).

    Examples
    --------
    Local linear reproduces an exact line at any bandwidth:

    >>> xs = [0.0, 1.0, 2.0, 3.0]
    >>> ys = [1.0, 3.0, 5.0, 7.0]
    >>> out = wasserman_local_polynomial(1.5, xs, ys, 0.7, 1)
    >>> round(out["estimate"], 10)
    4.0
    >>> round(out["derivatives"][0], 10)
    2.0
    >>> nw = wasserman_local_polynomial(1.5, xs, ys, 0.7, 0)
    >>> from morie.fn.wsmcrk import wasserman_kernel_regression
    >>> round(nw["estimate"], 12) == round(wasserman_kernel_regression(1.5, xs, ys, 0.7)["estimate"], 12)
    True
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    xd = np.atleast_1d(np.asarray(x_data, dtype=float))
    yd = np.atleast_1d(np.asarray(y_data, dtype=float))
    h = float(h)
    p = int(p)
    if xd.size != yd.size:
        raise ValueError(f"x_data ({xd.size}) and y_data ({yd.size}) lengths differ.")
    if h <= 0:
        raise ValueError(f"the bandwidth must be positive; got {h}.")
    if p < 0:
        raise ValueError(f"the degree must be >= 0; got {p}.")
    if xd.size <= p:
        raise ValueError(f"local degree-{p} fitting needs n > p; got n={xd.size}.")
    vals, ders = [], []
    for xi in x:
        w = np.exp(-0.5 * ((xi - xd) / h) ** 2)
        D = np.vander(xd - xi, p + 1, increasing=True)
        sw = np.sqrt(w)
        b, _, rank, _ = np.linalg.lstsq(D * sw[:, None], yd * sw, rcond=None)
        if rank < p + 1:
            vals.append(float("nan")); ders.append(float("nan")); continue
        vals.append(float(b[0]))
        ders.append(float(b[1]) if p >= 1 else float("nan"))
    return RichResult(payload={
        "estimate": vals[0], "values": vals, "derivatives": ders,
        "h": h, "p": p, "n": int(xd.size),
        "method": "local polynomial WLS, Gaussian kernel; b0 = fit, b1 = slope"})


def cheatsheet():
    return "wsmlpr: WLS on (X-x)^j basis with kernel weights; p=0 == NW"
