"""Kernel smoother for a binary outcome (modified Hosmer-Lemeshow/Copas).

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (15.15).
"""

import math

from ._richresult import RichResult

__all__ = ["logrsm"]


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def logrsm(x, y, pts=None, h=1.2):
    """P-hat(x) = sum w_j Y_j / sum w_j with w_i = I(|X_i - x| < h) exp(-(X_i - x)^2) (15.15).

    X (and the evaluation points) are first standardized as (X - M_x)/MADN_x,
    with M_x the median and MADN_x = MAD/0.6745.

    Parameters
    ----------
    x : sequence of float
    y : sequence of 0/1
    pts : sequence of float, optional
        Evaluation points on the original scale (default x).
    h : float
        Window half-width on the standardized scale.

    Returns
    -------
    RichResult
        Keys: phat (NaN where no X lies within h), pts.

    References
    ----------
    Hosmer, D. W. & Lemeshow, S. (1989). Applied Logistic Regression, p. 85.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (15.15).

    Examples
    --------
    >>> logrsm([1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 1, 1], pts=[3.5])["phat"]
    [0.5]
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if len(x) != len(y) or len(x) < 2 or any(v not in (0.0, 1.0) for v in y):
        raise ValueError("x and y must match and y must be 0/1")
    m = _median(x)
    madn = _median([abs(v - m) for v in x]) / 0.6745
    if madn <= 0:
        raise ValueError("MAD of x is zero")
    zs = [(v - m) / madn for v in x]
    pts = x if pts is None else [float(v) for v in pts]
    out = []
    for t in pts:
        z = (t - m) / madn
        w = [math.exp(-((a - z) ** 2)) if abs(a - z) < h else 0.0 for a in zs]
        s = sum(w)
        out.append(sum(a * b for a, b in zip(w, y)) / s if s > 0 else float("nan"))
    return RichResult(
        title="Binary-outcome smoother", summary_lines=[("points", len(out))], payload={"phat": out, "pts": pts}
    )


def cheatsheet():
    return "logrsm: kernel smoother for P(Y=1|x), weights I(|z_i - z| < h) exp(-(z_i - z)^2). Wilcox (2017) eq (15.15)."
