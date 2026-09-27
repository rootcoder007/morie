"""Algina-Keselman-Penfield robust effect size for one group (or paired differences).

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (8.6).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qnorm
from .trimse import _tmean, _winvar

__all__ = ["akpd"]


def akpd(x, y=None, tr=0.2, null_value=0.0):
    """d = k (Xbar_t - null) / s_w, k chosen so s_w / k estimates sigma under normality.

    k^2 is the Winsorized variance of the standard normal,
    (1 - 2G) - 2 z phi(z) + 2 G z^2 with z = Phi^{-1}(1 - G); k = 0.642 for
    20% trimming (8.6). With ``y`` the paired differences x - y are used.

    Parameters
    ----------
    x, y : sequences of float (y optional)
    tr : float
    null_value : float

    Returns
    -------
    RichResult
        Keys: d, k.

    References
    ----------
    Algina, J., Keselman, H. J. & Penfield, R. D. (2005). Psychological Methods 10, 317-328.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (8.6).

    Examples
    --------
    >>> round(akpd([2.1, 3.4, 1.9, 5.6, 4.4])["k"], 3)
    0.642
    """
    x = [float(v) for v in x]
    if y is not None:
        y = [float(v) for v in y]
        if len(y) != len(x):
            raise ValueError("x and y must be paired")
        x = [a - b for a, b in zip(x, y)]
    if len(x) < 2 or not 0 <= tr < 0.5:
        raise ValueError("need n >= 2 and 0 <= tr < 0.5")
    k = 1.0
    if tr > 0:
        z = qnorm(1 - tr)
        phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        k = math.sqrt((1 - 2 * tr) - 2 * z * phi + 2 * tr * z * z)
    d = k * (_tmean(x, tr) - null_value) / math.sqrt(_winvar(x, tr))
    return RichResult(title="AKP robust effect size", summary_lines=[("d", d)], payload={"d": d, "k": k})


def cheatsheet():
    return "akpd: Algina-Keselman-Penfield d = k (Xbar_t - null)/s_w. Wilcox (2017) eq (8.6)."
