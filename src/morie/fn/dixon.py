# morie.fn -- function file (rootcoder007/morie)
"""Dixon's Q ratio for a single outlier.

Source: Dixon, W. J. (1953), "Processing data for outliers",
*Biometrics* 9(1):74-89, and the null distribution of the ratios in
Dixon, W. J. (1951), "Ratios involving extreme values", *Annals of
Mathematical Statistics* 22(1):68-78.  Both are paywalled and were NOT
read directly.  The six ratios were taken from the reference
implementation in the R package **outliers** (``dixon.test``), and are
reproduced verbatim below on x sorted ascending, 1-based:

    type   low value suspect            high value suspect
    10     (x2-x1)/(xn-x1)              (xn-x[n-1])/(xn-x1)
    11     (x2-x1)/(x[n-1]-x1)          (xn-x[n-1])/(xn-x2)
    12     (x2-x1)/(x[n-2]-x1)          (xn-x[n-1])/(xn-x3)
    20     (x3-x1)/(xn-x1)              (xn-x[n-2])/(xn-x1)
    21     (x3-x1)/(x[n-1]-x1)          (xn-x[n-2])/(xn-x2)
    22     (x3-x1)/(x[n-2]-x1)          (xn-x[n-2])/(xn-x3)

Type 10 is the classical Q = gap / range recorded in the ledger and is
the default.

NO p-VALUE IS RETURNED, deliberately.  Dixon's ratios have no
closed-form null distribution under normality; ``outliers`` obtains its
p-value from ``pdixon``, which interpolates a stored 15 x 30 table of
simulated critical values, and that table is data, not a formula.  It
could not be obtained in a form this module could verify entry by entry,
and a p-value transcribed from a table that has not been checked is
worse than none.  What is returned is the statistic, which end of the
sample it refers to, and the ratio type, so a caller may compare it
against a published table it holds itself.

DEFECT FIXED.  The previous body of this module computed a
Kolmogorov-Smirnov goodness-of-fit statistic -- sorted data, an
empirical CDF, sup|F_n - F| -- and returned it labelled "Dixon Q test",
together with a KS p-value.  It never formed any ratio of gaps.  Its
``cdf`` argument was a leftover of that copy and is gone.
"""

from ._richresult import RichResult

__all__ = ["dixon_test"]

# (numerator offset, denominator offset) as distances from the suspect end
_RATIOS = {10: (1, 0), 11: (1, 1), 12: (1, 2),
           20: (2, 0), 21: (2, 1), 22: (2, 2)}
_MIN_N = {10: 3, 11: 4, 12: 5, 20: 4, 21: 5, 22: 6}


def dixon_test(x, type=10, opposite=False):
    """Dixon's ratio for the most extreme observation.

    Parameters
    ----------
    x : array-like
        Sample.
    type : {10, 11, 12, 20, 21, 22}
        Which of Dixon's ratios to form.  10 is Q = gap / range.
    opposite : bool
        Test the end NOT selected by the larger deviation from the mean,
        matching the ``opposite`` argument of ``outliers::dixon.test``.

    Returns
    -------
    RichResult
        ``statistic``, ``type``, ``outlier``, ``side``, ``numerator``,
        ``denominator``, ``n``.
    """
    if type not in _RATIOS:
        raise ValueError("type must be one of 10, 11, 12, 20, 21, 22")
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < _MIN_N[type]:
        raise ValueError("Dixon type %d needs at least %d observations"
                         % (type, _MIN_N[type]))
    num_off, den_off = _RATIOS[type]
    m = sum(xs) / n
    take_high = (xs[n - 1] - m) >= (m - xs[0])
    if opposite:
        take_high = not take_high
    if take_high:
        num = xs[n - 1] - xs[n - 1 - num_off]
        den = xs[n - 1] - xs[den_off]
        idx = n - 1
        side = "max"
    else:
        num = xs[num_off] - xs[0]
        den = xs[n - 1 - den_off] - xs[0]
        idx = 0
        side = "min"
    if den == 0.0:
        raise ValueError("Dixon's denominator is zero; the ratio is undefined")
    return RichResult(payload={
        "statistic": float(num / den), "type": int(type),
        "outlier": float(xs[idx]), "side": side,
        "numerator": float(num), "denominator": float(den), "n": n,
        "method": "Dixon (1953) ratio type %d, outliers::dixon.test; "
                  "no p-value, the null distribution is tabulated only"
                  % (type,)})


def cheatsheet():
    return "dixon: Dixon (1953) Q ratio for a single outlier"


# compact alias per ledger/NAMING.md
dixontest = dixon_test
