"""Likelihood-ratio (G) tests for frequencies."""

import math

from ._richresult import RichResult
from ._stats_core import chi2

__all__ = ["g_test"]


def g_test(observed, expected=None, n_estimated=0):
    r"""Likelihood-ratio statistic :math:`G = 2\sum n_i\log(n_i/e_i)` for frequencies.

    Hedderich, Sachs & Reynarowych (2023, eq 7.48): for k observed counts
    against expected counts (or probabilities) :math:`G` is asymptotically
    :math:`\chi^2` with :math:`k - 1 - m` degrees of freedom, m the number of
    parameters estimated from the data. Given an r x c table and no
    ``expected``, the test is of independence against the margins, on
    :math:`(r-1)(c-1)` degrees of freedom -- the log-linear
    :math:`G^2` (DescTools::GTest). Cells with :math:`n_i = 0` contribute
    zero. The book's Hardy-Weinberg example prints LR = 1.92; its own
    counts give 1.19.

    Parameters
    ----------
    observed : sequence of counts, or an r x c table
    expected : sequence, optional
        Expected counts or probabilities (rescaled to the total).
    n_estimated : int
        Parameters estimated from the data (one-way case).

    Returns
    -------
    RichResult
        ``statistic``, ``df``, ``p_value``, ``expected``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eq (7.48). Agresti, A. (2013). Categorical
    Data Analysis, 3rd ed. Wiley, Sec. 3.2.
    """
    table = isinstance(observed[0], (list, tuple))
    if table:
        t = [[float(v) for v in row] for row in observed]
        rs = [sum(r) for r in t]
        cs = [sum(r[j] for r in t) for j in range(len(t[0]))]
        n = sum(rs)
        exp = [[a * b / n for b in cs] for a in rs]
        g = 2.0 * sum(o * math.log(o / e) for ro, re in zip(t, exp) for o, e in zip(ro, re) if o > 0)
        df = (len(t) - 1) * (len(t[0]) - 1)
    else:
        o = [float(v) for v in observed]
        n = sum(o)
        if expected is None:
            e = [n / len(o)] * len(o)
        else:
            e = [float(v) for v in expected]
            s = sum(e)
            e = [v * n / s for v in e]
        g = 2.0 * sum(a * math.log(a / b) for a, b in zip(o, e) if a > 0)
        df = len(o) - 1 - int(n_estimated)
        exp = e
    if df < 1:
        raise ValueError("no degrees of freedom left")
    return RichResult(
        title="Likelihood-ratio (G) test",
        summary_lines=[("G", g), ("df", df)],
        payload={"statistic": g, "df": df, "p_value": float(chi2.sf(g, df)), "expected": exp},
    )


def cheatsheet():
    return "lrgtst: G = 2 sum n log(n/e), goodness of fit or r x c independence"
