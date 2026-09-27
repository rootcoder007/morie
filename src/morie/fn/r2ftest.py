"""F test that all slopes are zero, from R^2.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eqs (6.11)-(6.13).
"""

from ._richresult import RichResult
from ._rrng_core import pf

__all__ = ["r2ftest"]


def r2ftest(r2, n, p):
    """F = ((n - p - 1)/p) R^2 / (1 - R^2) on (p, n - p - 1) degrees of freedom (6.13).

    Parameters
    ----------
    r2 : float
        Coefficient of determination, 0 <= R^2 < 1.
    n : int
        Sample size.
    p : int
        Number of predictors.

    Returns
    -------
    RichResult
        Keys: statistic, df1, df2, p_value.

    References
    ----------
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (6.13).

    Examples
    --------
    >>> round(r2ftest(0.5, 23, 2)["statistic"], 12)
    10.0
    """
    if not 0 <= r2 < 1 or p < 1 or n - p - 1 < 1:
        raise ValueError("need 0 <= R^2 < 1, p >= 1 and n > p + 1")
    df2 = n - p - 1
    f = df2 / p * r2 / (1 - r2)
    return RichResult(
        title="F test of all slopes zero",
        summary_lines=[("F", f), ("df", (p, df2))],
        payload={"statistic": f, "df1": p, "df2": df2, "p_value": pf(f, p, df2, lower_tail=False)},
    )


def cheatsheet():
    return "r2ftest: F = ((n-p-1)/p) R^2/(1-R^2). Wilcox (2017) eq (6.13)."
