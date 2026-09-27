"""Friedman's test in its F form (handles ties through the ranks).

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (11.6).
"""

from ._richresult import RichResult
from ._rrng_core import pf

__all__ = ["friedf"]


def _midranks(row):
    order = sorted(range(len(row)), key=lambda i: row[i])
    r = [0.0] * len(row)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and row[order[j + 1]] == row[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def friedf(X):
    """F = (n - 1)(B - C) / (A - B) on (J - 1, (n - 1)(J - 1)) df (11.6).

    Ranks R_ij are assigned within rows (midranks for ties);
    A = sum R_ij^2, B = (1/n) sum_j R_j^2 with R_j the column rank sums,
    C = n J (J + 1)^2 / 4. H0 is also rejected when A = B.

    Parameters
    ----------
    X : n x J nested sequence
        Rows are subjects, columns the J dependent groups.

    Returns
    -------
    RichResult
        Keys: statistic, df1, df2, p_value, A, B, C.

    References
    ----------
    Friedman, M. (1937). JASA 32, 675-701.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (11.6).

    Examples
    --------
    >>> round(friedf([[9, 7, 12], [1, 10, 4], [8, 2, 1], [5, 6, 9]])["C"], 12)
    48.0
    """
    X = [[float(v) for v in row] for row in X]
    n = len(X)
    J = len(X[0]) if n else 0
    if n < 2 or J < 2 or any(len(r) != J for r in X):
        raise ValueError("need an n x J table with n, J >= 2")
    R = [_midranks(r) for r in X]
    A = sum(v * v for r in R for v in r)
    Bv = sum(sum(r[j] for r in R) ** 2 for j in range(J)) / n
    C = n * J * (J + 1) ** 2 / 4
    df1, df2 = J - 1, (n - 1) * (J - 1)
    if Bv == A:
        f, p = float("inf"), 0.0
    else:
        f = (n - 1) * (Bv - C) / (A - Bv)
        p = pf(f, df1, df2, lower_tail=False)
    return RichResult(
        title="Friedman's test (F form)",
        summary_lines=[("F", f), ("df", (df1, df2)), ("p", p)],
        payload={"statistic": f, "df1": df1, "df2": df2, "p_value": p, "A": A, "B": Bv, "C": C},
    )


def cheatsheet():
    return "friedf: Friedman's test, F = (n-1)(B-C)/(A-B). Wilcox (2017) eq (11.6)."
