# morie.fn -- function file (rootcoder007/morie)
"""Cronbach's alpha (classical test theory).

Source: Cronbach, L. J. (1951), "Coefficient alpha and the internal
structure of tests", *Psychometrika* 16(3):297-334.  The article states
the coefficient as

    alpha = (k / (k - 1)) * (1 - sum_i V_i / V_t)

where k is the number of items, V_i is the variance of item i over
persons and V_t is the variance of the person total score.  That is the
form implemented here; the ledger records the same expression.

Variances are the unbiased (n - 1) sample variances, which is the
convention of every reference implementation of alpha and the one under
which the k/(k-1) factor gives the standard result.  Alpha is a function
of the ratio of the two variances, so the choice of divisor cancels
exactly whenever it is applied consistently to both.

No standard error is returned.  Interval estimates for alpha come from
Feldt (1965) / van Zyl, Neudecker & Nel (2000), neither of which was
obtained; rather than ship an unverified sampling distribution this
module reports the coefficient and both variance components so that a
caller may build an interval from a source it can check.
"""

from ._richresult import RichResult

__all__ = ["ctt_alpha_classic"]


def _column_var(rows, j, n):
    """Unbiased sample variance of column j of an n-row table."""
    s = 0.0
    for i in range(n):
        s += rows[i][j]
    m = s / n
    ss = 0.0
    for i in range(n):
        dv = rows[i][j] - m
        ss += dv * dv
    return ss / (n - 1)


def _total_var(rows, n, k):
    """Unbiased sample variance of the row totals."""
    tot = [0.0] * n
    for i in range(n):
        t = 0.0
        for j in range(k):
            t += rows[i][j]
        tot[i] = t
    m = sum(tot) / n
    ss = 0.0
    for v in tot:
        dv = v - m
        ss += dv * dv
    return ss / (n - 1), tot


def _as_table(X):
    rows = [[float(v) for v in row] for row in X]
    n = len(rows)
    if n < 2:
        raise ValueError("alpha needs at least two persons")
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        raise ValueError("every person must be scored on the same items")
    if k < 2:
        raise ValueError("alpha needs at least two items")
    return rows, n, k


def ctt_alpha_classic(X):
    """Cronbach's alpha for an n persons x k items score matrix.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Rows are persons, columns are items.

    Returns
    -------
    RichResult
        ``alpha``, ``estimate`` (the same number), ``item_var`` (the k
        item variances), ``sum_item_var``, ``total_var``, ``n_items``,
        ``n``.
    """
    rows, n, k = _as_table(X)
    iv = [_column_var(rows, j, n) for j in range(k)]
    sv = sum(iv)
    tv, _tot = _total_var(rows, n, k)
    if tv <= 0.0:
        raise ValueError("total score has zero variance; alpha is undefined")
    alpha = (k / (k - 1.0)) * (1.0 - sv / tv)
    return RichResult(payload={
        "alpha": float(alpha), "estimate": float(alpha),
        "item_var": [float(v) for v in iv], "sum_item_var": float(sv),
        "total_var": float(tv), "n_items": k, "n": n,
        "method": "Cronbach (1951) alpha = k/(k-1) (1 - sum V_i / V_t)"})


def cheatsheet():
    return "ctta1c: Cronbach (1951) coefficient alpha"
