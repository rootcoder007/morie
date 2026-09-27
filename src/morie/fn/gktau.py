"""Goodman and Kruskal's tau, the proportional reduction in prediction error."""

from ._richresult import RichResult

__all__ = ["goodman_kruskal_tau"]


def goodman_kruskal_tau(table):
    r"""Goodman and Kruskal's :math:`\tau` for an r x c table, in both directions.

    Hedderich, Sachs & Reynarowych (2023, eqs 3.9-3.10, p. 77): the
    proportional reduction in the error of predicting one variable's
    category once the other's is known,

    .. math::

        \tau_{C|R} = \frac{n\sum_{i,j} n_{ij}^2/n_{i\cdot} - \sum_j n_{\cdot j}^2}
                          {n^2 - \sum_j n_{\cdot j}^2},

    predicting the column category from the row, and :math:`\tau_{R|C}` with
    the roles exchanged. :math:`\tau = 0` under independence and 1 when one
    variable determines the other; the measure is not symmetric.

    Parameters
    ----------
    table : r x c array-like of counts

    Returns
    -------
    RichResult
        ``tau_col_given_row``, ``tau_row_given_col``, ``n``.

    References
    ----------
    Goodman, L. A. & Kruskal, W. H. (1954). Measures of association for
    cross classifications. JASA 49, 732-764. Hedderich, J., Sachs, L. &
    Reynarowych, Z. (2023). Applied Statistics: Methods Using R. Springer,
    eqs (3.9)-(3.10).
    """
    t = [[float(v) for v in row] for row in table]
    if len(t) < 2 or len(t[0]) < 2 or any(len(r) != len(t[0]) for r in t):
        raise ValueError("`table` must be an r x c array with r, c >= 2")
    if any(v < 0 for r in t for v in r):
        raise ValueError("counts must be non-negative")
    rows = [sum(r) for r in t]
    cols = [sum(r[j] for r in t) for j in range(len(t[0]))]
    n = sum(rows)
    if n <= 0:
        raise ValueError("the table is empty")

    def tau(mat, rsum, csum):
        num = n * sum(v * v / rs for row, rs in zip(mat, rsum) if rs > 0 for v in row) - sum(c * c for c in csum)
        den = n * n - sum(c * c for c in csum)
        return num / den if den > 0 else float("nan")

    tt = [list(c) for c in zip(*t)]
    return RichResult(
        title="Goodman-Kruskal tau",
        summary_lines=[("tau(col | row)", tau(t, rows, cols)), ("tau(row | col)", tau(tt, cols, rows))],
        payload={"tau_col_given_row": tau(t, rows, cols), "tau_row_given_col": tau(tt, cols, rows), "n": n},
    )


def cheatsheet():
    return "gktau: Goodman-Kruskal tau both ways, proportional reduction of prediction error"
