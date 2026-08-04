# morie.fn -- function file (rootcoder007/morie)
"""Median polish of a spatial grid."""

from math import fsum

from ._richresult import RichResult
from ._spx import mat, median

__all__ = [
    "spatial_detrending",
    "medpolish",
]


def spatial_detrending(values, grid=None, iters=10):
    """Tukey's median polish, used to strip a large-scale spatial trend.

    NOT IN SCHABENBERGER & GOTWAY. A fixed-string search of the book for
    "polish" returns nothing. The book's own remedy for a non-constant
    mean is the trend-surface model of Sec. 5.3.1 (ordinary least squares
    on a polynomial in the coordinates), and the difference matters: OLS
    trend surfaces are pulled by outliers, median polish is not, which is
    why the geostatistical literature reaches for it before computing a
    semivariogram on residuals.

    The algorithm is Tukey, J. W. (1977), *Exploratory Data Analysis*,
    Addison-Wesley: fit the additive decomposition

        y_uv = overall + row_u + col_v + residual_uv

    by alternately subtracting row medians and column medians until the
    sweeps stop moving anything, accumulating each sweep's medians into
    the row, column and overall effects.

    Two details that are easy to get wrong and are handled here: the
    OVERALL term must collect the median of the row effects and of the
    column effects at each sweep, or the effects drift and no longer sum
    to zero; and a sweep must run row-then-column in a fixed order, since
    median polish is not order-invariant and the two orders give different
    (both legitimate) fits.

    Parameters
    ----------
    values : 2-D array-like, or 1-D with `grid`
        Grid of observations, row-major.
    grid : (nrow, ncol), optional
        Shape, when `values` is supplied flat.
    iters : int
        Sweeps. Fixed, not tolerance-based, so both language arms do
        identical arithmetic.

    Returns
    -------
    RichResult
        ``overall``, ``row``, ``col``, ``residuals``, ``fitted``,
        ``abs_residual_sum``, ``nrow``, ``ncol``, ``n``, ``method``.
    """
    if grid is not None:
        flat = [float(t) for t in values]
        g = [int(t) for t in grid]
        if len(g) != 2 or g[0] < 1 or g[1] < 1:
            raise ValueError("`grid` must be (nrow, ncol), both positive")
        if len(flat) != g[0] * g[1]:
            raise ValueError("`values` has %d entries but `grid` asks for %d"
                             % (len(flat), g[0] * g[1]))
        y = [flat[r * g[1]:(r + 1) * g[1]] for r in range(g[0])]
    else:
        y = mat(values, "values")
    nr = len(y)
    nc = len(y[0])
    if nr < 2 or nc < 2:
        raise ValueError("median polish needs at least a 2 by 2 grid")
    iters = int(iters)
    if iters < 1:
        raise ValueError("`iters` must be positive")

    res = [list(r) for r in y]
    row = [0.0] * nr
    col = [0.0] * nc
    overall = 0.0
    for _ in range(iters):
        for i in range(nr):
            d = median(res[i])
            row[i] = row[i] + d
            for j in range(nc):
                res[i][j] = res[i][j] - d
        d = median(col)
        overall = overall + d
        for j in range(nc):
            col[j] = col[j] - d
        for j in range(nc):
            d = median([res[i][j] for i in range(nr)])
            col[j] = col[j] + d
            for i in range(nr):
                res[i][j] = res[i][j] - d
        d = median(row)
        overall = overall + d
        for i in range(nr):
            row[i] = row[i] - d

    fitted = [[overall + row[i] + col[j] for j in range(nc)]
              for i in range(nr)]

    return RichResult(payload={
        "overall": overall,
        "row": row,
        "col": col,
        "residuals": res,
        "fitted": fitted,
        "abs_residual_sum": fsum([abs(res[i][j]) for i in range(nr)
                                  for j in range(nc)]),
        "sweeps": iters,
        "resistant_to_outliers": True,
        "nrow": nr,
        "ncol": nc,
        "n": nr * nc,
        "method": ("Median polish (Tukey 1977, *Exploratory Data "
                   "Analysis*); NOT in Schabenberger & Gotway, whose "
                   "trend removal is the OLS trend surface of Sec. 5.3.1"),
    })


def cheatsheet():
    return "spdetr: median polish for spatial detrending"


# compact alias per ledger/NAMING.md
medpolish = spatial_detrending
