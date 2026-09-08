# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Missing-value imputation using column median (numeric) or mode (categorical)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_imputation_median"]

_METHOD = "Median / mode imputation"


def geron_imputation_median(X, missing_values=None, add_indicator=True):
    """
    Missing-value imputation using column median (numeric) or mode (categorical).

    Formula: x_ij = median(x_j) if x_ij is missing

    Median rather than mean, because the mean is dragged by the same
    outliers that make a column worth imputing carefully in the first
    place; mode for non-numeric columns, where a median is undefined.

    The fitted statistics are returned so the *same* values can be
    applied to held-out data.  Recomputing the median on the test set is
    leakage, and it is the most common way this step goes wrong.

    A missingness indicator column is returned by default: imputation
    destroys the information that a value was missing, and that fact is
    often predictive on its own.

    A column that is entirely missing has no median at all and raises,
    rather than being filled with 0 -- there is no value to impute and
    pretending otherwise invents data.

    Parameters
    ----------
    X : array-like, shape (m, n) or (m,)
        Data.  Numeric arrays use NaN as the missing marker unless
        ``missing_values`` says otherwise; object arrays use ``None``
        and NaN.
    missing_values : scalar, optional
        Explicit missing marker (e.g. ``-999``).
    add_indicator : bool
        Also return a boolean matrix of what was missing.

    Returns
    -------
    result : RichResult
        Keys: X_imputed, statistics, indicator, n_missing,
        missing_fraction, estimate, n, method.

    Examples
    --------
    The median of 1, 3, 5, 9 is 4, and that is what fills the hole:

    >>> r = geron_imputation_median([[1.0], [3.0], [np.nan], [5.0], [9.0]])
    >>> [float(v) for v in r["X_imputed"].ravel()]
    [1.0, 3.0, 4.0, 5.0, 9.0]
    >>> float(r["statistics"][0])
    4.0

    The median resists an outlier that would move the mean a long way:
    the mean of ``1, 3, 5, 1000`` is 252.25 but the median is 4.

    >>> o = geron_imputation_median([[1.0], [3.0], [np.nan], [5.0], [1000.0]])
    >>> float(o["statistics"][0])
    4.0

    A sentinel value can be named explicitly:

    >>> s = geron_imputation_median([[1.0], [-999.0], [3.0]], missing_values=-999.0)
    >>> [float(v) for v in s["X_imputed"].ravel()]
    [1.0, 2.0, 3.0]

    Categorical columns get the mode:

    >>> c = geron_imputation_median(np.array([["a"], ["b"], [None], ["a"]], dtype=object))
    >>> [str(v) for v in c["X_imputed"].ravel()]
    ['a', 'b', 'a', 'a']

    An all-missing column is refused:

    >>> geron_imputation_median([[np.nan], [np.nan]])
    Traceback (most recent call last):
        ...
    ValueError: geron_imputation_median: column 0 is entirely missing, so there is no median to impute with

    References
    ----------
    Géron Ch 2
    """
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_imputation_median: X must be a non-empty 2-D array, got shape {A.shape}")
    m, n = A.shape

    numeric = np.issubdtype(A.dtype, np.number)
    out = A.astype(float).copy() if numeric else A.astype(object).copy()
    indicator = np.zeros((m, n), dtype=bool)
    stats = []

    for j in range(n):
        col = A[:, j]
        if numeric:
            miss = np.isnan(col.astype(float))
        else:
            miss = np.asarray([v is None or (isinstance(v, float) and np.isnan(v)) for v in col])
        if missing_values is not None:
            miss = miss | np.asarray([v == missing_values for v in col])
        indicator[:, j] = miss
        present = col[~miss]
        if present.size == 0:
            raise ValueError(
                f"geron_imputation_median: column {j} is entirely missing, so there is no median to impute with"
            )
        if numeric:
            fill = float(np.median(present.astype(float)))
        else:
            vals, counts = np.unique(present.astype(str), return_counts=True)
            fill = vals[int(np.argmax(counts))]
        stats.append(fill)
        if np.any(miss):
            out[miss, j] = fill

    stats = np.asarray(stats) if numeric else np.asarray(stats, dtype=object)
    n_missing = int(np.count_nonzero(indicator))

    return RichResult(
        title="Median / mode imputation",
        summary_lines=[
            ("Rows x columns", f"{m} x {n}"),
            ("Missing cells", n_missing),
            ("Missing fraction", float(n_missing) / (m * n)),
        ],
        interpretation=(
            "Fit these statistics on the training set only; recomputing them on held-out data leaks. "
            "The indicator preserves the fact that a value was absent."
        ),
        payload={
            "X_imputed": out,
            "statistics": stats,
            "indicator": indicator if add_indicator else None,
            "n_missing": n_missing,
            "missing_fraction": float(n_missing) / (m * n),
            "strategy": "median" if numeric else "mode",
            "estimate": float(n_missing) / (m * n),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmimp: median (numeric) / mode (categorical) imputation with reusable statistics and a missingness indicator"
