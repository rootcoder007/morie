# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simple imputation: replace NaNs with a per-column statistic."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_simple_imputer"]

_METHOD = "Simple imputation (mean / median / most frequent)"

_STRATEGIES = ("mean", "median", "mode", "most_frequent")


def geron_simple_imputer(X, strategy="mean"):
    r"""Fill missing values column by column.

    .. math::
        x_{ij} := \mathrm{agg}(X_j) \quad \text{whenever } x_{ij}
        \text{ is NaN},\qquad \mathrm{agg}\in\{\text{mean, median, mode}\}

    The statistic is computed from the *observed* entries of that
    column only.  ``median`` is the safe default for skewed features --
    a single absurd value drags the mean, and every imputed row then
    carries that damage.

    The fitted ``statistics_`` are returned so the same fill can be
    replayed on a test set; recomputing them there leaks test
    information into training.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, n)
        Missing entries marked ``nan``.
    strategy : {"mean", "median", "mode", "most_frequent"}, optional

    Returns
    -------
    RichResult
        Payload keys ``imputed``, ``statistics``, ``n_missing``,
        ``missing_by_column``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Data Cleaning / Imputation section.

    Examples
    --------
    The mean of the observed 1 and 3 fills the hole:

    >>> import numpy as np
    >>> r = geron_simple_imputer([[1.0], [np.nan], [3.0]])
    >>> r["imputed"]
    [[1.0], [2.0], [3.0]]
    >>> r["n_missing"]
    1

    One wild value shows why median is the safer default:

    >>> X = [[1.0], [2.0], [300.0], [np.nan]]
    >>> geron_simple_imputer(X, "mean")["statistics"]
    [101.0]
    >>> geron_simple_imputer(X, "median")["statistics"]
    [2.0]
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}.")
    A = np.asarray(X, dtype=float)
    vector = A.ndim == 1
    if vector:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim {A.ndim}.")
    if A.shape[0] == 0:
        raise ValueError("X has no rows.")
    if np.any(np.isinf(A)):
        raise ValueError("X contains +/-inf; only nan marks a missing value.")

    miss = np.isnan(A)
    all_missing = np.flatnonzero(miss.all(axis=0))
    if all_missing.size:
        raise ValueError(
            f"columns {all_missing.tolist()} are entirely missing, so no "
            f"statistic can be computed for them; drop those columns."
        )

    stats = np.empty(A.shape[1], dtype=float)
    for j in range(A.shape[1]):
        col = A[~miss[:, j], j]
        if strategy == "mean":
            stats[j] = col.mean()
        elif strategy == "median":
            stats[j] = np.median(col)
        else:
            vals, counts = np.unique(col, return_counts=True)
            stats[j] = vals[counts.argmax()]      # ties -> smallest value

    out = np.where(miss, stats, A)

    return RichResult(
        title=f"Simple imputer ({strategy})",
        summary_lines=[("Strategy", strategy), ("Missing filled", int(miss.sum()))],
        payload={
            "imputed": out.ravel().tolist() if vector else out.tolist(),
            "statistics": stats.tolist(),
            "n_missing": int(miss.sum()),
            "missing_by_column": miss.sum(axis=0).tolist(),
            "strategy": strategy,
            "estimate": out.ravel().tolist() if vector else out.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grimp: fill NaN per column with mean/median/mode of the observed entries"
