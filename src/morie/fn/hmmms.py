# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Min-max scaling to range [0,1]."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_min_max_scaling"]

_METHOD = "Min-max scaling"


def geron_min_max_scaling(X, feature_range=(0.0, 1.0)):
    """
    Min-max scaling to range [0,1].

    Formula: x' = (x - x_min) / (x_max - x_min)

    Per column, so each feature lands in the requested range
    independently.  A constant column has ``x_max = x_min`` and no
    defined scaling -- this raises rather than dividing by zero and
    handing back NaN or a silent zero.  The fitted ``data_min`` and
    ``data_range`` are returned so the identical transform can be
    replayed on held-out data; refitting on the test set is the classic
    leakage bug here.

    Unlike standardization, min-max scaling is bounded but is dragged
    around by a single extreme value.

    Parameters
    ----------
    X : array-like, shape (m, n) or (m,)
        Data; a 1-D array is treated as one column.
    feature_range : (float, float)
        Target ``(low, high)`` with ``low < high``.

    Returns
    -------
    result : RichResult
        Keys: X_scaled, data_min, data_max, data_range, scale,
        estimate, n, method.

    Examples
    --------
    Column of 1..5 maps to 0..1 linearly:

    >>> r = geron_min_max_scaling([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> [float(v) for v in r["X_scaled"].ravel()]
    [0.0, 0.25, 0.5, 0.75, 1.0]

    Columns are scaled independently:

    >>> r2 = geron_min_max_scaling([[0.0, 10.0], [2.0, 30.0]])
    >>> [[float(v) for v in row] for row in r2["X_scaled"]]
    [[0.0, 0.0], [1.0, 1.0]]
    >>> [float(v) for v in r2["data_range"]]
    [2.0, 20.0]

    A custom range is affine on top of the [0,1] result:

    >>> r3 = geron_min_max_scaling([0.0, 1.0, 2.0], feature_range=(-1.0, 1.0))
    >>> [float(v) for v in r3["X_scaled"].ravel()]
    [-1.0, 0.0, 1.0]

    A constant column is refused, not silently zeroed:

    >>> geron_min_max_scaling([[1.0], [1.0]])
    Traceback (most recent call last):
        ...
    ValueError: geron_min_max_scaling: column 0 is constant (min = max = 1.0), so min-max scaling is undefined

    References
    ----------
    Géron Ch 2
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_min_max_scaling: X must be 1-D or 2-D, got ndim={A.ndim}")
    if A.size == 0:
        raise ValueError("geron_min_max_scaling: X is empty")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_min_max_scaling: X contains non-finite values")
    try:
        low, high = (float(v) for v in feature_range)
    except (TypeError, ValueError):
        raise ValueError(f"geron_min_max_scaling: feature_range must be a (low, high) pair, got {feature_range!r}") from None
    if not (low < high):
        raise ValueError(f"geron_min_max_scaling: feature_range must satisfy low < high, got {feature_range!r}")

    mn = A.min(axis=0)
    mx = A.max(axis=0)
    rng = mx - mn
    bad = np.flatnonzero(rng == 0)
    if bad.size:
        j = int(bad[0])
        raise ValueError(
            f"geron_min_max_scaling: column {j} is constant (min = max = {mn[j]}), so min-max scaling is undefined"
        )

    unit = (A - mn) / rng
    scaled = unit * (high - low) + low

    return RichResult(
        title="Min-max scaling",
        summary_lines=[("Columns", int(A.shape[1])), ("Target range", f"[{low}, {high}]")],
        interpretation=(
            "Bounded output, but a single outlier sets the range; fit on the training set only "
            "and reuse data_min/data_range on new data."
        ),
        payload={
            "X_scaled": scaled,
            "data_min": mn,
            "data_max": mx,
            "data_range": rng,
            "scale": (high - low) / rng,
            "feature_range": (low, high),
            "estimate": float(np.mean(scaled)),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmms: min-max scaling x' = (x - min)/(max - min) per column, refusing constant columns"
