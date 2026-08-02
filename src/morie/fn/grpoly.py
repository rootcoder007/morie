# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Polynomial feature expansion up to a given degree (no interactions)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_polynomial_features"]

_METHOD = "Polynomial feature expansion (powers only)"


def geron_polynomial_features(X, degree, include_bias=True):
    r"""Append integer powers of every column.

    .. math::
        \phi(X) = [\,1,\; X,\; X^2,\; \dots,\; X^d\,]

    Powers only -- no cross terms.  That is what "no interactions" in the
    spec means, and it matters: with ``n`` features and degree ``d`` this
    gives ``n d`` columns instead of the :math:`\binom{n+d}{d}` explosion
    of the full expansion.  The linear model stays linear in its
    parameters; only the *features* are curved.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, n)
    degree : int
        Highest power, at least 1.
    include_bias : bool, optional
        Prepend a column of ones.

    Returns
    -------
    RichResult
        Payload keys ``features``, ``powers`` (per column, as
        ``(feature_index, power)``), ``n_features``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Polynomial Regression section.

    Examples
    --------
    >>> r = geron_polynomial_features([1.0, 2.0, 3.0], degree=2)
    >>> r["features"]
    [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0]]
    >>> r["powers"]
    [(0, 0), (0, 1), (0, 2)]

    Two features, degree 2, no bias: four columns, no cross term.

    >>> r2 = geron_polynomial_features([[2.0, 3.0]], degree=2, include_bias=False)
    >>> r2["features"]
    [[2.0, 3.0, 4.0, 9.0]]
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty 1-D or 2-D array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be at least 1, got {degree}.")

    cols = []
    powers = []
    if include_bias:
        cols.append(np.ones((A.shape[0], 1)))
        powers.append((0, 0))
    for p in range(1, degree + 1):
        for j in range(A.shape[1]):
            cols.append(A[:, j : j + 1] ** p)
            powers.append((j, p))
    F = np.hstack(cols)

    return RichResult(
        title="Polynomial features",
        summary_lines=[("Degree", degree), ("Columns", int(F.shape[1]))],
        payload={
            "features": F.tolist(),
            "powers": powers,
            "n_features": int(F.shape[1]),
            "estimate": F.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpoly: [1, X, X^2, ..., X^d] per feature, no cross terms -> n*d + 1 columns"
