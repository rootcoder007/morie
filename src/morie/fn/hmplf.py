# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Polynomial feature expansion up to a given degree."""

from itertools import combinations_with_replacement

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_polynomial_features"]


def geron_polynomial_features(X, degree, include_bias=True, interaction_only=False):
    """
    Polynomial feature expansion up to a given degree.

    Formula: phi(x) = [1, x, x^2, ..., x^d]

    With several inputs the expansion is every monomial of total degree
    at most ``degree``, cross terms included -- that is the whole point:
    a linear model on phi(x) can fit interactions a linear model on x
    cannot. The count grows as C(n + d, d), so the returned
    ``n_output_features`` is the combinatorial explosion Geron warns
    about, not an afterthought.

    Parameters
    ----------
    X : array-like, shape (m, n) or (m,)
        Input features.
    degree : int
        Maximum total degree (>= 1).
    include_bias : bool, default True
        Prepend the constant column.
    interaction_only : bool, default False
        Drop powers above 1 in any single feature.

    Returns
    -------
    result : RichResult
        Keys: features, powers, names, n_output_features, estimate, n,
        method.

    Examples
    --------
    >>> r = geron_polynomial_features([[2.0, 3.0]], 2)
    >>> [float(v) for v in r["features"][0]]
    [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
    >>> r["names"]
    ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
    >>> int(r["n_output_features"])
    6
    >>> [float(v) for v in geron_polynomial_features([[5.0]], 3)["features"][0]]
    [1.0, 5.0, 25.0, 125.0]

    References
    ----------
    Geron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_polynomial_features: X must be 2-D, got ndim={A.ndim}")
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("geron_polynomial_features: X is empty")
    d = int(degree)
    if d < 1:
        raise ValueError(f"geron_polynomial_features: degree must be >= 1, got {degree!r}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_polynomial_features: X contains non-finite values")

    n = A.shape[1]
    combos = []
    if include_bias:
        combos.append(())
    for k in range(1, d + 1):
        for c in combinations_with_replacement(range(n), k):
            if interaction_only and len(set(c)) != len(c):
                continue
            combos.append(c)

    powers = np.zeros((len(combos), n), dtype=int)
    for i, c in enumerate(combos):
        for j in c:
            powers[i, j] += 1
    feats = np.prod(A[:, None, :] ** powers[None, :, :], axis=2)

    names = []
    for row in powers:
        if row.sum() == 0:
            names.append("1")
            continue
        parts = [f"x{j}" if e == 1 else f"x{j}^{e}" for j, e in enumerate(row) if e]
        names.append(" ".join(parts))

    return RichResult(
        title="Polynomial features",
        summary_lines=[("Input features", n), ("Degree", d), ("Output features", len(combos))],
        interpretation="Output width is C(n+d, d); at n=100, d=3 that is 176851 columns, so degree is a budget.",
        payload={
            "features": feats,
            "powers": powers,
            "names": names,
            "n_output_features": int(len(combos)),
            "degree": d,
            "estimate": feats,
            "n": int(A.shape[0]),
            "method": "Monomial expansion of total degree <= d",
        },
    )


def cheatsheet():
    return "hmplf: Polynomial feature expansion up to given degree"
