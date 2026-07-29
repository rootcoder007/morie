# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-hot encoding of categorical variables."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_one_hot_encoding"]


def geron_one_hot_encoding(X, drop_first=False, categories=None):
    """
    One-hot encoding: represent a categorical variable with indicator columns.

    Formula: x_ik = I(category_i = k)

    K columns are produced by default. Their sum is a column of ones, so
    with an intercept in the model the design is collinear -- the dummy
    trap. ``drop_first=True`` emits K-1 columns instead, which is the
    right choice for a linear model with a bias and the wrong one for a
    tree or a neural net, where the redundant column costs nothing and
    keeps every level symmetric.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, p)
        Categorical values; each column is encoded independently.
    drop_first : bool, default False
        Emit K-1 columns per variable, omitting the first level.
    categories : sequence, optional
        Level order to use (one sequence per column when X is 2-D).
        Unseen values raise rather than being encoded as all-zeros.

    Returns
    -------
    result : RichResult
        Keys: encoded, categories, names, n_columns, estimate, n, method.

    Examples
    --------
    >>> r = geron_one_hot_encoding(["a", "b", "a"])
    >>> r["encoded"].tolist()
    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    >>> [str(c) for c in r["categories"][0]]
    ['a', 'b']

    Dropping the reference level leaves one indicator:

    >>> geron_one_hot_encoding(["a", "b", "a"], drop_first=True)["encoded"].tolist()
    [[0.0], [1.0], [0.0]]

    Every row of a full encoding sums to the number of variables:

    >>> r2 = geron_one_hot_encoding([["a", "x"], ["b", "y"]])
    >>> [float(v) for v in r2["encoded"].sum(axis=1)]
    [2.0, 2.0]

    References
    ----------
    Geron Ch 2
    """
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_one_hot_encoding: X must be 1-D or 2-D, got ndim={A.ndim}")
    m, p = A.shape
    if m == 0 or p == 0:
        raise ValueError("geron_one_hot_encoding: X is empty")

    if categories is None:
        cats = [np.unique(A[:, j]) for j in range(p)]
    else:
        cs = list(categories)
        if p == 1 and cs and np.ndim(cs[0]) == 0:
            cs = [cs]
        if len(cs) != p:
            raise ValueError(f"geron_one_hot_encoding: categories has {len(cs)} entries for {p} columns")
        cats = [np.asarray(c) for c in cs]

    blocks = []
    names = []
    for j in range(p):
        levels = cats[j]
        if levels.size < 1:
            raise ValueError(f"geron_one_hot_encoding: column {j} has no categories")
        idx = np.searchsorted(levels, A[:, j]) if np.all(levels[:-1] <= levels[1:]) else None
        eq = A[:, j][:, None] == levels[None, :]
        unseen = ~eq.any(axis=1)
        if unseen.any():
            bad = np.unique(A[:, j][unseen])
            raise ValueError(f"geron_one_hot_encoding: column {j} has values not in categories: {bad.tolist()}")
        block = eq.astype(float)
        keep = slice(1, None) if drop_first else slice(None)
        blocks.append(block[:, keep])
        names.extend([f"x{j}={lv}" for lv in levels[keep]])
        del idx

    enc = np.hstack(blocks)
    return RichResult(
        title="One-hot encoding",
        summary_lines=[("Variables", p), ("Columns", int(enc.shape[1])), ("Drop first", bool(drop_first))],
        interpretation="Keep all K columns for trees and nets; drop one for a linear model with an intercept.",
        payload={
            "encoded": enc,
            "categories": cats,
            "names": names,
            "n_columns": int(enc.shape[1]),
            "drop_first": bool(drop_first),
            "estimate": enc,
            "n": int(m),
            "method": "One-hot indicator encoding" + (" with reference level dropped" if drop_first else ""),
        },
    )


def cheatsheet():
    return "hmohe: One-hot encoding of categorical variables"
