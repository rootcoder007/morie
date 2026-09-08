# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shannon entropy impurity for a node."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_entropy_impurity"]


def geron_entropy_impurity(y):
    """
    Shannon entropy impurity for a node.

    Formula: H = -sum_k p_k log2(p_k)

    Empty classes contribute nothing (``0 log 0 = 0``), a pure node scores
    0 and ``K`` equally frequent classes score ``log2(K)`` bits, returned
    as ``max_possible``.

    Parameters
    ----------
    y : array-like
        Class labels at the node.

    Returns
    -------
    result : RichResult
        Keys: entropy, entropy_nats, proportions, classes, counts,
        max_possible, estimate, n, method.

    Examples
    --------
    >>> geron_entropy_impurity([0, 0, 1, 1])["entropy"]
    1.0
    >>> geron_entropy_impurity([5, 5, 5])["entropy"]
    0.0
    >>> round(geron_entropy_impurity([0, 0, 0, 1])["entropy"], 9)
    0.811278124
    >>> round(geron_entropy_impurity([0, 1, 2, 3])["max_possible"], 12)
    2.0

    References
    ----------
    Géron Ch 5
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("geron_entropy_impurity: y is empty; entropy is undefined for an empty node")
    classes, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    h = float(-np.sum(p * np.log2(p))) + 0.0
    h = 0.0 if h == 0 else h
    K = int(classes.size)
    return RichResult(
        title="Entropy impurity",
        summary_lines=[("Entropy (bits)", h), ("Classes", K)],
        interpretation="0 bits means pure; log2(K) bits means maximally mixed over K classes.",
        payload={
            "entropy": h,
            "entropy_nats": float(h * np.log(2.0)),
            "proportions": p.tolist(),
            "classes": classes.tolist(),
            "counts": counts.astype(int).tolist(),
            "n_classes": K,
            "max_possible": float(np.log2(K)),
            "estimate": h,
            "n": int(y.size),
            "method": "Shannon entropy H = -sum_k p_k log2 p_k",
        },
    )


def cheatsheet():
    return "hment: Shannon entropy impurity for a node"
