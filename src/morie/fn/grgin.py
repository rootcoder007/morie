# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gini impurity at a tree node."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gini_impurity"]

_METHOD = "Gini impurity (Eq 5-1)"


def geron_gini_impurity(y):
    r"""Géron Eq 5-1.

    .. math::
        G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2

    Read it as a probability: draw two instances from the node with
    replacement; ``G`` is the chance they carry different labels.  So a
    pure node scores 0, and the worst case for ``K`` classes is
    :math:`1 - 1/K`, reached when the node is perfectly balanced --
    which is why Gini never gets to 1 for binary problems, only to 0.5.

    Parameters
    ----------
    y : array-like
        Class labels at the node, any hashable dtype.

    Returns
    -------
    RichResult
        Payload keys ``gini``, ``proportions``, ``classes``,
        ``max_possible``, ``majority_class``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 5, Eq 5-1 (Gini Impurity).  ``grent`` is the entropy
    alternative and ``grig`` the split criterion built on either.

    Examples
    --------
    Géron's iris node -- 0 setosa, 49 versicolor, 5 virginica -- has
    Gini ``1 - (49/54)^2 - (5/54)^2``:

    >>> r = geron_gini_impurity([0] * 49 + [1] * 5)
    >>> round(r["gini"], 6)
    0.168038

    A pure node scores zero; a balanced binary node scores the 0.5
    maximum:

    >>> geron_gini_impurity(["a", "a", "a"])["gini"]
    0.0
    >>> geron_gini_impurity([0, 1])["gini"]
    0.5
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("y is empty; impurity of an empty node is undefined.")
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    gini = float(1.0 - np.sum(p**2))
    K = int(classes.size)

    return RichResult(
        title="Gini impurity",
        summary_lines=[("Gini", gini), ("Classes", K), ("n", int(y.size))],
        payload={
            "gini": gini,
            "proportions": p.tolist(),
            "classes": classes.tolist(),
            "counts": counts.tolist(),
            "max_possible": 1.0 - 1.0 / K,
            "majority_class": classes[counts.argmax()].item(),
            "estimate": gini,
            "n": int(y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgin: G = 1 - sum p_k^2, the chance two draws differ -- Geron Eq 5-1"
