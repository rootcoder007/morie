# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feature importance via mean decrease in impurity across a forest."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_feature_importance_mdi"]

_METHOD = "Mean decrease in impurity (forest feature importance)"


def geron_feature_importance_mdi(tree_importances):
    r"""Average per-tree MDI into a forest-level importance.

    .. math::
        \mathrm{imp}(f) = \frac{1}{B}\sum_{b}\ \sum_{n \in T_b(f)}
        \frac{m_n}{m}\,\Delta\text{impurity}(n)

    Each tree's vector is normalised to sum to 1 *before* averaging.
    Without that, a deep tree with many splits would out-vote a shallow
    one purely on total impurity decrease rather than on where it put
    the decrease -- and every tree in a forest gets one vote.

    The known weakness comes with the definition: MDI is computed on the
    training splits, so it inflates high-cardinality and continuous
    features.  ``spread`` (the across-tree standard deviation) is
    reported for exactly that reason -- a feature that only some trees
    like has a large spread and a shaky importance.

    Parameters
    ----------
    tree_importances : array-like, shape (B, F)
        Per-tree impurity decreases, one row per tree, non-negative.

    Returns
    -------
    RichResult
        Payload keys ``importance``, ``spread``, ``ranking`` (feature
        indices, most important first), ``per_tree_normalized``,
        ``n_trees``, ``n_features``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 6, Feature Importance section.

    Examples
    --------
    One tree that splits the two features evenly, one that only ever
    uses feature 0:

    >>> r = geron_feature_importance_mdi([[0.5, 0.5], [1.0, 0.0]])
    >>> r["importance"]
    [0.75, 0.25]
    >>> r["ranking"]
    [0, 1]

    Rows are normalised first, so a tree with larger raw decreases does
    not get a larger vote:

    >>> r2 = geron_feature_importance_mdi([[5.0, 5.0], [1.0, 0.0]])
    >>> r2["importance"]
    [0.75, 0.25]
    >>> round(sum(r2["importance"]), 12)
    1.0
    """
    A = np.atleast_2d(np.asarray(tree_importances, dtype=float))
    if A.ndim != 2:
        raise ValueError(f"tree_importances must be 2-D (B, F), got shape {A.shape}.")
    if A.size == 0:
        raise ValueError("tree_importances is empty.")
    if not np.all(np.isfinite(A)):
        raise ValueError("tree_importances must be finite.")
    if np.any(A < 0):
        raise ValueError("impurity decreases are non-negative by construction; got a negative entry.")
    rowsum = A.sum(axis=1)
    dead = np.flatnonzero(rowsum == 0)
    if dead.size:
        raise ValueError(
            f"trees {dead.tolist()} have zero total impurity decrease (a stump that "
            f"never split); they cannot be normalised. Drop them."
        )

    N = A / rowsum[:, None]
    imp = N.mean(axis=0)
    spread = N.std(axis=0, ddof=1) if N.shape[0] > 1 else np.zeros(N.shape[1])
    order = np.argsort(-imp, kind="stable")

    return RichResult(
        title="Feature importance (MDI)",
        summary_lines=[("Trees", int(A.shape[0])), ("Features", int(A.shape[1])),
                       ("Top feature", int(order[0]))],
        payload={
            "importance": imp.tolist(),
            "spread": spread.tolist(),
            "ranking": order.tolist(),
            "per_tree_normalized": N.tolist(),
            "n_trees": int(A.shape[0]),
            "n_features": int(A.shape[1]),
            "estimate": imp.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grfim: per-tree impurity decrease normalised then averaged; spread flags shaky features"
