# morie.fn -- slice s04 (rootcoder007/morie)
"""Permutation-based RF variable importance (out-of-bag permutation).

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 633-681], Chapter 15, Section
15.4, pp. 642-643, read as rendered page images.

The book gives this measure in PROSE ONLY.  It states no equation and
attaches no equation number to it anywhere in the chapter, and the
chapter's numbered equations (15.5)-(15.8) are the splitting rules, not
the importance.  What it says is that the prediction error is computed on
the out-of-bag observations, then "the values of the jth variable are
randomly permuted in the OOB observations and j new PE is computed.  The
differences between the two are then averaged over all the trees, and
normalized by the standard deviation of the differences.  The variable
showing the largest decrease in prediction accuracy is the most important
variable."  Page 640 defines the out-of-bag set: "Each tree makes use of
around two-thirds (63.2%) of the observations to build the tree.  The
remaining observations are referred to as Out-Of-Bag (OOB)."

Implemented exactly as that prose reads:

    Imp(X_j) = mean_b [ PE_b(X_j permuted) - PE_b ] / sd_b [ same ],

with PE_b the mean squared error over tree b's own out-of-bag rows.  The
normalisation by the standard deviation across trees is the book's, and
is what makes the result a z-like score rather than a raw error
difference; pass normalise=False for the raw mean difference.

DETERMINISM.  The book says "randomly permuted".  The permutation used is
the reversal of the out-of-bag row order, which is deterministic, is a
genuine permutation, and is identical in both arms.  The out-of-bag sets
themselves come from the shared deterministic bootstrap described in
_mvsmlrf, whose out-of-bag fraction reproduces the book's own 36.8%.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _mvsmlrf as rf

from ._richresult import RichResult

__all__ = ["rf_permutation_importance"]


def rf_permutation_importance(forest, X, y, mtry=None, nodesize=5,
                              normalise=True):
    """Out-of-bag permutation VIM of Chapter 15 pp. 642-643.

    Parameters
    ----------
    forest : int or None
        The number of trees to grow.  None means 100.  A forest object is
        not accepted: this package's forests are grown deterministically
        from the data, so the number of trees is the only thing that
        needs to be carried.
    X : array-like
        n-by-p matrix of independent variables.
    y : array-like
        Length-n response, or an n-by-q matrix for the multivariate case.
    mtry, nodesize : optional
        As in rf_multivariate; the p. 643 regression defaults.
    normalise : bool
        Divide by the standard deviation of the per-tree differences, as
        the book's own sentence prescribes.

    Returns
    -------
    estimate   : the largest importance, the book's "most important variable"
    importance : one value per variable
    ranking    : the variable indices ordered most to least important
    oob_size   : the number of out-of-bag rows per tree
    """
    XX = core.mat(X)
    YY = core.mat(y) if isinstance(y[0], (list, tuple)) else [[float(v)] for v in y]
    n, p, q = rf.check_xy(XX, YY)
    B = 100 if forest is None else int(forest)
    if B < 2:
        raise ValueError("rf_permutation_importance: need at least two trees to normalise")
    ns = int(nodesize)
    if ns < 1:
        raise ValueError("rf_permutation_importance: nodesize must be at least 1")
    m = rf.default_mtry(p) if mtry is None else int(mtry)
    if m < 1 or m > p:
        raise ValueError("rf_permutation_importance: mtry must lie between 1 and "
                         "the number of columns of X")
    Ys = rf.standardize(YY, n, q)
    trees, oob = rf.build_forest(XX, Ys, B, ns, m, q)
    imp = rf.perm_importance(trees, oob, XX, Ys, q, normalise)
    order = sorted(range(p), key=lambda j: (-imp[j], j))
    return RichResult(
        title="RF permutation importance",
        summary_lines=[("rows", n), ("variables", p), ("trees", B)],
        payload={
            "estimate": imp[order[0]],
            "importance": imp,
            "ranking": order,
            "oob_size": [len(o) for o in oob],
            "mtry": m,
            "n": n,
            "method": "OOB permutation VIM of Chapter 15 pp. 642-643, prose only -- "
                      "the book states no equation for it",
        },
    )


def cheatsheet():
    return "rfpmi: Permutation-based RF variable importance (out-of-bag permutation)"
