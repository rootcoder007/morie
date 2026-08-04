# morie.fn -- slice s04 (rootcoder007/morie)
"""Random forest mean decrease in impurity (MDI) variable importance.

NOT IN THE BOOK.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, was searched in full -- all seventeen page-range
volumes and the index, [Pages 683-691].  Chapter 15, volume [Pages
633-681], is the random forest chapter, and the only variable importance
it defines is the out-of-bag PERMUTATION measure of pp. 642-643, which
this package implements separately as rfpmi.  The words "mean decrease in
impurity" do not occur; impurity appears only as the splitting criterion,
where p. 643 names Gini for classification and the "weighted mean squared
error splitting criterion" for regression, both attributed to Breiman,
Friedman, Olshen and Stone (1984), Chapter 8.4 and Chapter 4.3.

The measure is therefore taken from the primary sources.

CITATION CARE, because the obvious attribution is wrong.  Mean decrease
in impurity is NOT defined in Breiman, L. (2001), Random forests,
*Machine Learning* 45(1), 5-32, doi:10.1023/A:1010933404324.  That paper
defines only the permutation measure; its Section 10 computes importance
by permuting a variable in the out-of-bag cases, and it never sums
impurity decreases over the nodes that split on a variable.

The formula implemented here -- the sum, over the nodes of a tree that
split on X_j, of the weighted impurity decrease, averaged over the
trees -- is the one stated in Louppe, G., Wehenkel, L., Sutera, A. and
Geurts, P. (2013), Understanding variable importances in forests of
randomized trees, *Advances in Neural Information Processing Systems* 26,
which writes it for a forest of M trees as

    Imp(X_j) = (1/M) sum_T sum_{t in T : v(s_t) = X_j}
               p(t) * Delta i(s_t, t),

with p(t) = N_t/N the fraction of samples reaching node t and
Delta i(s_t, t) = i(t) - (N_tL/N_t) i(t_L) - (N_tR/N_t) i(t_R).  The
impurity function i itself, and the practice of accumulating its
decrease, are from Breiman, Friedman, Olshen and Stone (1984),
*Classification and Regression Trees*, Wadsworth, the source the book's
own p. 643 cites for the splitting rules.  The impurity used here is the
within-node sum of squares, matching the regression rule the book names.

DETERMINISM is described in _mvsmlrf: the bootstrap of step 1 is drawn by
a fixed LCG and the mtry candidates of step 2(a) by van der Corput
offsets, so both arms grow the identical forest and accumulate the
identical decreases.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _mvsmlrf as rf

from ._richresult import RichResult

__all__ = ["rf_mdi_importance"]


def rf_mdi_importance(forest, X, y, mtry=None, nodesize=5):
    """Mean decrease in impurity, accumulated over the splitting nodes.

    Parameters
    ----------
    forest : int or None
        The number of trees to grow; None means 100.
    X : array-like
        n-by-p matrix of independent variables.
    y : array-like
        Length-n response, or an n-by-q matrix for the multivariate case.
    mtry, nodesize : optional
        As in rf_multivariate; the p. 643 regression defaults.

    Returns
    -------
    estimate   : the largest importance
    importance : one value per variable, in the raw impurity scale
    relative   : the same, scaled to sum to one
    ranking    : the variable indices ordered most to least important
    total      : the sum of the importances, which is the total impurity
                 the forest explains per tree
    """
    XX = core.mat(X)
    YY = core.mat(y) if isinstance(y[0], (list, tuple)) else [[float(v)] for v in y]
    n, p, q = rf.check_xy(XX, YY)
    B = 100 if forest is None else int(forest)
    if B < 1:
        raise ValueError("rf_mdi_importance: n_trees must be at least 1")
    ns = int(nodesize)
    if ns < 1:
        raise ValueError("rf_mdi_importance: nodesize must be at least 1")
    m = rf.default_mtry(p) if mtry is None else int(mtry)
    if m < 1 or m > p:
        raise ValueError("rf_mdi_importance: mtry must lie between 1 and "
                         "the number of columns of X")
    Ys = rf.standardize(YY, n, q)
    trees, _ = rf.build_forest(XX, Ys, B, ns, m, q)
    imp = rf.mdi_importance(trees, p)
    tot = 0.0
    for v in imp:
        tot += v
    rel = [v / tot if tot > 0.0 else 0.0 for v in imp]
    order = sorted(range(p), key=lambda j: (-imp[j], j))
    return RichResult(
        title="RF mean decrease in impurity",
        summary_lines=[("rows", n), ("variables", p), ("trees", B)],
        payload={
            "estimate": imp[order[0]],
            "importance": imp,
            "relative": rel,
            "ranking": order,
            "total": tot,
            "mtry": m,
            "n": n,
            "method": "MDI of Louppe et al. (2013) with the CART (1984) sum-of-squares "
                      "impurity; not in the book, and not in Breiman (2001) either",
        },
    )


def cheatsheet():
    return "rfmdi: Random forest mean decrease in impurity (MDI) variable importance"
