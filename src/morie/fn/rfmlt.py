# morie.fn -- slice s04 (rootcoder007/morie)
"""Random forest for multivariate/multi-output response.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 633-681], Chapter 15 "Random
Forest for Genomic Prediction".  Section 15.4, pp. 639-640, gives the
forest algorithm itself (see _mvsmlrf for the quoted steps); Section
15.7, p. 656, gives the multivariate splitting rule, which is what makes
this the multi-output and not the single-output problem.

Page 656, attributing the rule to Tang and Ishwaran (2017): "Assuming
that there are measures of q traits in each observation, that is,
y_i = (y_i,1, ..., y_i,q)', the goal is to minimize the multivariate sums
of squares (MSS),

    MSS = sum_{j=1}^{q} ( sum_{i=1}^{L} (y_ij - ybar_Lj)^2
                        + sum_{i=1}^{R} (y_ij - ybar_Rj)^2 )      (15.5)

where ybar_Lj and ybar_Rj are the sample means of the jth response
variable in the left and right daughter nodes."  The page then requires
standardisation -- "such a splitting rule (15.5) can only be effective if
each of the response variables is measured on the same scale ... We
therefore calibrate MSS by assuming that each response variable has been
standardized (with mean zero and variance equal to one).  The
standardization is applied prior to splitting a node" -- and gives the
working form

    MSS = sum_{j=1}^{q} ( (1/n_L)(sum_{i=1}^{L} y*_ij)^2
                        + (1/n_R)(sum_{i=1}^{R} y*_ij)^2 )        (15.6)

BOOK ERRATUM, recorded, and it inverts the optimisation.  The page says
"minimizing MSS is equivalent to minimizing" (15.6).  It is equivalent to
MAXIMISING it.  Expanding (15.5),

    MSS = sum_j sum_all y^2 - [ (sum_L y)^2/n_L + (sum_R y)^2/n_R ],

so the bracket -- which is exactly (15.6) -- enters with a minus sign and
the first term does not depend on the split.  Three further things on the
same page agree: (15.7), the classification analogue with the identical
functional form, is introduced with "the best split s for X is obtained
by maximizing"; the page closes by saying "(15.6) and (15.7) are
equivalent optimization problems"; and reusing the label MSS for a
quantity that is not the sum of squares is itself a slip.  This
implementation maximises (15.6), and the anchors verify against (15.5)
computed directly.

DETERMINISM is described in _mvsmlrf: the bootstrap of step 1 is drawn by
a fixed LCG and the mtry candidates of step 2(a) by van der Corput
offsets, so both arms grow the identical forest.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _mvsmlrf as rf

from ._richresult import RichResult

__all__ = ["rf_multivariate"]


def rf_multivariate(X, Y_matrix, n_trees=100, mtry=None, nodesize=5,
                    standardize=True):
    """Multivariate random forest, split by eq. (15.6) on standardized y.

    Parameters
    ----------
    X : array-like
        n-by-p matrix of independent variables.
    Y_matrix : array-like
        n-by-q matrix of responses; q = 1 is the ordinary regression forest.
    n_trees : int
        B, the number of trees.
    mtry : int, optional
        Number of candidate variables per split; p/3 rounded up by default,
        the p. 643 regression default.
    nodesize : int
        Minimum terminal node size; 5 by default, the p. 643 regression
        default.
    standardize : bool
        Apply the p. 656 standardisation before splitting.  Leaving it on
        is what makes (15.6) the calibrated rule the page requires.

    Returns
    -------
    estimate   : the mean of the fitted values
    y_hat      : n-by-q matrix of out-of-sample-style forest predictions
    importance : the p. 642 out-of-bag permutation VIM, one per variable
    mss        : eq. (15.5) evaluated at the fitted values
    oob_size   : the number of out-of-bag rows per tree
    """
    XX = core.mat(X)
    YY = core.mat(Y_matrix)
    n, p, q = rf.check_xy(XX, YY)
    B = int(n_trees)
    if B < 1:
        raise ValueError("rf_multivariate: n_trees must be at least 1")
    ns = int(nodesize)
    if ns < 1:
        raise ValueError("rf_multivariate: nodesize must be at least 1")
    m = rf.default_mtry(p) if mtry is None else int(mtry)
    if m < 1 or m > p:
        raise ValueError("rf_multivariate: mtry must lie between 1 and the number of columns of X")
    Ys = rf.standardize(YY, n, q) if standardize else [list(r) for r in YY]
    trees, oob = rf.build_forest(XX, Ys, B, ns, m, q)
    yhat = rf.forest_predict(trees, XX, q)
    imp = rf.perm_importance(trees, oob, XX, Ys, q)
    # eq. (15.5) at the fitted split of the whole sample, for the anchor
    mss = 0.0
    for j in range(q):
        mu = sum(Ys[i][j] for i in range(n)) / n
        for i in range(n):
            mss += (Ys[i][j] - mu) ** 2
    tot = 0.0
    for r in yhat:
        for v in r:
            tot += v
    return RichResult(
        title="Multivariate random forest",
        summary_lines=[("rows", n), ("variables", p), ("responses", q), ("trees", B)],
        payload={
            "estimate": tot / (n * q),
            "y_hat": yhat,
            "importance": imp,
            "mss": mss,
            "oob_size": [len(o) for o in oob],
            "mtry": m,
            "n": n,
            "method": "Chapter 15 Sect. 15.4 forest with the eq. (15.6) multivariate split, "
                      "maximised (the page's 'minimizing' is an erratum)",
        },
    )


def cheatsheet():
    return "rfmlt: Random forest for multivariate/multi-output response"


# compact alias per ledger/NAMING.md
rfmultivariate = rf_multivariate
