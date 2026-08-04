# morie.fn -- function file (rootcoder007/morie)
"""SVM soft margin with slack variables (support vector classifier).

MVSML (2022) eqs. (9.34)-(9.37) pp.354-355; Wolfe primal (9.38) and
dual (9.44)-(9.45) pp.356-357.  Read from the chapter-9 split PDF.

Canonical implementation is ``msm218.softsvm``; this module is a
re-export shim onto the shared core, not a second solver.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svm_soft_margin"]


def svm_soft_margin(X, y, C):
    """Support vector classifier with slack variables.

    (9.34) maximizes M subject to sum_j beta_j^2 = 1 (9.35),
    y_i(beta_0 + sum_j beta_j x_ij) >= M(1 - zeta_i) (9.36), and
    zeta_i >= 0 with sum_i zeta_i <= T (9.37).  zeta_i = 0 puts
    observation i on the correct side of the margin, zeta_i > 0 breaks
    the margin and zeta_i > 1 lands on the wrong side of the
    hyperplane.  The dual (9.44)-(9.45) differs from the hard margin
    dual only by the box bound 0 <= alpha_i <= T.

    The book writes T both for the slack budget of (9.37) and for the
    box bound of (9.45).  Only (9.45) is directly solvable, so ``C``
    here is the box bound and the realized slack total is reported as
    ``slack_sum``.

    Parameters
    ----------
    X : (n, p) array-like of inputs.
    y : (n,) array-like of labels in {-1, +1}.
    C : float, the box bound T of (9.45).

    Returns
    -------
    RichResult with keys estimate (the margin), beta, beta0, margin,
    norm_beta, zeta, slack_sum, n_violating, n_misclassified, alpha,
    support_vectors, objective, method.

    References
    ----------
    MVSML (2022) eqs. (9.34)-(9.37) pp.354-355, (9.44)-(9.45) p.357.
    """
    f = dict(_gp.soft_margin_classifier(X, y, C))
    f["estimate"] = float(f["margin"])
    f["method"] = "SVM soft margin (MVSML 2022 eqs. 9.34-9.37)"
    return with_describe_pointer(RichResult(payload=f), "svmsl")


def cheatsheet():
    return "svmsl: SVM soft margin with slack variables (C-SVM)"


# compact alias per ledger/NAMING.md
svmsoftmargin = svm_soft_margin
