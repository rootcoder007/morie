# morie.fn -- function file (rootcoder007/morie)
"""SVM maximum margin hyperplane (hard margin).

Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0), eqs. (9.6)-(9.8) pp.344-346, dual
(9.32)-(9.33) p.349.  Read from the chapter-9 split PDF.

The same equations already have a canonical implementation in
``msm175.hardsvm``; this module is a re-export shim onto the shared
core rather than a second solver.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svm_hyperplane"]


def svm_hyperplane(X, y):
    """Maximum margin (hard margin) hyperplane.

    (9.6) maximizes the margin M subject to sum_j beta_j^2 = 1 and
    y_i(beta_0 + x_i'beta) >= M.  With the scale fixed by the norm
    constraint M = 1/||beta||, so (9.6) is equivalent to minimizing
    (1/2)||beta||^2 (9.7) subject to y_i(beta_0 + x_i'beta) >= 1
    (9.8); the full street is 2M = 2/||beta||.  Solved through the
    Wolfe dual (9.32)-(9.33), whose box bound is absent in the hard
    margin case.

    Parameters
    ----------
    X : (n, p) array-like of inputs.
    y : (n,) array-like of labels in {-1, +1}.

    Returns
    -------
    RichResult with keys estimate (the margin M), beta, beta0,
    margin, street_width, objective, norm_beta, functional_margin,
    min_functional_margin, constraint_ok, alpha, support_vectors,
    method.

    References
    ----------
    MVSML (2022) eqs. (9.6)-(9.8) pp.344-346.
    """
    f = dict(_gp.max_margin_classifier(X, y))
    f["estimate"] = float(f["margin"])
    f["method"] = "SVM maximum margin hyperplane (MVSML 2022 eqs. 9.6-9.8)"
    return with_describe_pointer(RichResult(payload=f), "svmhp")


def cheatsheet():
    return "svmhp: SVM maximum margin hyperplane (hard margin)"


# compact alias per ledger/NAMING.md
svmhyperplane = svm_hyperplane
