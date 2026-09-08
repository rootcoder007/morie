# morie.fn -- function file (rootcoder007/morie)

"""Support vector machine with a kernel.

Implements eq. (9.46), (9.47) p.360 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ksvmdual", "mvsml_ridge_lasso_elastic_eq_9_46"]


def ksvmdual(X, y, T, kernel='linear', gamma=None, K=None):

    """maximize L(alpha) = sum_i alpha_i
    - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
    (eq. 9.46) subject to 0 <= alpha_i <= T and sum_i alpha_i y_i = 0
    (eq. 9.47).  Because the dual of the support vector classifier
    touches the data only through the inner products x_i . x_j, every
    instance of one can be replaced by a positive definite symmetric
    kernel, which implicitly defines an inner product in an enlarged
    feature space; that substitution is the whole difference between
    (9.44) and (9.46), and it is what turns the classifier into a
    support vector machine.  It also means nonvectorial inputs --
    sequences, trees, graphs -- can be handled, since only K is ever
    needed.  Keys: alpha, beta, beta0, objective, support_vectors,
    balance, bounded, at_bound, K, kernel.
    """

    res = RichResult(payload=_gp.ksvm_dual(X, y, T, kernel=kernel, gamma=gamma, K=K))

    return with_describe_pointer(res, "msm234")


mvsml_ridge_lasso_elastic_eq_9_46 = ksvmdual


def cheatsheet():
    return "msm234: Support vector machine with a kernel"
