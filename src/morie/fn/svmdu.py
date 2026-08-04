# morie.fn -- function file (rootcoder007/morie)
"""Wolfe dual objective of the support vector machine."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['svmwolfe', 'svm_dual_wolfe', 'svmdualwolfe']


def svmwolfe(alpha, X, y, K=None):
    """Wolfe dual objective of the support vector machine.

    Formula: L(alpha) = sum_i alpha_i - 0.5 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)

    Parameters
    ----------
    alpha : array-like
        Dual variables, length n.
    X : array-like, shape (n, p)
        One record per row.
    y : array-like
        Class labels coded +1 and -1.
    K : array-like or None
        Gram matrix; None uses the linear kernel x_i'x_j.

    Returns
    -------
    RichResult
        ``dual``, ``linear_term``, ``quadratic_term``, ``constraint_sum``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 9, Eq. (9.32) p. 349.  The dual depends on the data only through inner products, which is exactly what lets a kernel replace them; ``constraint_sum`` reports sum_i alpha_i y_i, which the dual problem constrains to zero.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    a = C.vec(alpha)
    yv = C.vec(y)
    if len(a) != len(yv):
        raise ValueError("alpha and y must have the same length")
    L = G.svm_dual_objective(a, X, yv, K=K)
    lin = sum(a)
    return RichResult(payload={
        "dual": L, "linear_term": lin, "quadratic_term": lin - L,
        "constraint_sum": sum(u * w for u, w in zip(a, yv)), "n": len(a),
        "method": "SVM Wolfe dual objective, MVSML Eq. (9.32)"})


svm_dual_wolfe = svmwolfe
svmdualwolfe = svmwolfe


def cheatsheet():
    return 'svmdu: Wolfe dual objective of the support vector machine.'
