# morie.fn -- function file (rootcoder007/morie)
"""Kernel ridge regression -- an alias for :mod:`krreg`.

``ledger/wave2/DUPMAP.tsv`` records ``krrFDA`` as a duplicate of
``krreg`` and it is: the same dual ridge solve on the same Gram matrix.
Only the argument order and the penalty's name differ.
"""

from .krreg import krreg

__all__ = ["kernel_ridge_regression"]


def kernel_ridge_regression(X, y, kernel="gaussian", lam=1.0, x_eval=None,
                            bandwidth=None):
    """Ridge regression carried out on inner products instead of features.

    The dual form is what makes a kernel method possible: the fit depends
    on the data only through the Gram matrix, so the implied feature space
    never has to be built.  The penalty earns its place twice over --
    it controls smoothness, and it is what keeps the near-singular Gram
    matrix of any smooth kernel invertible.

    Formula: ``alpha = (K + lambda I)^{-1} y``, prediction
    ``m(x0) = sum_i alpha_i K_h(x0 - x_i)``.

    This is an alias.  The solver lives in ``morie.fn.krreg``.

    Parameters
    ----------
    X : array-like, shape (n,)
        Predictor.
    y : array-like, shape (n,)
        Response.
    kernel : str, default 'gaussian'
        ``'gaussian'``, ``'epanechnikov'`` or ``'uniform'``.
    lam : float, default 1.0
        Ridge penalty, strictly positive.
    x_eval : array-like or None
        Evaluation points; defaults to ``X``.
    bandwidth : float or None
        Kernel bandwidth; Silverman's rule when omitted.

    Returns
    -------
    dict
        Whatever ``krreg.krreg`` returns, unchanged.

    References
    ----------
    Saunders, C., Gammerman, A. and Vovk, V. (1998).  Ridge regression
    learning algorithm in dual variables.  Proceedings of the 15th
    International Conference on Machine Learning, 515-521.
    """
    return krreg(X, y, x_eval, bandwidth=bandwidth, penalty=lam,
                 kernel=kernel)


def cheatsheet():
    return "krrFDA: kernel ridge regression (alias of krreg)"
