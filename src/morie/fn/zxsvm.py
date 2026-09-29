# morie.fn -- function file (rootcoder007/morie)
"""Spatial SVM"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_svm(y, X, coords, *, gamma=None, C=10.0):
    r"""Spatial SVM

    Least-squares support vector machine regression (Suykens and Vandewalle
    1999) with the Gaussian kernel ``exp(-gamma ||z_i - z_j||^2)`` on the
    standardised spatial features ``[X, s1, s2]`` (default ``gamma = 1/p``):
    the dual solves ``[0, 1'; 1, K + I/C] [b; a] = [0; y]`` and predicts ``b +
    sum a_i k(z, z_i)`` -- kernel ridge regression with an unpenalised bias.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    gamma : float, optional
        Kernel width.
    C : float
        Regularisation.

    Returns
    -------
    DescriptiveResult
        ``value`` is the training RMSE; ``extra`` has ``fitted``, ``bias``, ``alpha``.

    References
    ----------
    Suykens, J. A. K. and Vandewalle, J. (1999). Least squares support vector machine classifiers.
    *Neural Processing Letters*, 9(3), 293-300.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> round(spatial_svm(y, X, S).value, 10)
    0.0695522118
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, 1)
    g = 1.0 / len(Z[0]) if gamma is None else float(gamma)
    model = sm.lssvm(Z, yv, g, float(C))
    fit = sm.lssvm_predict(model, g, Z)
    return DescriptiveResult(
        name="zxsvm", value=sm.rmse(yv, fit), extra={"fitted": fit, "bias": model[3], "alpha": model[4], "gamma": g}
    )


spat = spatial_svm
spatialsvm = spatial_svm


def cheatsheet() -> str:
    return "spatial_svm(...) -> Spatial SVM"
