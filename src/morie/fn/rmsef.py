# morie.fn -- function file (rootcoder007/morie)
"""Root mean squared error of a set of predictions."""

import math


from ._richresult import RichResult

__all__ = ['rmsetst', 'rmse_metric', 'rmsemetric']


def rmsetst(y, yhat):
    """Root mean squared error of a set of predictions.

    Formula: RMSE = sqrt((1/T) sum_i (y_i - yhat_i)^2)

    Parameters
    ----------
    y : array-like
        Numeric vector.
    yhat : array-like
        Predicted values, same length as y.

    Returns
    -------
    RichResult
        ``rmse``, ``mse``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Sect. 4.5.1, Eq. (4.1) p. 129 (MSE), Eq. (4.2) p. 129 (Pearson accuracy) and Eq. (4.3) p. 131 (MAE).  Read from the chapter PDF, not recalled.  The RMSE is named in the paragraph under Eq. (4.1) as the square root of MSE_TST.
    """
    y = [float(v) for v in y]
    yhat = [float(v) for v in yhat]
    if len(y) != len(yhat):
        raise ValueError("y and yhat must have the same length")
    n = len(y)
    if n == 0:
        raise ValueError("y must be non-empty")
    mse = sum((a - b) ** 2 for a, b in zip(y, yhat)) / n
    return RichResult(payload={
        "rmse": math.sqrt(mse), "mse": mse, "n": n,
        "method": "Test-set root mean squared error, MVSML Sect. 4.5.1"})


rmse_metric = rmsetst
rmsemetric = rmsetst


def cheatsheet():
    return 'rmsef: Root mean squared error of a set of predictions.'
