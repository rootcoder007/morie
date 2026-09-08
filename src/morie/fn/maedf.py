# morie.fn -- function file (rootcoder007/morie)
"""Mean absolute error of a set of predictions."""


from ._richresult import RichResult

__all__ = ['maetst', 'mae_metric', 'maemetric']


def maetst(y, yhat):
    """Mean absolute error of a set of predictions.

    Formula: MAE = (1/T) sum_i |y_i - yhat_i|

    Parameters
    ----------
    y : array-like
        Numeric vector.
    yhat : array-like
        Predicted values, same length as y.

    Returns
    -------
    RichResult
        ``mae``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Sect. 4.5.1, Eq. (4.1) p. 129 (MSE), Eq. (4.2) p. 129 (Pearson accuracy) and Eq. (4.3) p. 131 (MAE).  Read from the chapter PDF, not recalled.
    """
    y = [float(v) for v in y]
    yhat = [float(v) for v in yhat]
    if len(y) != len(yhat):
        raise ValueError("y and yhat must have the same length")
    n = len(y)
    if n == 0:
        raise ValueError("y must be non-empty")
    mae = sum(abs(a - b) for a, b in zip(y, yhat)) / n
    return RichResult(payload={
        "mae": mae, "n": n,
        "method": "Test-set mean absolute error, MVSML Eq. (4.3)"})


mae_metric = maetst
maemetric = maetst


def cheatsheet():
    return 'maedf: Mean absolute error of a set of predictions.'
