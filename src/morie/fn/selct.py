# morie.fn -- function file (rootcoder007/morie)
"""Genomic selection accuracy: Pearson correlation of observed and predicted."""

import math


from ._richresult import RichResult

__all__ = ['predacc', 'genomic_selection_accuracy']


def predacc(y, yhat):
    """Genomic selection accuracy: Pearson correlation of observed and predicted.

    Formula: r = sum (yhat - mean yhat)(y - mean y) / sqrt(sum (yhat - mean yhat)^2 * sum (y - mean y)^2)

    Parameters
    ----------
    y : array-like
        Numeric vector.
    yhat : array-like
        Predicted values, same length as y.

    Returns
    -------
    RichResult
        ``accuracy``, ``r2``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 4, Sect. 4.5.1, Eq. (4.1) p. 129 (MSE), Eq. (4.2) p. 129 (Pearson accuracy) and Eq. (4.3) p. 131 (MAE).  Read from the chapter PDF, not recalled.  Eq. (4.2) is Pearson's correlation between the T test-set predictions and the T observed values; the book calls it the prediction accuracy in plant breeding.
    """
    y = [float(v) for v in y]
    yhat = [float(v) for v in yhat]
    if len(y) != len(yhat):
        raise ValueError("y and yhat must have the same length")
    n = len(y)
    if n < 2:
        raise ValueError("need at least two observations")
    my = sum(y) / n
    mh = sum(yhat) / n
    num = sum((b - mh) * (a - my) for a, b in zip(y, yhat))
    dy = sum((a - my) ** 2 for a in y)
    dh = sum((b - mh) ** 2 for b in yhat)
    if dy <= 0.0 or dh <= 0.0:
        raise ValueError("observed and predicted must both vary")
    r = num / math.sqrt(dh * dy)
    return RichResult(payload={
        "accuracy": r, "r2": r * r, "n": n,
        "method": "Pearson prediction accuracy, MVSML Eq. (4.2)"})


genomic_selection_accuracy = predacc


def cheatsheet():
    return 'selct: Genomic selection accuracy: Pearson correlation of observed and predicted.'
