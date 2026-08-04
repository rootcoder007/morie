# morie.fn -- function file (rootcoder007/morie)
"""Sum-of-squared-error loss for continuous outcomes."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ssello', 'mse_loss_continuous']


def ssello(Y, Yhat):
    """Sum-of-squared-error loss for continuous outcomes.

    Formula: L(w) = 0.5 * sum_i sum_j (yhat_ij - y_ij)^2

    Parameters
    ----------
    Y : array-like, shape (n, L)
        Observed values, one row per record; a flat vector is read as one column.
    Yhat : array-like, shape (n, L)
        Predicted values, same shape as the observed matrix.

    Returns
    -------
    RichResult
        ``loss``, ``mean_loss``, ``n``, ``L``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.7, pp. 400-403.  Read from the chapter PDF, not recalled.  The book notes that dividing by two is for convenience in the backpropagation gradient, and that it is also common to report the SSE divided by n times L; ``mean_loss`` is that per-cell value.
    """
    Y = C.mat(Y); H = C.mat(Yhat)
    n = len(Y)
    if n == 0 or n != len(H) or len(Y[0]) != len(H[0]):
        raise ValueError("Y and Yhat must be non-empty and the same shape")
    L = len(Y[0])
    loss = 0.5 * sum((H[i][j] - Y[i][j]) ** 2 for i in range(n) for j in range(L))
    return RichResult(payload={
        "loss": loss, "mean_loss": loss / (n * L), "n": n, "L": L,
        "method": "SSE loss, MVSML Sect. 10.7.1"})


mse_loss_continuous = ssello


def cheatsheet():
    return 'msebl: Sum-of-squared-error loss for continuous outcomes.'
