"""Least squares for several responses at once (ESL sec 3.2.4)."""

from ._richresult import RichResult
from .linsys import _householder_ls

__all__ = ["esl_multi_output_ls"]


def esl_multi_output_ls(X, Y, add_intercept=True):
    r"""Multiple-output least squares :math:`\hat B = (X^TX)^{-1}X^TY` (ESL eqs 3.34-3.39).

    With uncorrelated errors the K responses decouple: each column of
    :math:`\hat B` is the ordinary least-squares fit of that response
    (solved by Householder QR), so :math:`\hat Y = X\hat B`. The residual
    covariance :math:`\hat\Sigma = E^TE/(N - q)` (q columns in the design)
    is reported alongside; correlated errors do not change :math:`\hat B`.

    Parameters
    ----------
    X : N x p nested sequence
    Y : N x K nested sequence of responses
    add_intercept : bool
        Prepend a column of ones.

    Returns
    -------
    RichResult
        ``coefficients`` (q x K, rows are predictors), ``fitted`` (N x K),
        ``rss`` (per response), ``residual_covariance`` (K x K), ``n``, ``q``.

    References
    ----------
    Hastie, T., Tibshirani, R. & Friedman, J. (2009). The Elements of
    Statistical Learning (2nd ed.), sec. 3.2.4.
    """
    rows = [([1.0] if add_intercept else []) + [float(v) for v in r] for r in X]
    Ym = [[float(v) for v in r] for r in Y]
    n, q, K = len(rows), len(rows[0]), len(Ym[0])
    if len(Ym) != n or n <= q:
        raise ValueError("need X and Y with the same N rows and N greater than the design columns")
    cols = [_householder_ls(rows, [Ym[i][k] for i in range(n)]) for k in range(K)]
    B = [[cols[k][0][j] for k in range(K)] for j in range(q)]
    fitted = [[sum(rows[i][j] * B[j][k] for j in range(q)) for k in range(K)] for i in range(n)]
    E = [[Ym[i][k] - fitted[i][k] for k in range(K)] for i in range(n)]
    sig = [[sum(E[i][a] * E[i][b] for i in range(n)) / (n - q) for b in range(K)] for a in range(K)]
    return RichResult(
        title="Multiple-output least squares",
        summary_lines=[("K", K), ("q", q)],
        payload={
            "coefficients": B,
            "fitted": fitted,
            "rss": [c[1] for c in cols],
            "residual_covariance": sig,
            "n": n,
            "q": q,
        },
    )


def cheatsheet():
    return "eslmol: B = (X'X)^-1 X'Y column by column (QR); Sigma = E'E/(N - q)"
