"""Regressions and partial correlations implied by a precision matrix (ESL sec 17.3)."""

import math

from ._richresult import RichResult

__all__ = ["esl_precision_regression"]


def esl_precision_regression(Theta):
    r"""Read the node-wise regressions off :math:`\Theta = \Sigma^{-1}`.

    ESL eqs 17.6-17.9: regressing :math:`X_j` on the rest has coefficients
    :math:`\beta_{jk} = -\theta_{jk}/\theta_{jj}` and residual variance
    :math:`1/\theta_{jj}`, so :math:`\theta_{12} = -\beta\,\theta_{22}`; the partial
    correlation of :math:`X_j, X_k` given the rest is
    :math:`-\theta_{jk}/\sqrt{\theta_{jj}\theta_{kk}}`, zero exactly when the edge is
    missing.

    Parameters
    ----------
    Theta : p x p nested sequence
        Positive-definite precision matrix.

    Returns
    -------
    RichResult
        ``coefficients`` (row j: regression of X_j on the others, 0 on the
        diagonal), ``residual_variance``, ``partial_correlation``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 17.3.
    """
    T = [[float(v) for v in r] for r in Theta]
    p = len(T)
    if any(len(r) != p for r in T) or min(T[j][j] for j in range(p)) <= 0:
        raise ValueError("Theta must be square with a positive diagonal")
    B = [[0.0 if j == k else -T[j][k] / T[j][j] for k in range(p)] for j in range(p)]
    P = [[1.0 if j == k else -T[j][k] / math.sqrt(T[j][j] * T[k][k]) for k in range(p)] for j in range(p)]
    return RichResult(
        title="Precision-matrix regressions",
        summary_lines=[("p", p)],
        payload={"coefficients": B, "residual_variance": [1 / T[j][j] for j in range(p)], "partial_correlation": P},
    )


def cheatsheet():
    return "eslprr: beta_jk = -theta_jk / theta_jj, var 1/theta_jj, partial corr -theta_jk / sqrt(theta_jj theta_kk)"
