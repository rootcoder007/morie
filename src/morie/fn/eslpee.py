"""Prediction error of a linear least-squares fit (ESL sec 7.3)."""

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_linear_prediction_error"]


def esl_linear_prediction_error(X, x0, sigma2, bias2=0.0):
    r"""Expected prediction error of the least-squares fit at :math:`x_0` and on average in-sample.

    ESL eq 7.11: with :math:`h(x_0) = X(X^TX)^{-1}x_0`,
    :math:`\mathrm{Err}(x_0) = \sigma_\varepsilon^2 + \mathrm{Bias}^2 + \|h(x_0)\|^2\sigma_\varepsilon^2`;
    averaged over the training points (eq 7.12) the variance term is exactly
    :math:`(p/N)\sigma_\varepsilon^2`, p the number of columns of X.

    Parameters
    ----------
    X : N x p nested sequence
        Design (include the intercept column if the fit has one).
    x0 : sequence of p floats
    sigma2 : float
        Noise variance.
    bias2 : float
        Squared bias at x0 (0 for a correct linear model).

    Returns
    -------
    RichResult
        ``variance`` (:math:`\|h(x_0)\|^2\sigma^2`), ``err_x0``,
        ``in_sample_variance`` (:math:`p\sigma^2/N`), ``h_norm2``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 7.3.
    """
    rows = [[float(v) for v in r] for r in X]
    x0 = [float(v) for v in x0]
    n, p = len(rows), len(rows[0])
    if len(x0) != p or sigma2 < 0 or n <= p:
        raise ValueError("need x0 with p entries, sigma2 >= 0 and N > p")
    G = _inverse([[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)])
    g = [sum(G[a][b] * x0[b] for b in range(p)) for a in range(p)]
    hn2 = sum(sum(r[a] * g[a] for a in range(p)) ** 2 for r in rows)
    return RichResult(
        title="Linear-fit prediction error",
        summary_lines=[("err_x0", sigma2 + bias2 + hn2 * sigma2)],
        payload={
            "variance": hn2 * sigma2,
            "err_x0": sigma2 + bias2 + hn2 * sigma2,
            "in_sample_variance": p * sigma2 / n,
            "h_norm2": hn2,
        },
    )


def cheatsheet():
    return "eslpee: Err(x0) = s2 + bias2 + |X (X'X)^-1 x0|^2 s2; in-sample variance p s2 / N"
