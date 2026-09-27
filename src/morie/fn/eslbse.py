"""Least-squares basis-expansion fit with pointwise standard errors (ESL sec 8.2.1)."""

import math

from ._richresult import RichResult
from ._stats_core import norm
from .linsys import _householder_ls
from .nlsgn import _inverse

__all__ = ["esl_basis_fit_se"]


def esl_basis_fit_se(H, y, Hnew=None, level=0.95, divisor="N"):
    r"""Fit :math:`\mu(x) = h(x)^T\beta` by least squares and give :math:`\widehat{se}[\hat\mu(x)]`.

    ESL eqs 8.2-8.4: :math:`\hat\beta = (H^TH)^{-1}H^Ty`,
    :math:`\hat\sigma^2 = \sum_i(y_i - \hat\mu(x_i))^2/N` and
    :math:`\widehat{se}[\hat\mu(x)] = [h(x)^T(H^TH)^{-1}h(x)]^{1/2}\hat\sigma`, with the
    pointwise band :math:`\hat\mu(x) \pm z_{(1+\text{level})/2}\,\widehat{se}`.
    ``divisor="N-p"`` uses the unbiased variance, as ``predict.lm``.

    Parameters
    ----------
    H : N x p nested sequence
        Basis evaluated at the training inputs (e.g. B-splines).
    y : sequence of N floats
    Hnew : M x p nested sequence, optional
        Basis at evaluation points (default H).
    level : float
    divisor : {"N", "N-p"}

    Returns
    -------
    RichResult
        ``beta``, ``sigma2``, ``fit``, ``se``, ``lower``, ``upper``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 8.2.1.
    """
    rows = [[float(v) for v in r] for r in H]
    yy = [float(v) for v in y]
    n, p = len(rows), len(rows[0])
    if divisor not in ("N", "N-p") or len(yy) != n or n <= p:
        raise ValueError("need matching H and y with N > p and divisor 'N' or 'N-p'")
    beta, rss = _householder_ls(rows, yy)
    s2 = rss / (n if divisor == "N" else n - p)
    G = _inverse([[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)])
    Q = rows if Hnew is None else [[float(v) for v in r] for r in Hnew]
    fit = [sum(b * v for b, v in zip(beta, r)) for r in Q]
    se = [math.sqrt(s2 * sum(r[a] * G[a][b] * r[b] for a in range(p) for b in range(p))) for r in Q]
    z = float(norm.ppf((1 + level) / 2))
    return RichResult(
        title="Basis fit with pointwise standard errors",
        summary_lines=[("sigma2", s2)],
        payload={
            "beta": beta,
            "sigma2": s2,
            "fit": fit,
            "se": se,
            "lower": [f - z * s for f, s in zip(fit, se)],
            "upper": [f + z * s for f, s in zip(fit, se)],
        },
    )


def cheatsheet():
    return "eslbse: se(mu(x)) = sqrt(h' (H'H)^-1 h) sigma, sigma^2 = RSS / N, ESL 8.4"
