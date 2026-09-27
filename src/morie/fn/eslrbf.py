"""Radial basis function network with fixed centres and scales (ESL sec 6.7)."""

import math

from ._richresult import RichResult
from .linsys import _householder_ls

__all__ = ["esl_rbf_network"]


def esl_rbf_network(X, y, centers, scales, normalized=False, query=None):
    r"""Least-squares fit of :math:`f(x) = \beta_0 + \sum_j \beta_j D_j(x)` on Gaussian radial bases.

    :math:`D_j(x) = \exp\{-(x-\xi_j)^T(x-\xi_j)/\lambda_j^2\}` (ESL eqs 6.28-6.29)
    with the prototypes :math:`\xi_j` and scales :math:`\lambda_j` held fixed,
    so the criterion is linear in :math:`\beta`. ``normalized=True`` uses the
    renormalised bases :math:`h_j = D_j/\sum_k D_k` of eq 6.30, which avoid the
    holes between prototypes; they sum to one, so that fit has no separate
    intercept.

    Parameters
    ----------
    X : n x p nested sequence
    y : sequence of n floats
    centers : J x p nested sequence
    scales : sequence of J positive floats
    normalized : bool
    query : m x p nested sequence, optional

    Returns
    -------
    RichResult
        ``coefficients`` (intercept first unless normalized), ``fitted`` (rows of query),
        ``rss``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 6.7.
    """
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    C = [[float(v) for v in r] for r in centers]
    L = [float(v) for v in scales]
    if len(yy) != len(rows) or len(L) != len(C) or min(L) <= 0 or len(rows) <= len(C) + 1:
        raise ValueError("need matching X and y, one positive scale per centre and n > J + 1")

    def basis(x):
        d = [math.exp(-sum((a - b) ** 2 for a, b in zip(x, c)) / (s * s)) for c, s in zip(C, L)]
        if normalized:
            t = sum(d)
            return [v / t for v in d]
        return [1.0] + d

    beta, rss = _householder_ls([basis(r) for r in rows], yy)
    Q = rows if query is None else [[float(v) for v in r] for r in query]
    fit = [sum(b * v for b, v in zip(beta, basis(r))) for r in Q]
    return RichResult(
        title="RBF network (fixed prototypes)",
        summary_lines=[("rss", rss)],
        payload={"coefficients": beta, "fitted": fit, "rss": rss},
    )


def cheatsheet():
    return "eslrbf: LS on [1, exp(-|x - xi_j|^2 / lambda_j^2)] (normalised: divide by the row sum), ESL 6.28-6.30"
