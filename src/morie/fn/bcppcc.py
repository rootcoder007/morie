"""Box-Cox lambda by the normal probability plot correlation."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["box_cox_ppcc"]


def _cor(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    return sab / math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))


def box_cox_ppcc(x, lambdas=None):
    r"""Pick the Box-Cox :math:`\lambda` whose transform is most normal on a QQ plot.

    For each :math:`\lambda`, :math:`T_\lambda(x) = (x^\lambda - 1)/\lambda`
    (:math:`\log x` at 0) is correlated with the normal quantiles of
    ``qqnorm`` (``ppoints``: :math:`(i - a)/(n + 1 - 2a)`, :math:`a = 3/8`
    for :math:`n \le 10`, else 1/2); the maximiser is returned (Hedderich,
    Sachs & Reynarowych 2023, eq 7.27 and its ``bcplot``; Filliben 1975).

    Parameters
    ----------
    x : sequence of float
        Positive data.
    lambdas : sequence of float, optional
        Candidate values, default -5, -4.9, ..., 2.

    Returns
    -------
    RichResult
        ``lambda``, ``cor`` (list, one per candidate), ``lambdas``.

    References
    ----------
    Box, G. E. P. & Cox, D. R. (1964). JRSS B 26, 211-252.
    """
    xs = [float(v) for v in x]
    if len(xs) < 3 or min(xs) <= 0:
        raise ValueError("need at least 3 positive values")
    lam = [-5 + i * 0.1 for i in range(71)] if lambdas is None else [float(v) for v in lambdas]
    n = len(xs)
    a = 3 / 8 if n <= 10 else 0.5
    q = [float(norm.ppf((i - a) / (n + 1 - 2 * a))) for i in range(1, n + 1)]
    cors = []
    for lm in lam:
        y = sorted(math.log(v) if lm == 0 else (v**lm - 1) / lm for v in xs)
        cors.append(_cor(q, y))
    best = lam[max(range(len(lam)), key=lambda k: (cors[k], -k))]
    return RichResult(
        title="Box-Cox by probability plot correlation",
        summary_lines=[("lambda", best), ("cor", max(cors))],
        payload={"lambda": best, "cor": cors, "lambdas": lam},
    )


def cheatsheet():
    return "bcppcc: argmax_lambda cor(qnorm(ppoints(n)), sort(T_lambda(x)))"
