"""Parametric bootstrap for a Gaussian basis-expansion fit (ESL sec 8.2.1)."""

import math
import random

from ._richresult import RichResult
from .eslbse import esl_basis_fit_se
from .linsys import _householder_ls

__all__ = ["esl_parametric_bootstrap"]


def esl_parametric_bootstrap(H, y, Hnew=None, B=200, level=0.95, seed=None):
    r"""Pointwise bands for :math:`\hat\mu(x)` from :math:`y^*_i = \hat\mu(x_i) + \varepsilon^*_i`.

    ESL eq 8.6: :math:`\varepsilon^*_i \sim N(0, \hat\sigma^2)` with
    :math:`\hat\sigma^2 = RSS/N`; each bootstrap response is refitted by least
    squares and the band is formed from the empirical quantiles (type 7) at
    each evaluation point. As B grows the bootstrap standard errors agree
    with eq 8.4, since the fit is linear in y. Draws use
    ``random.Random(seed)``.

    Parameters
    ----------
    H : N x p nested sequence
    y : sequence of N floats
    Hnew : M x p nested sequence, optional
    B : int
    level : float
    seed : int, optional

    Returns
    -------
    RichResult
        ``fit``, ``boot_se``, ``lower``, ``upper``, ``B``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 8.2.1; Efron, B. & Tibshirani,
    R. (1993). An Introduction to the Bootstrap, ch. 6.
    """
    base = esl_basis_fit_se(H, y, Hnew)
    rows = [[float(v) for v in r] for r in H]
    Q = rows if Hnew is None else [[float(v) for v in r] for r in Hnew]
    mu = [sum(b * v for b, v in zip(base["beta"], r)) for r in rows]
    sd = math.sqrt(base["sigma2"])
    rng = random.Random(seed)
    draws = [[] for _ in Q]
    for _ in range(int(B)):
        ys = [m + rng.gauss(0.0, sd) for m in mu]
        bb = _householder_ls(rows, ys)[0]
        for k, r in enumerate(Q):
            draws[k].append(sum(b * v for b, v in zip(bb, r)))

    def quant(v, q):
        s = sorted(v)
        h = (len(s) - 1) * q
        lo = math.floor(h)
        return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])

    a = (1 - level) / 2
    se = [math.sqrt(sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) for d in draws]
    return RichResult(
        title="Parametric bootstrap bands",
        summary_lines=[("B", int(B))],
        payload={
            "fit": base["fit"],
            "boot_se": se,
            "lower": [quant(d, a) for d in draws],
            "upper": [quant(d, 1 - a) for d in draws],
            "B": int(B),
        },
    )


def cheatsheet():
    return "eslpbt: y* = mu_hat + N(0, RSS/N) noise, refit B times, percentile bands; ESL 8.6"
