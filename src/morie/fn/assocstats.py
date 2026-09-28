# morie.fn -- function file (rootcoder007/morie)
"""Descriptive association measures and moment identities: the variation ratio for nominal data,
Yule's Q and Y for 2 x 2 tables, the two one-sided tests (TOST) of correlation equivalence via
Fisher's z, and the mean and covariance of affine transformations of a random vector."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rrng_core import pnorm, qnorm

__all__ = ["variation_ratio", "yule_association", "tost_correlation", "affine_moments"]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def variation_ratio(x):
    r"""Variation ratio ``VR = 1 - N_modal / N_total`` of a nominal or ordinal variable.

    The proportion of cases outside the modal category (Freeman 1965);
    ``x`` holds the observations (any hashable labels).

    References
    ----------
    Freeman, L. C. (1965). *Elementary Applied Statistics*. Wiley.

    Wooditch, A., Johnson, N. J., Solymosi, R., Medina Ariza, J. and Langton, S.
    (2021). *A Beginner's Guide to Statistics for Criminology and Criminal
    Justice Using R*. Springer, eq 5.1.

    Examples
    --------
    >>> variation_ratio(["a", "b", "a", "c", "a"])
    0.4
    """
    vals = list(x)
    if not vals:
        raise ValueError("x is empty")
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return 1.0 - max(counts.values()) / len(vals)


def yule_association(table, conf_level=0.95):
    r"""Yule's Q and Y for a 2 x 2 table ``[[a, b], [c, d]]``.

    ``Q = (ad - bc) / (ad + bc)``, ``Y = (sqrt(ad) - sqrt(bc)) / (sqrt(ad) +
    sqrt(bc))`` and ``Q = 2Y / (1 + Y^2)``. Standard errors ``se(Q) = (1 - Q^2)
    sqrt(1/a + 1/b + 1/c + 1/d) / 2`` and ``se(Y) = (1 - Y^2) sqrt(...) / 4``
    with Wald intervals.

    References
    ----------
    Yule, G. U. (1912). On the methods of measuring association between two
    attributes. *JRSS* 75, 579-652.

    Bishop, Y. M. M., Fienberg, S. E. and Holland, P. W. (1975). *Discrete
    Multivariate Analysis*. MIT Press, section 11.2.

    Examples
    --------
    >>> r = yule_association([[20, 10], [5, 15]])
    >>> round(r.Q, 12), round(r.Y, 12)
    (0.714285714286, 0.420204102887)
    """
    (a, b), (c, d) = [[float(v) for v in r] for r in table]
    ad, bc = a * d, b * c
    Q = (ad - bc) / (ad + bc)
    Y = (math.sqrt(ad) - math.sqrt(bc)) / (math.sqrt(ad) + math.sqrt(bc))
    s = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d) if min(a, b, c, d) > 0 else math.inf
    z = qnorm(1 - (1 - conf_level) / 2)
    seq, sey = (1 - Q * Q) * s / 2, (1 - Y * Y) * s / 4
    return RichResult(
        payload={
            "Q": Q,
            "Y": Y,
            "se_Q": seq,
            "se_Y": sey,
            "ci_Q": [Q - z * seq, Q + z * seq],
            "ci_Y": [Y - z * sey, Y + z * sey],
        }
    )


def tost_correlation(r, n, low, high, alpha=0.05):
    r"""Two one-sided tests (TOST) of equivalence for a correlation (Fisher's z).

    ``z_low = (atanh r - atanh low) sqrt(n - 3)`` tests ``rho > low``
    (``p = 1 - Phi(z_low)``) and ``z_high = (atanh r - atanh high) sqrt(n - 3)``
    tests ``rho < high`` (``p = Phi(z_high)``); equivalence is declared when the
    larger p-value is below ``alpha``. The ``1 - 2 alpha`` interval is
    ``tanh(atanh r +- z_{1-alpha} / sqrt(n - 3))``.

    References
    ----------
    Lakens, D. (2017). Equivalence tests: a practical primer for t tests,
    correlations, and meta-analyses. *Social Psychological and Personality
    Science* 8, 355-362.

    Examples
    --------
    >>> t = tost_correlation(0.02, 200, -0.2, 0.2)
    >>> round(t.p_value, 12), t.equivalent
    (0.005162714042, True)
    """
    if not (-1 < low < high < 1 and -1 < r < 1 and n > 3):
        raise ValueError("need -1 < low < high < 1, -1 < r < 1 and n > 3")
    sq = math.sqrt(n - 3)
    zl = (math.atanh(r) - math.atanh(low)) * sq
    zh = (math.atanh(r) - math.atanh(high)) * sq
    pl, ph = pnorm(-zl), pnorm(zh)
    zc = qnorm(1 - alpha)
    p = max(pl, ph)
    return RichResult(
        payload={
            "z_low": zl,
            "z_high": zh,
            "p_low": pl,
            "p_high": ph,
            "p_value": p,
            "equivalent": p < alpha,
            "ci": [math.tanh(math.atanh(r) - zc / sq), math.tanh(math.atanh(r) + zc / sq)],
        }
    )


def affine_moments(mean, cov, A, c=None, B=None, d=None):
    r"""Mean and covariance of affine transformations of a random vector ``y``.

    ``E[Ay + c] = A mu + c`` and ``Cov(Ay + c, By + d) = A Sigma B'`` (``B``
    defaults to ``A``, giving ``Var(Ay + c) = A Sigma A'``); the constants do
    not affect the covariance.

    References
    ----------
    Searle, S. R. (1971). *Linear Models*. Wiley, section 2.5.

    Examples
    --------
    >>> r = affine_moments([1.0, 2.0], [[2.0, 0.5], [0.5, 1.0]], [[1.0, 1.0]], [3.0])
    >>> r.mean, r.cov
    ([6.0], [[4.0]])
    """
    mu = [float(v) for v in np.asarray(mean, dtype=float).ravel().tolist()]
    S, Am = _mat(cov), _mat(A)
    Bm = Am if B is None else _mat(B)
    cc = [0.0] * len(Am) if c is None else [float(v) for v in np.asarray(c, dtype=float).ravel().tolist()]
    k = len(mu)
    m = [ssum(Am[i][j] * mu[j] for j in range(k)) + cc[i] for i in range(len(Am))]
    AS = [[ssum(Am[i][j] * S[j][q] for j in range(k)) for q in range(k)] for i in range(len(Am))]
    C = [[ssum(AS[i][q] * Bm[r][q] for q in range(k)) for r in range(len(Bm))] for i in range(len(Am))]
    out = {"mean": m, "cov": C}
    if B is not None:
        dd = [0.0] * len(Bm) if d is None else [float(v) for v in np.asarray(d, dtype=float).ravel().tolist()]
        out["mean_B"] = [ssum(Bm[i][j] * mu[j] for j in range(k)) + dd[i] for i in range(len(Bm))]
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "variation_ratio / yule_association / tost_correlation / affine_moments -> association measures."
