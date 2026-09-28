# morie.fn -- function file (rootcoder007/morie)
"""Issue salience and valence in spatial politics: punctuated issue attention (Jones and
Baumgartner), Schofield's valence convergence coefficient for the mean-voter equilibrium, and
Shepsle's structure-induced equilibrium under issue-by-issue jurisdictions."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["attention_punctuation", "valence_convergence", "structure_induced_equilibrium"]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def attention_punctuation(series):
    r"""Punctuation of issue attention: kurtosis and L-kurtosis of pooled percentage changes.

    ``series`` holds one attention (or budget) series per issue; the period
    changes ``(x_t - x_{t-1}) / x_{t-1}`` of all issues are pooled. Returns the
    moment kurtosis ``m4 / m2^2`` (3 for a normal distribution), the sample
    L-moments ``l1, l2`` and the L-kurtosis ``tau4 = l4 / l2`` (0.1226 for a
    normal distribution) from the unbiased probability-weighted moments of
    Hosking (1990). Values well above the normal benchmarks indicate the
    punctuated (leptokurtic) change distribution of Jones and Baumgartner.

    References
    ----------
    Jones, B. D. and Baumgartner, F. R. (2005). *The Politics of Attention*.
    University of Chicago Press.

    Hosking, J. R. M. (1990). L-moments: analysis and estimation of
    distributions using linear combinations of order statistics. *JRSS B* 52,
    105-124.

    Examples
    --------
    >>> r = attention_punctuation([[10, 11, 10, 30, 29], [5, 5, 6, 6, 2]])
    >>> round(r.l_kurtosis, 12), len(r.changes)
    (0.793266391022, 8)
    """
    ch = []
    for s in series:
        v = [float(x) for x in s]
        ch += [(v[t] - v[t - 1]) / v[t - 1] for t in range(1, len(v))]
    x = sorted(ch)
    n = len(x)
    m = ssum(x) / n
    m2 = ssum((v - m) ** 2 for v in x) / n
    m4 = ssum((v - m) ** 4 for v in x) / n
    b = [0.0] * 4
    for i, v in enumerate(x):
        w = 1.0
        for r in range(4):
            b[r] += w * v
            w *= (i - r) / (n - 1 - r) if r < n - 1 else 0.0
    b = [v / n for v in b]
    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return RichResult(
        payload={
            "changes": ch,
            "kurtosis": m4 / (m2 * m2),
            "l_moments": [l1, l2, l3 / l2, l4 / l2],
            "l_kurtosis": l4 / l2,
        }
    )


def valence_convergence(ideals, valence, beta):
    r"""Schofield's (2007) convergence coefficient for the mean-voter equilibrium with valence.

    In the multinomial-logit model with utilities ``lambda_j - beta ||x_i -
    z_j||^2``, all parties at the electoral mean is a local Nash equilibrium
    only if the lowest-valence party does not gain by moving. With ``rho_1 =
    exp(lambda_1) / sum_k exp(lambda_k)`` its vote share at the mean, ``A_1 =
    beta (1 - 2 rho_1)`` and the electoral covariance ``V* = (1/n) sum (x_i -
    xbar)(x_i - xbar)'``, the characteristic matrix is ``C_1 = 2 A_1 V* - I``;
    the mean is a local equilibrium iff ``C_1`` is negative definite, and a
    necessary condition is ``c = 2 A_1 trace(V*) < w`` (``w`` dimensions).

    References
    ----------
    Schofield, N. (2007). The mean voter theorem: necessary and sufficient
    conditions for convergent equilibrium. *Review of Economic Studies* 74,
    965-980.

    Examples
    --------
    >>> r = valence_convergence([[1.0, 0.0], [-1.0, 0.5], [0.0, -0.5]], [0.0, 1.0, 1.5], 1.0)
    >>> round(r.c, 12), r.local_equilibrium
    (1.260161158968, False)
    """
    X = _mat(ideals)
    n, w = len(X), len(X[0])
    lam = [float(v) for v in valence]
    mean = [ssum(r[k] for r in X) / n for k in range(w)]
    V = [[ssum((r[a] - mean[a]) * (r[b] - mean[b]) for r in X) / n for b in range(w)] for a in range(w)]
    j = min(range(len(lam)), key=lambda i: lam[i])
    mx = max(lam)
    den = ssum(math.exp(v - mx) for v in lam)
    rho = [math.exp(v - mx) / den for v in lam]
    A = beta * (1 - 2 * rho[j])
    C = [[2 * A * V[a][b] - (1.0 if a == b else 0.0) for b in range(w)] for a in range(w)]
    ev = [float(v) for v in np.linalg.eigvalsh(np.asarray(C, dtype=float)).tolist()]
    c = 2 * A * ssum(V[k][k] for k in range(w))
    return RichResult(
        payload={
            "c": c,
            "rho": rho,
            "A": A,
            "lowest": j,
            "characteristic_matrix": C,
            "eigenvalues": ev,
            "local_equilibrium": max(ev) < 0,
            "necessary": c < w,
        }
    )


def structure_induced_equilibrium(ideals, weights=None):
    r"""Shepsle's (1979) structure-induced equilibrium: the issue-by-issue weighted median.

    With separable preferences and a germaneness rule that lets each issue be
    amended only along its own dimension (committee jurisdictions), the point
    whose ``k``-th coordinate is the weighted median of the ideal points on
    dimension ``k`` is an equilibrium even when no majority-rule core exists.
    The weighted median is the smallest ideal with cumulative weight at least
    half of the total.

    References
    ----------
    Shepsle, K. A. (1979). Institutional arrangements and equilibrium in
    multidimensional voting models. *American Journal of Political Science* 23,
    27-59.

    Examples
    --------
    >>> structure_induced_equilibrium([[0.0, 3.0], [2.0, 1.0], [1.0, 0.0]])
    [1.0, 1.0]
    """
    X = _mat(ideals)
    n = len(X)
    wt = [1.0] * n if weights is None else [float(v) for v in weights]
    tot = ssum(wt)
    out = []
    for k in range(len(X[0])):
        order = sorted(range(n), key=lambda i: (X[i][k], i))
        acc = 0.0
        for i in order:
            acc += wt[i]
            if acc >= tot / 2:
                out.append(X[i][k])
                break
    return out


def cheatsheet() -> str:
    return "attention_punctuation / valence_convergence / structure_induced_equilibrium -> issue salience."
