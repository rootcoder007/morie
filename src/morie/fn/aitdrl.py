# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Log-likelihood of a Dirichlet sample.

For N independent compositions x_n on the open (D-1)-simplex,

    l(alpha | X) = N [ ln Gamma(sum_i alpha_i) - sum_i ln Gamma(alpha_i) ]
                   + sum_i (alpha_i - 1) sum_n ln x_{ni},

which is the log of the product of the Dirichlet densities with the
sufficient statistic sum_n ln x_{ni} factored out.  The stub cites Wilks
(1962), Mathematical Statistics, Wiley; that text was not retrievable
here, so the expression is the standard published form and is pinned by
closed forms instead: at alpha = (1, ..., 1) it reduces to
N ln Gamma(D) = N ln (D-1)! for every data set, and at D = 2 it is the
beta log-likelihood.

The score is returned as well, since it costs one digamma call per part
and makes the maximum-likelihood condition checkable:

    dl/dalpha_i = N [ psi(sum alpha) - psi(alpha_i) ] + sum_n ln x_{ni}.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["dirichlet_loglik"]

_SIMPLEX_TOL = 1e-8


def dirichlet_loglik(alpha, X):
    """Dirichlet log-likelihood and score at alpha.

    Parameters
    ----------
    alpha : array-like
        Strictly positive concentration parameters.
    X : array-like
        One composition, or a matrix whose N rows are compositions.

    Returns
    -------
    ll : the log-likelihood
    score : the D partial derivatives with respect to alpha
    sum_log_x : the sufficient statistic, one entry per part
    """
    aa = [float(v) for v in k.vec(alpha)]
    D = len(aa)
    if D < 2:
        raise ValueError("dirichlet_loglik: a composition needs at least 2 parts")
    for v in aa:
        if not (v > 0.0):
            raise ValueError("dirichlet_loglik: alpha must be strictly positive")
    try:
        first = X[0]
    except (TypeError, IndexError, KeyError):
        raise ValueError("dirichlet_loglik: X is empty")
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        rows = [[float(v) for v in r] for r in X]
    else:
        rows = [[float(v) for v in X]]
    N = len(rows)
    if N == 0:
        raise ValueError("dirichlet_loglik: X is empty")
    slx = [0.0] * D
    for r in rows:
        if len(r) != D:
            raise ValueError("dirichlet_loglik: a row of X has a length alpha does not match")
        s = 0.0
        for v in r:
            if not (v > 0.0):
                raise ValueError("dirichlet_loglik: X must lie strictly inside the simplex")
            s += v
        if abs(s - 1.0) > _SIMPLEX_TOL:
            raise ValueError("dirichlet_loglik: a row of X does not sum to one")
        for i in range(D):
            slx[i] += math.log(r[i])
    a0 = 0.0
    for v in aa:
        a0 += v
    lc = k.lgamma(a0)
    for v in aa:
        lc -= k.lgamma(v)
    ll = N * lc
    for i in range(D):
        ll += (aa[i] - 1.0) * slx[i]
    d0 = k.digamma(a0)
    score = [N * (d0 - k.digamma(aa[i])) + slx[i] for i in range(D)]
    gmax = 0.0
    for v in score:
        if abs(v) > gmax:
            gmax = abs(v)
    return RichResult(
        title="Dirichlet log-likelihood",
        summary_lines=[("N", N), ("ll", ll)],
        payload={
            "ll": ll,
            "estimate": ll,
            "score": score,
            "score_max_abs": gmax,
            "sum_log_x": slx,
            "log_const": lc,
            "N": N,
            "D": D,
            "method": "l = N[lnG(sum a) - sum lnG(a_i)] + sum_i (a_i-1) sum_n ln x_ni",
        },
    )


def cheatsheet():
    return "aitdrl: Log-likelihood of a Dirichlet sample"


# compact alias per ledger/NAMING.md
dirichletloglik = dirichlet_loglik
