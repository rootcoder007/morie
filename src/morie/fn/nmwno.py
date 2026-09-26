# morie.fn -- function file (rootcoder007/morie)
"""W-NOMINATE likelihood and fit at given ideal points and outcome locations."""

import math

from . import _stats_core as stats
from ._richresult import RichResult

__all__ = ["wnominate"]


def _rows(a, ncol):
    """Nested list of rows; a flat vector is one column."""
    a = a.tolist() if hasattr(a, "tolist") else list(a)
    if a and not isinstance(a[0], (list, tuple)):
        return [[float(v)] for v in a]
    return [[float(v) for v in r] for r in a]


def wnominate(votes, x, z_yea, z_nay, beta=15.0, w=None):
    r"""W-NOMINATE log-likelihood and fit statistics at given positions.

    Legislator :math:`i` at ideal point :math:`x_i` votes on roll call
    :math:`j` whose Yea and Nay outcomes sit at :math:`z_{jy}`, :math:`z_{jn}`.
    With Gaussian utility and normal errors (Poole & Rosenthal 1997;
    Poole 2005, ch. 4),

    .. math::

        U_{ijy} = \beta \exp\!\Big(-\tfrac12 \sum_k w_k^2 (x_{ik} - z_{jyk})^2\Big),
        \qquad P(\text{Yea}) = \Phi(U_{ijy} - U_{ijn}).

    This evaluates the likelihood of an observed vote matrix at supplied
    positions -- the objective that W-NOMINATE's alternating estimation
    maximises -- and the two fit statistics reported with it: the
    geometric mean probability :math:`\mathrm{GMP} = \exp(\ell / N)` and
    the share of votes correctly classified. It does not estimate the
    positions.

    Parameters
    ----------
    votes : array-like, legislators x roll calls
        1 = Yea, 0 = Nay, NaN = not voting (skipped).
    x : array-like, legislators x dimensions (a vector for one dimension)
    z_yea, z_nay : array-like, roll calls x dimensions
    beta : float
        Signal-to-noise weight (W-NOMINATE's default start is 15).
    w : sequence, optional
        Dimension salience weights (default all 1).

    Returns
    -------
    RichResult
        ``loglik``, ``GMP``, ``correct_classification``, ``n_correct``,
        ``n_total``, ``prob_yea`` (legislators x roll calls).

    References
    ----------
    Poole, K. T. (2005). Spatial Models of Parliamentary Voting.
    Cambridge University Press, ch. 4.

    Examples
    --------
    >>> r = wnominate([[1, 0], [0, 1]], [-0.5, 0.5], [-0.5, 0.5], [0.5, -0.5])
    >>> round(r["correct_classification"], 6)
    1.0
    """
    X = _rows(x, None)
    Zy = _rows(z_yea, None)
    Zn = _rows(z_nay, None)
    V = _rows(votes, None)
    p = len(X[0])
    if w is None:
        w = [1.0] * p
    if len(Zy) != len(Zn) or len(V) != len(X) or any(len(r) != len(Zy) for r in V):
        raise ValueError("wnominate: votes must be legislators x roll calls, matching x and z")
    ll = 0.0
    n_total = n_correct = 0
    prob = []
    for i, xi in enumerate(X):
        prow = []
        for j in range(len(Zy)):
            dy = sum((w[k] * (xi[k] - Zy[j][k])) ** 2 for k in range(p))
            dn = sum((w[k] * (xi[k] - Zn[j][k])) ** 2 for k in range(p))
            u = beta * (math.exp(-0.5 * dy) - math.exp(-0.5 * dn))
            prow.append(float(stats.norm.cdf(u)))
            v = V[i][j]
            if v != v:
                continue
            # log Phi(u) and log Phi(-u) directly: no clipping, no cancellation
            ll += float(stats.norm.logcdf(u)) if v == 1 else float(stats.norm.logcdf(-u))
            n_total += 1
            n_correct += int((u > 0) == (v == 1))
        prob.append(prow)
    gmp = math.exp(ll / n_total) if n_total else float("nan")
    cc = n_correct / n_total if n_total else float("nan")
    return RichResult(
        title="W-NOMINATE fit at given positions",
        summary_lines=[("log-likelihood", ll), ("GMP", gmp), ("correctly classified", cc)],
        payload={
            "loglik": ll,
            "GMP": gmp,
            "correct_classification": cc,
            "n_correct": n_correct,
            "n_total": n_total,
            "prob_yea": prob,
            "method": "wnominate",
        },
    )


wnom = wnominate


def cheatsheet() -> str:
    return "wnominate(votes, x, z_yea, z_nay) -> W-NOMINATE log-likelihood, GMP, classification"
