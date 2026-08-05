# morie.fn -- function file (rootcoder007/morie)
"""McDonald's omega total from a factor solution."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["omega_total"]


def omega_total(X, factor_loadings):
    """
    McDonald's omega total

    Formula: omega = (sum lambda_i)^2 / Var(T) using factor loadings

    Var(T) is the variance of the total score, i.e. the sum of every
    entry of the item covariance matrix.  Equivalently
    omega = (sum lambda)^2 / ((sum lambda)^2 + sum theta) with
    theta_i = S_ii - lambda_i^2 the unique variances.  Under
    tau-equivalence (all loadings equal) omega coincides with Cronbach's
    alpha; otherwise omega is the larger of the two.

    Parameters
    ----------
    X : array-like
        Either an n x p data matrix or a p x p item covariance matrix
        (detected by squareness and symmetry).
    factor_loadings : array-like
        Loading of each item on the general factor, length p.

    Returns
    -------
    result : dict
        Keys: estimate (omega), omega, alpha, var_total, uniquenesses, p.

    References
    ----------
    McDonald (1999), Test Theory: A Unified Treatment, Erlbaum, ch. 6.
    """
    lam = core.vec(factor_loadings)
    p = len(lam)
    if p < 2:
        raise ValueError("omega needs at least two items")
    M = core.mat(X)
    if len(M) == p and all(len(r) == p for r in M) and all(
            abs(M[i][j] - M[j][i]) < 1e-12 for i in range(p) for j in range(p)):
        S = M
    else:
        n = len(M)
        if n < 2:
            raise ValueError("need at least two observations to form a covariance")
        if len(M[0]) != p:
            raise ValueError("X and factor_loadings imply different item counts")
        mu = [sum(M[i][j] for i in range(n)) / n for j in range(p)]
        S = [[sum((M[i][a] - mu[a]) * (M[i][b] - mu[b]) for i in range(n)) / (n - 1)
              for b in range(p)] for a in range(p)]
    var_total = sum(S[i][j] for i in range(p) for j in range(p))
    if var_total <= 0.0:
        raise ValueError("total score variance is not positive")
    th = [S[i][i] - lam[i] * lam[i] for i in range(p)]
    sl = sum(lam)
    omega = sl * sl / var_total
    tr = sum(S[i][i] for i in range(p))
    alpha = p / (p - 1.0) * (1.0 - tr / var_total)
    return RichResult(payload={
        "estimate": omega,
        "omega": omega,
        "alpha": alpha,
        "var_total": var_total,
        "uniquenesses": th,
        "p": p,
        "method": "McDonald omega total",
    })


def cheatsheet():
    return "ctomeg: McDonald omega total"


# compact alias per ledger/NAMING.md
omegatotal = omega_total
