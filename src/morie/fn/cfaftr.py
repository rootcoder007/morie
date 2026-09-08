# morie.fn -- function file (rootcoder007/morie)
"""One-factor confirmatory factor analysis by maximum likelihood."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .cfafm2 import _cov_or_S, _fa_em

__all__ = ["cfa_one_factor"]


def cfa_one_factor(X, factor_structure=None):
    """
    CFA one factor, ML estimation

    Formula: X = lambda * F + eps; ML estimation

    The single factor is standardised, so the model covariance is
    lambda lambda' + Psi and the ML solution is reached by the
    Rubin-Thayer EM algorithm.  For three items the ML solution is the
    Spearman closed form lambda_i^2 = s_ij s_ik / s_jk, which is
    reported alongside as an independent read of the same data.

    Parameters
    ----------
    X : array-like
        n x p data matrix, or a p x p item covariance matrix.
    factor_structure : array-like or None
        Optional length-p 0/1 mask of which items load on the factor.
        None lets every item load.

    Returns
    -------
    result : dict
        Keys: estimate (variance explained), loadings, uniquenesses,
        fml, max_resid, communality, spearman, n_iter, p.

    References
    ----------
    Joreskog (1969), Psychometrika 34(2):183-202.
    Spearman (1904), Am. J. Psychology 15(2):201-292.
    """
    S = _cov_or_S(X)
    p = len(S)
    if p < 3:
        raise ValueError("a one-factor model needs at least three items")
    if factor_structure is None:
        mask = [[1] for _ in range(p)]
    else:
        m = core.vec(factor_structure)
        if len(m) != p:
            raise ValueError("factor_structure must have one entry per item")
        mask = [[1 if abs(v) > 0.0 else 0] for v in m]
        if sum(r[0] for r in mask) == 0:
            raise ValueError("factor_structure frees no loading at all")
    lam, psi, fml, resid, it = _fa_em(S, mask)
    load = [r[0] for r in lam]
    comm = [v * v for v in load]
    tr = sum(S[i][i] for i in range(p))
    # Spearman's tetrad solution from the first three items, sign matched
    num = S[0][1] * S[0][2]
    sp = math.sqrt(abs(num / S[1][2])) if S[1][2] != 0.0 else float("nan")
    if load[0] < 0.0:
        sp = -sp
    return RichResult(payload={
        "estimate": sum(comm) / tr,
        "loadings": load,
        "uniquenesses": psi,
        "fml": fml,
        "max_resid": resid,
        "communality": comm,
        "spearman": sp,
        "n_iter": it,
        "p": p,
        "method": "one-factor CFA, ML by EM",
    })


def cheatsheet():
    return "cfaftr: one-factor CFA, ML estimation"


# compact alias per ledger/NAMING.md
cfaonefactor = cfa_one_factor
