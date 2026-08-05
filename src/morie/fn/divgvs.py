# morie.fn -- function file (rootcoder007/morie)
"""Fornell-Larcker discriminant validity criterion."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["discriminant_validity"]


def discriminant_validity(AVE, factor_correlations):
    """
    Fornell-Larcker discriminant validity

    Formula: sqrt(AVE_i) > correlation(F_i, F_j)

    A construct passes when the square root of its average variance
    extracted exceeds its correlation with every other construct, i.e.
    it shares more variance with its own indicators than with any other
    factor.  The reported margin is the smallest sqrt(AVE_i) - |r_ij|
    over all off-diagonal pairs: positive means the whole model passes.

    Parameters
    ----------
    AVE : array-like
        Average variance extracted, one entry per construct.
    factor_correlations : array-like
        k x k matrix of inter-construct correlations.

    Returns
    -------
    result : dict
        Keys: estimate (minimum margin), sqrt_ave, pass_factor,
        n_violations, discriminant, k.

    References
    ----------
    Fornell & Larcker (1981), J. Marketing Research 18(1):39-50.
    """
    ave = core.vec(AVE)
    k = len(ave)
    if k == 0:
        raise ValueError("empty input: no AVE values supplied")
    R = core.mat(factor_correlations)
    if len(R) != k or any(len(r) != k for r in R):
        raise ValueError("factor_correlations must be a k x k matrix")
    for v in ave:
        if v < 0.0 or v > 1.0:
            raise ValueError("AVE must lie in [0, 1]")
    sq = [math.sqrt(v) for v in ave]
    margin = float("inf")
    viol = 0
    pass_factor = []
    for i in range(k):
        ok = 1
        for j in range(k):
            if i == j:
                continue
            m = sq[i] - abs(R[i][j])
            if m < margin:
                margin = m
            if m <= 0.0:
                ok = 0
                viol += 1
        pass_factor.append(ok if k > 1 else 1)
    if k == 1:
        margin = float("nan")
    return RichResult(payload={
        "estimate": margin,
        "sqrt_ave": sq,
        "pass_factor": pass_factor,
        "n_violations": viol,
        "discriminant": 1 if viol == 0 else 0,
        "k": k,
        "method": "Fornell-Larcker discriminant validity",
    })


def cheatsheet():
    return "divgvs: Fornell-Larcker discriminant validity"


# compact alias per ledger/NAMING.md
discriminantvalidity = discriminant_validity
