# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preacher-Hayes bootstrap for specific and total indirect effects."""

from . import _array_core as np

from ._richresult import RichResult
from ._rrng_core import RRandom

__all__ = ["prehay", "preacher_hayes_indirect"]


def _paths(x, M, y):
    # a_k from OLS of each M_k on (1, X); b_1..b_j and c-prime from a
    # single OLS of Y on (1, X, M_1, ..., M_j).
    n, j = M.shape
    one = np.ones(n)
    Xa = np.stack([one, x], axis=1)
    AtA = Xa.T @ Xa
    a = np.asarray([float(np.linalg.solve(AtA, Xa.T @ M[:, k])[1])
                    for k in range(j)])
    Xb = np.concatenate([Xa, M], axis=1)
    cb = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    c_prime = float(cb[1])
    b = np.asarray([float(cb[2 + k]) for k in range(j)])
    return a, b, c_prime


def prehay(x, M, y, B=1000, alpha=0.05, seed=1):
    """
    Multiple-mediator indirect effects with percentile bootstrap CIs
    (Preacher and Hayes 2008).

    In the multiple mediation model M_k = i_k + a_k X and
    Y = i + c' X + sum_k b_k M_k, the specific indirect effect through
    mediator k is a_k b_k and the total indirect effect is
    sum_k a_k b_k, which equals c - c', the difference between the
    total and direct effects (Preacher and Hayes 2008, p. 880-881 and
    Figure 2). Each a_k comes from the regression of M_k on X; the b_k
    and c' come from one regression of Y on X and all mediators.

    B bootstrap resamples of the n cases are drawn with replacement;
    each specific indirect effect and the total are recomputed per
    resample, and percentile intervals use the (B * alpha/2)-th and
    (B * (1 - alpha/2) + 1)-th order statistics, the same rule the
    companion 2004 paper states for B = 1000 (25th and 976th values).
    Resampling uses the R-compatible Mersenne-Twister stream so the R
    arm reproduces identical resamples.

    Parameters
    ----------
    x : array-like
        Independent variable, length n.
    M : array-like, shape (n, j)
        Mediator matrix, one column per mediator.
    y : array-like
        Outcome, length n.
    B : int
        Bootstrap resamples (default 1000).
    alpha : float
        Two-sided miss probability (default 0.05).
    seed : int
        Seed for the R-compatible RNG.

    Returns
    -------
    result : RichResult
        Keys: estimate (total indirect), specific (per-mediator a_k
        b_k), a, b, c_prime, ci_lower, ci_upper (total), specific_lower,
        specific_upper (per mediator), se (bootstrap sd of total), B, n.

    References
    ----------
    Preacher, K. J. and Hayes, A. F. (2008), "Asymptotic and resampling
    strategies for assessing and comparing indirect effects in multiple
    mediator models", Behavior Research Methods 40(3), 879-891,
    doi:10.3758/BRM.40.3.879; specific and total indirect effects
    pp. 880-881, bootstrap pp. 883-884. Local source:
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    preacher-hayes-2008-asymptotic-resampling-multiple-mediators-BRM40.pdf
    Percentile rank rule: Preacher and Hayes (2004), Behavior Research
    Methods, Instruments, and Computers 36(4), 717-731, p. 722.
    """
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    if M.ndim == 1:
        M = M.reshape((-1, 1))
    n = len(x)
    if M.shape[0] != n or len(y) != n:
        raise ValueError("x, M, y must have matching first dimension")
    j = M.shape[1]
    B = int(B)
    if B < 2:
        raise ValueError("B must be at least 2")
    a, b, c_prime = _paths(x, M, y)
    spec = np.asarray([float(a[k] * b[k]) for k in range(j)])
    total = float(np.sum(spec))
    rng = RRandom(seed)
    boot_spec = [[] for _ in range(j)]
    boot_tot = []
    for _ in range(B):
        idx = [i - 1 for i in rng.sample_int(n, n, replace=True)]
        xr = np.asarray([x[i] for i in idx])
        Mr = np.stack([M[i] for i in idx], axis=0)
        yr = np.asarray([y[i] for i in idx])
        ar, br, _cr = _paths(xr, Mr, yr)
        tot_r = 0.0
        for k in range(j):
            v = float(ar[k] * br[k])
            boot_spec[k].append(v)
            tot_r += v
        boot_tot.append(tot_r)
    lo_i = int(B * (alpha / 2.0))          # 1-based rank, PH2004 p.722
    hi_i = int(B * (1.0 - alpha / 2.0)) + 1
    lo_i = min(max(lo_i, 1), B)
    hi_i = min(max(hi_i, 1), B)
    st = sorted(boot_tot)
    sl = []
    su = []
    for k in range(j):
        sk = sorted(boot_spec[k])
        sl.append(float(sk[lo_i - 1]))
        su.append(float(sk[hi_i - 1]))
    return RichResult(payload={
        "estimate": total,
        "specific": spec,
        "a": a, "b": b, "c_prime": c_prime,
        "ci_lower": float(st[lo_i - 1]),
        "ci_upper": float(st[hi_i - 1]),
        "specific_lower": np.asarray(sl),
        "specific_upper": np.asarray(su),
        "se": float(np.std(np.asarray(boot_tot), ddof=1)),
        "B": B, "n": n, "conf_level": 1.0 - alpha,
        "method": "Preacher-Hayes (2008) multiple-mediator percentile bootstrap",
    })


preacher_hayes_indirect = prehay


def cheatsheet():
    return "prehay(x, M, y, B, alpha, seed) -> specific and total indirect effects with percentile bootstrap CIs."
