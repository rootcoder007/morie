# morie.fn -- function file (rootcoder007/morie)
"""FAST-MCD: the C-step algorithm for the minimum covariance determinant.

Rousseeuw, P. J. and Van Driessen, K. (1999), "A fast algorithm for the
minimum covariance determinant estimator", *Technometrics* 41(3),
212-223.  The algorithm and its key theorem were read from Hubert,
Debruyne and Rousseeuw (2018), "Minimum Covariance Determinant and
Extensions", arXiv:1709.07045, section "COMPUTATION", which restates
them verbatim:

    Theorem.  Take X = {x_1, ..., x_n} and let H1 be a subset of size
    h.  Put mu1 and S1 the empirical mean and covariance of the data in
    H1.  If |S1| != 0 define d1(i) = d(x_i, mu1, S1).  Now take H2 such
    that {d1(i); i in H2} are the h smallest distances, and compute mu2
    and S2 based on H2.  Then |S2| <= |S1|, with equality if and only
    if mu2 = mu1 and S2 = S1.

That is the C-step -- C for concentration.  The determinant therefore
never increases along the iteration, which is what makes it terminate,
and that monotonicity is asserted directly as an anchor rather than
taken on trust.

The published schedule, from the same section: apply only TWO C-steps
to each initial subset, keep the ten subsets with the lowest
determinants, and iterate only those to convergence.  Initial subsets
are grown from elemental (p+1)-subsets.

DETERMINISM.  The paper draws its elemental subsets at random.  This
implementation enumerates them in lexicographic order instead and
takes the first n_starts of them, so both language arms visit the same
candidates in the same sequence and return the same numbers, not
merely numbers of the same quality.  Nothing here draws a random
number.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k
from .mcdv import consistency_factor

from ._richresult import RichResult

__all__ = ["fast_mcd"]


def _grow(Xm, seed, h, n):
    """Grow an elemental subset to size h by nearest Mahalanobis distance.

    The paper's step: from a (p+1)-subset compute mu and S, then take
    the h observations with the smallest distances.  A singular S means
    the elemental subset is degenerate and the start is skipped.
    """
    mu, S = R.meancov(Xm, seed)
    dd = R.mahal2(Xm, mu, S)
    if dd is None:
        return None
    return sorted(R.osort(dd)[:h])


def fast_mcd(X, h=None, n_starts=500, max_iter=100, n_keep=10):
    """The FAST-MCD estimator.

    Parameters
    ----------
    X : array-like
        n-by-p data matrix.
    h : int, optional
        Subset size; defaults to [(n + p + 1) / 2].
    n_starts : int
        How many lexicographic elemental subsets to start from.
    max_iter : int
        Cap on the C-steps taken for each retained subset.
    n_keep : int
        How many best subsets to iterate to convergence; the paper uses 10.

    Returns
    -------
    estimate : the determinant reached
    center, cov_raw, cov, factor, subset, h, n, p
    n_starts_used : starts that produced a non-singular elemental covariance
    dets : the determinant chain of the winning subset, non-increasing
    """
    Xm = k.mat(X)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("fast_mcd: X is empty")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("fast_mcd: X has no columns")
    hh = R.mcd_h(n, p) if h is None else int(h)
    if hh <= p:
        raise ValueError("fast_mcd: h must exceed p, otherwise every subset is singular")
    if hh > n:
        raise ValueError("fast_mcd: h cannot exceed the number of observations")
    if n <= p + 1:
        raise ValueError("fast_mcd: need more than p + 1 observations")
    seeds = R.combos(n, p + 1, int(n_starts))
    cands = []
    used = 0
    for s in seeds:
        idx = _grow(Xm, s, hh, n)
        if idx is None:
            continue
        used += 1
        # two C-steps, as the paper prescribes
        for _ in range(2):
            step = R.cstep(Xm, idx, hh)
            if step is None:
                break
            idx = step[0]
        mu, S = R.meancov(Xm, idx)
        cands.append((R.ludet(S), idx))
    if not cands:
        raise ValueError("fast_mcd: every elemental subset was degenerate")
    order = R.osort([c[0] for c in cands])
    keep = [cands[i][1] for i in order[: int(n_keep)]]
    best_idx = None
    best_det = None
    best_chain = []
    for idx in keep:
        chain = []
        for _ in range(int(max_iter)):
            mu, S = R.meancov(Xm, idx)
            d = R.ludet(S)
            chain.append(d)
            step = R.cstep(Xm, idx, hh)
            if step is None:
                break
            if step[0] == idx:
                break
            idx = step[0]
        mu, S = R.meancov(Xm, idx)
        d = R.ludet(S)
        chain.append(d)
        if best_det is None or d < best_det:
            best_det = d
            best_idx = idx
            best_chain = chain
    mu, S = R.meancov(Xm, best_idx)
    c0 = consistency_factor(hh, n, p)
    Sc = [[S[a][b] * c0 for b in range(p)] for a in range(p)]
    return RichResult(
        title="FAST-MCD",
        summary_lines=[("n", n), ("p", p), ("h", hh), ("starts", used), ("det", best_det)],
        payload={
            "estimate": best_det,
            "center": mu,
            "cov_raw": S,
            "cov": Sc,
            "factor": c0,
            "subset": [float(v) for v in best_idx],
            "dets": best_chain,
            "n_starts_used": used,
            "h": hh,
            "n": n,
            "p": p,
            "method": "Rousseeuw-Van Driessen (1999) FAST-MCD, two C-steps per lexicographic elemental start, ten best iterated to convergence",
        },
    )


def cheatsheet():
    return "fastm: FAST-MCD by C-step concentration"
