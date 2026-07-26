# morie.fn -- function file (rootcoder007/morie)
"""Banzhaf and Shapley-Shubik voting-power indices (Banzhaf 1965; Shapley & Shubik 1954)."""

from itertools import combinations
from math import factorial

import numpy as np

from ._richresult import RichResult

__all__ = ["voting_power_index", "vtpwr"]


def voting_power_index(x, quota=None):
    """Banzhaf (β) and Shapley-Shubik (φ) voting-power indices for a
    weighted-voting game [q; w_1, …, w_n].

    Banzhaf: β_i = |swings of i| / Σ_k |swings of k|
    Shapley-Shubik: φ_i = (n!)⁻¹ * Σ_{orderings} I{i is pivotal}

    Parameters
    ----------
    x : array-like (n,)
        Voter weights w_i.
    quota : float, optional
        Winning threshold q. Default is ``sum(w)/2`` nudged up by 1e-9, so a
        coalition wins on strictly more than half the total weight -- the
        simple-majority rule. The epsilon is what makes "strictly more"
        strict: an exact half-and-half split must lose, and floating-point
        equality alone would let it win.

        This docstring previously said ``ceil(sum(w)/2 + 1)``. That is a
        different and much harsher rule -- for weights (2, 1, 1) it gives
        q = 3, requiring three quarters of the total, not a majority -- and
        it was never what the code did.

    Returns
    -------
    RichResult with keys: banzhaf, shapley_shubik, quota, weights

    References
    ----------
    Shapley, L. S., & Shubik, M. (1954). A method for evaluating the
        distribution of power in a committee system. *American Political
        Science Review*, 48(3), 787-792. Read from the PDF: power is "the
        chance he has of being critical to the success of a winning
        coalition" (p.787); the index counts how often a member is *pivotal*
        when "the voting order of the members" is "chosen randomly" (p.788);
        and "where all voters have the same number of votes, they will each
        be credited with 1/nth of the power, there being n participants"
        (p.788) -- the symmetry property the tests pin directly.
    Banzhaf, J. F. (1965). Weighted voting doesn't work: A mathematical
        analysis. *Rutgers Law Review*, 19(2), 317-343.

    Notes
    -----
    This module previously cited "Armstrong Ch 10". That citation is wrong:
    *Analyzing Spatial Models of Choice and Judgment* has six chapters, and
    the strings "Banzhaf" and "Shapley" do not appear in any of its 320
    pages. Both primary sources are now in the library and both were read
    from the PDF rather than taken on trust.

    Shapley & Shubik note (p.788) that a chairman with only a tie-breaking
    vote "in an *even* committee ... is never pivotal", and put the US Senate
    presiding officer's index at exactly 1/97 under the strict scheme. That
    asymmetry is a property of the committee, not of this function, which
    scores whatever weighted game it is handed.
    """
    w = np.asarray(x, dtype=float).ravel()
    n = int(w.size)
    if n == 0:
        return RichResult(
            payload={
                "banzhaf": np.array([]),
                "shapley_shubik": np.array([]),
                "quota": np.nan,
                "weights": w,
                "method": "voting_power_index",
            }
        )
    total = float(w.sum())
    if quota is None:
        quota = total / 2.0 + 1e-9  # strict majority
    quota = float(quota)
    if n > 20:
        # Banzhaf via Monte Carlo for large n
        rng = np.random.default_rng(0)
        N_mc = 20000
        swings = np.zeros(n)
        for _ in range(N_mc):
            mask = rng.integers(0, 2, size=n).astype(bool)
            tot_in = w[mask].sum()
            for i in range(n):
                if mask[i]:
                    swings[i] += tot_in >= quota and (tot_in - w[i]) < quota
                else:
                    swings[i] += (tot_in + w[i]) >= quota and tot_in < quota
        banzhaf = swings / max(swings.sum(), 1)
        # Shapley-Shubik via permutation MC
        ss = np.zeros(n)
        for _ in range(N_mc):
            order = rng.permutation(n)
            cum = 0.0
            for k in order:
                prev = cum
                cum += w[k]
                if prev < quota <= cum:
                    ss[k] += 1
                    break
        shapley = ss / N_mc
        return RichResult(
            title="Voting power indices (MC, n > 20)",
            summary_lines=[("quota q", quota), ("n voters", n)],
            payload={
                "banzhaf": banzhaf,
                "shapley_shubik": shapley,
                "quota": quota,
                "weights": w,
                "method": "voting_power_index_mc",
            },
        )
    # Exact Banzhaf
    swings = np.zeros(n)
    for size in range(0, n + 1):
        for coalition in combinations(range(n), size):
            tot_in = float(w[list(coalition)].sum()) if size else 0.0
            for i in range(n):
                if i in coalition:
                    if tot_in >= quota and (tot_in - w[i]) < quota:
                        swings[i] += 1
                else:
                    if (tot_in + w[i]) >= quota and tot_in < quota:
                        swings[i] += 1
    banzhaf = swings / max(swings.sum(), 1)
    # Exact Shapley-Shubik by enumerating ordered pivots via combinations
    shapley = np.zeros(n)
    if n <= 10:
        from itertools import permutations

        n_perm = factorial(n)
        for order in permutations(range(n)):
            cum = 0.0
            for k in order:
                prev = cum
                cum += w[k]
                if prev < quota <= cum:
                    shapley[k] += 1
                    break
        shapley = shapley / n_perm
    else:
        # Combinatorial pivotal-coalition formula (Shapley 1953):
        # φ_i = Σ_{S not containing i, |S|=s, v(S)<q, v(S∪{i})≥q}
        #         s!(n-s-1)!/n!
        for i in range(n):
            others = [k for k in range(n) if k != i]
            for s in range(0, n):
                for S in combinations(others, s):
                    vS = float(w[list(S)].sum()) if S else 0.0
                    if vS < quota <= vS + w[i]:
                        shapley[i] += factorial(s) * factorial(n - s - 1) / factorial(n)
    return RichResult(
        title="Voting power indices (exact)",
        summary_lines=[
            ("quota q", quota),
            ("n voters", n),
            ("Banzhaf β", list(np.round(banzhaf, 4))),
            ("Shapley-Shubik φ", list(np.round(shapley, 4))),
        ],
        payload={
            "banzhaf": banzhaf,
            "shapley_shubik": shapley,
            "quota": quota,
            "weights": w,
            "method": "voting_power_index_exact",
        },
    )


vtpwr = voting_power_index


def cheatsheet():
    return "vtpwr: Banzhaf and Shapley-Shubik voting-power indices."


# CANONICAL TEST
# >>> r = voting_power_index([3,2,1], quota=4)
# >>> assert abs(r["banzhaf"].sum() - 1.0) < 1e-9
