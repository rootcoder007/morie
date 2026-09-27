"""Voting power in weighted majority games (exact enumeration).

Banzhaf, J. F. (1965). Weighted voting doesn't work. Rutgers Law Review 19, 317-343. Shapley,
L. S. and Shubik, M. (1954). A method for evaluating the distribution of power in a committee
system. American Political Science Review 48, 787-792. Deegan, J. and Packel, E. W. (1978). A
new index of power for simple n-person games. International Journal of Game Theory 7, 113-123.
Johnston, R. J. (1978). On the measurement of power. Environment and Planning A 10, 907-914.
Holler, M. J. (1982). Forming coalitions and measuring voting power. Political Studies 30,
262-271.
"""

import math

from ._richresult import RichResult

__all__ = ["power_indices"]


def power_indices(weights, quota=None):
    r"""Power of each player in the weighted game [q; w_1, ..., w_n] (S wins iff sum_S w >= q).

    Banzhaf: swings eta_i = #{S containing i: S wins, S - i loses}; absolute eta_i / 2^(n-1),
    normalised eta_i / sum eta. Shapley-Shubik: phi_i = sum over losing S (i not in S) with
    S + i winning of |S|! (n - |S| - 1)! / n!. Over the minimal winning coalitions M:
    Deegan-Packel DP_i = (1/|M|) sum_{S in M, i in S} 1/|S| and Holler's public good index
    #{S in M: i in S} / sum_j #{S in M: j in S}. Johnston: each winning coalition with c(S) >= 1
    critical members gives 1/c(S) to each, normalised by the number of such coalitions.
    The default quota is a simple majority of the total weight.

    Parameters
    ----------
    weights : sequence of non-negative numbers (at most 20 players)
    quota : float, optional

    Returns
    -------
    RichResult
        Keys: banzhaf, banzhaf_absolute, shapley_shubik, deegan_packel, johnston, holler,
        minimal_winning (index lists), min_winning_size.

    References
    ----------
    Banzhaf, J. F. (1965). Rutgers Law Review 19, 317-343.
    Shapley, L. S. and Shubik, M. (1954). American Political Science Review 48, 787-792.
    Deegan, J. and Packel, E. W. (1978). International Journal of Game Theory 7, 113-123.
    Johnston, R. J. (1978). Environment and Planning A 10, 907-914.

    Examples
    --------
    >>> power_indices([3, 2, 2], 4)["banzhaf"]
    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    """
    w = [float(v) for v in weights]
    n = len(w)
    if n == 0 or n > 20 or min(w) < 0:
        raise ValueError("need 1 to 20 non-negative weights")
    tot = 0.0
    for v in w:
        tot += v
    q = tot / 2 + 1e-12 if quota is None else float(quota)
    if quota is None:
        q = math.floor(tot / 2) + 1 if all(v == int(v) for v in w) else tot / 2 + 1e-12
    W = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        W[mask] = W[mask ^ low] + w[low.bit_length() - 1]
    win = [W[m] >= q - 1e-12 for m in range(1 << n)]
    swings = [0] * n
    ss = [0.0] * n
    fact = [math.factorial(k) for k in range(n + 1)]
    johnston = [0.0] * n
    vulnerable = 0
    minimal = []
    for mask in range(1 << n):
        if not win[mask]:
            continue
        size = bin(mask).count("1")
        crit = [i for i in range(n) if mask >> i & 1 and not win[mask ^ (1 << i)]]
        for i in crit:
            swings[i] += 1
            ss[i] += fact[size - 1] * fact[n - size] / fact[n]
        if crit:
            vulnerable += 1
            for i in crit:
                johnston[i] += 1 / len(crit)
        if len(crit) == size:
            minimal.append([i for i in range(n) if mask >> i & 1])
    ts = sum(swings)
    dp = [0.0] * n
    hol = [0] * n
    for S in minimal:
        for i in S:
            dp[i] += 1 / (len(S) * len(minimal))
            hol[i] += 1
    th = sum(hol)
    return RichResult(
        title="Voting power indices",
        summary_lines=[("players", n), ("quota", q)],
        payload={
            "banzhaf": [s / ts for s in swings] if ts else [0.0] * n,
            "banzhaf_absolute": [s / 2 ** (n - 1) for s in swings],
            "shapley_shubik": ss,
            "deegan_packel": dp,
            "johnston": [j / vulnerable for j in johnston] if vulnerable else [0.0] * n,
            "holler": [h / th for h in hol] if th else [0.0] * n,
            "minimal_winning": sorted(minimal),
            "min_winning_size": min((len(S) for S in minimal), default=None),
            "quota": q,
        },
    )


def cheatsheet():
    return "svpowr: Banzhaf, Shapley-Shubik, Deegan-Packel, Johnston and Holler power indices"
