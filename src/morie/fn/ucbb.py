# SPDX-License-Identifier: AGPL-3.0-or-later
"""UCB1 multi-armed bandit policy."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ucbb", "ucb_bandit"]

_METHOD = "UCB1 index policy on a fixed reward table"


def ucb_bandit(x, T=None):
    r"""
    Run the deterministic UCB1 policy on a table of realized rewards.

    UCB1 (Auer, Cesa-Bianchi and Fischer 2002, figure 1) is: play each
    machine once, then always play the machine j maximizing

        xbar_j + sqrt( 2 ln n / n_j )

    where xbar_j is the average reward obtained from machine j, n_j the
    number of times j has been played so far, and n the overall number
    of plays done so far.  The policy is deterministic given the
    observed rewards, so it is run here against a caller-supplied
    (T, K) table ``x`` whose entry ``x[t, j]`` is the reward machine j
    would pay at play t (support in [0, 1] in the source; Theorem 1
    gives the logarithmic regret bound).  Initialization plays machines
    0..K-1 in order at plays 1..K; ties in the index break to the
    lowest machine.

    Parameters
    ----------
    x : array-like, shape (T, K)
        Realized reward table; row t is the reward each machine would
        pay at play t+1.
    T : int, optional
        Number of plays (default: all rows; must be >= K and <= rows).

    Returns
    -------
    result : dict
        Keys: ``estimate`` (0-based machine with the highest final
        average reward; ties to the lowest index), ``actions`` (0-based
        machine chosen at each play), ``rewards`` (reward received at
        each play), ``means`` (final average reward per machine),
        ``counts`` (final n_j), ``index`` (final UCB index per machine,
        computed with n = T), ``total_reward``, ``method``.

    References
    ----------
    Auer, P., Cesa-Bianchi, N. and Fischer, P. (2002). Finite-time
    analysis of the multiarmed bandit problem. Machine Learning 47,
    235-256.  Policy: figure 1, p. 237; regret bound: Theorem 1.
    Local source:
    fetched-wave3/auer-cesabianchi-fischer-2002-ucb1-finite-time-ML47.pdf.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a (T, K) reward table")
    rows, K = x.shape
    T = rows if T is None else int(T)
    if T < K:
        raise ValueError("need at least K = %d plays" % K)
    if T > rows:
        raise ValueError("x has only %d rows" % rows)
    counts = [0] * K
    sums = [0.0] * K
    actions = np.zeros(T)
    rewards = np.zeros(T)
    for t in range(T):
        if t < K:
            j = t
        else:
            n = t  # plays done so far
            best = 0
            bestidx = -float("inf")
            for k in range(K):
                idx = sums[k] / counts[k] + float(
                    np.sqrt(2.0 * float(np.log(float(n))) / counts[k]))
                if idx > bestidx:
                    bestidx = idx
                    best = k
            j = best
        r = float(x[t, j])
        counts[j] += 1
        sums[j] += r
        actions[t] = float(j)
        rewards[t] = r
    means = np.zeros(K)
    index = np.zeros(K)
    for k in range(K):
        means[k] = sums[k] / counts[k]
        index[k] = means[k] + float(
            np.sqrt(2.0 * float(np.log(float(T))) / counts[k]))
    best = 0
    for k in range(1, K):
        if means[k] > means[best]:
            best = k
    return RichResult(payload={
        "estimate": float(best),
        "actions": actions,
        "rewards": rewards,
        "means": means,
        "counts": np.asarray([float(c) for c in counts]),
        "index": index,
        "total_reward": float(np.sum(rewards)),
        "method": _METHOD,
    })


ucbb = ucb_bandit


def cheatsheet():
    return "ucbb(x) -> deterministic UCB1 play sequence on a (T, K) reward table (Auer et al 2002, fig 1)."

# public names resolved by fn/_lazy_map.json
ucbbandit = ucb_bandit
