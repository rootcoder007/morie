# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exp3 adversarial bandit algorithm."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["exp3", "exp3_bandit"]

_METHOD = "Exp3 exponential-weight adversarial bandit"


def exp3_bandit(x, gamma_, T=None, seed=0):
    r"""
    Run Exp3 on a table of adversarial rewards.

    Exp3 ("Exponential-weight algorithm for Exploration and
    Exploitation", Auer, Cesa-Bianchi, Freund and Schapire 2002,
    figure 1) with parameter gamma in (0, 1]: weights start at
    w_i(1) = 1; at each trial t

        p_i(t) = (1 - gamma) w_i(t) / sum_j w_j(t) + gamma / K,

    an action i_t is drawn from p(t), reward x_{i_t}(t) in [0, 1] is
    received, and only the chosen action's importance-weighted reward
    updates its weight:

        xhat_j(t) = x_j(t) / p_j(t) if j = i_t else 0,
        w_j(t+1)  = w_j(t) exp( gamma xhat_j(t) / K ).

    Theorem 3.1 of the source bounds the expected weak regret.  The
    reward table plays the role of the adversary's fixed assignment
    (their Section 2 model).

    Determinism conventions (mirrored bit-exactly in the R arm): the
    draw consumes exactly one uniform per trial, by inverse CDF on the
    probabilities in index order (the .ghc_choice_p convention).

    Parameters
    ----------
    x : array-like, shape (T, K)
        Reward assignment: ``x[t, j]`` is the reward action j pays at
        trial t+1, in [0, 1].
    gamma_ : float
        Mixing/learning parameter gamma in (0, 1].
    T : int, optional
        Number of trials (default: all rows).
    seed : int
        SplitMix64 seed for the action draws.

    Returns
    -------
    result : dict
        Keys: ``estimate`` (0-based action with the largest final
        weight), ``actions`` (0-based drawn actions), ``rewards``
        (received rewards), ``probs`` ((T, K) action probabilities at
        each trial), ``weights`` (final weights), ``total_reward``,
        ``method``.

    References
    ----------
    Auer, P., Cesa-Bianchi, N., Freund, Y. and Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem. SIAM Journal on
    Computing 32(1), 48-77.  Algorithm: figure 1 (Section 3); regret
    bound: Theorem 3.1.  Local source:
    fetched-wave3/auer-cesabianchi-freund-schapire-2002-exp3-nonstochastic-bandit.pdf.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a (T, K) reward table")
    rows, K = x.shape
    T = rows if T is None else int(T)
    if T > rows:
        raise ValueError("x has only %d rows" % rows)
    g = float(gamma_)
    if not (0.0 < g <= 1.0):
        raise ValueError("gamma_ must be in (0, 1]")
    rng = np.random.default_rng(seed)
    w = [1.0] * K
    probs = np.zeros((T, K))
    actions = np.zeros(T)
    rewards = np.zeros(T)
    for t in range(T):
        tot = 0.0
        for j in range(K):
            tot += w[j]
        p = [(1.0 - g) * w[j] / tot + g / K for j in range(K)]
        for j in range(K):
            probs[t, j] = p[j]
        u = float(rng.uniform())
        c = 0.0
        i = K - 1
        for j in range(K):
            c += p[j]
            if u <= c:
                i = j
                break
        r = float(x[t, i])
        w[i] = w[i] * float(np.exp(g * (r / p[i]) / K))
        actions[t] = float(i)
        rewards[t] = r
    best = 0
    for j in range(1, K):
        if w[j] > w[best]:
            best = j
    return RichResult(payload={
        "estimate": float(best),
        "actions": actions,
        "rewards": rewards,
        "probs": probs,
        "weights": np.asarray(w),
        "total_reward": float(np.sum(rewards)),
        "method": _METHOD,
    })


exp3 = exp3_bandit


def cheatsheet():
    return "exp3(x, gamma_) -> Exp3 adversarial bandit on a (T, K) reward table (Auer et al 2002, fig 1)."
