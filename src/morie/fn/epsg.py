# morie.fn -- slice s03 (rootcoder007/morie)
"""Epsilon-greedy action selection on a stationary bandit.

Source consulted (FETCHED from incompleteideas.net): Sutton, R. S. and
Barto, A. G. (2018).  *Reinforcement Learning: An Introduction*, 2nd
edition, section 2.2: "behave greedily most of the time, but every once
in a while, say with small probability eps, instead select randomly from
among all the actions with equal probability".  So a *specific* action a
is chosen with probability

    P(a) = 1 - eps + eps / k   if a is greedy,
           eps / k             otherwise

and the action values are the sample averages of section 2.4,

    Q_(n+1) = Q_n + (1/n) (R_n - Q_n).

DETERMINISM.  The exploration decisions are not drawn from a generator.
The van der Corput sequence supplies the uniforms, one per pull, so the
long-run share of exploratory pulls is eps by construction and the run
is reproducible in both arms.  Rewards are the arms' true means unless a
reward stream is supplied; nothing here consults a clock or a seed.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["epsilon_greedy"]


def epsilon_greedy(arms, epsilon=0.1, T=100, rewards=None, q0=0.0):
    """Run a deterministic eps-greedy bandit for T pulls.

    Parameters
    ----------
    arms : array-like
        True mean reward of each arm.
    epsilon : float
        Exploration probability.
    T : int
        Number of pulls.
    rewards : 2-D array-like, optional
        Realised reward per (pull, arm); the arm means are used when absent.
    q0 : float
        Optimistic initial value for every arm.

    Returns
    -------
    RichResult with payload:
        estimate    : mean reward per pull
        q           : final action-value estimates
        counts      : pulls per arm
        total_reward, regret
        p_greedy    : 1 - eps + eps/k, the probability of the greedy action
    """
    mu = k.vec(arms)
    kk = len(mu)
    e = float(epsilon)
    n = int(T)
    q = [float(q0)] * kk
    cnt = [0.0] * kk
    total = 0.0
    best = 0
    for a in range(1, kk):
        if mu[a] > mu[best]:
            best = a
    for t in range(n):
        u = k.vdc(t, 2)
        if u < e:
            a = int(k.vdc(t, 3) * kk)
            if a >= kk:
                a = kk - 1
        else:
            a = 0
            for j in range(1, kk):
                if q[j] > q[a]:
                    a = j
        rw = float(rewards[t][a]) if rewards is not None else mu[a]
        cnt[a] += 1.0
        q[a] = q[a] + (rw - q[a]) / cnt[a]
        total += rw
    return RichResult(
        title="Epsilon-greedy bandit",
        summary_lines=[("mean reward", total / n if n else float("nan"))],
        payload={
            "estimate": total / n if n else float("nan"),
            "q": q,
            "counts": cnt,
            "total_reward": total,
            "regret": n * mu[best] - total if kk else float("nan"),
            "p_greedy": 1.0 - e + e / kk if kk else float("nan"),
            "n": n,
            "method": "Epsilon-greedy with sample-average values (Sutton and Barto 2018, sec. 2.2-2.4)",
        },
    )


def cheatsheet():
    return "epsg: e-greedy exploration"


epsilongreedy = epsilon_greedy
