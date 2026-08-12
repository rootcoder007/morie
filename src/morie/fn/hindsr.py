r"""Hindsight Experience Replay: goal relabelling of stored transitions.

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder,
P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017) "Hindsight
Experience Replay", arXiv:1707.01495.

The idea in one line: an episode that failed at the goal it was given
still *succeeded* at whatever goal it happened to reach, so store it
again under that goal. Algorithm 1 of the paper: after an episode
:math:`s_0, \dots, s_T` is finished, every transition
:math:`s_t \to s_{t+1}` goes into the buffer once with the original goal
:math:`g`,

.. math:: (s_t \| g,\ a_t,\ r(s_t, a_t, g),\ s_{t+1} \| g),

and then once more for each :math:`g' \in \mathcal{S}(\text{episode})`,
with the reward *recomputed* under :math:`g'`. Recomputation is what
makes it work, and it is legitimate because the goal influences the
agent's actions but not the environment dynamics -- so an off-policy
learner may be handed the relabelled transition as if it were real.

The four sampling strategies :math:`\mathcal{S}`, exactly as listed in
section 4.5:

``"final"``
    The goal achieved at the end of the episode, :math:`m(s_T)`. One
    extra copy per transition; ``k`` is ignored.
``"future"``
    ``k`` random states from the same episode observed *after* the
    transition being replayed. The paper's best strategy, and the only
    one that solves sliding; ``k = 4`` is its recommended value and the
    default here.
``"episode"``
    ``k`` random states from the same episode, before or after.
``"random"``
    ``k`` random states encountered anywhere in training so far. The
    paper reports this is the one strategy that does *not* solve
    pushing and pick-and-place.

The reward function is the paper's sparse one,

.. math:: r(s, a, g) = -\,[\,f_g(s') = 0\,],

i.e. :math:`0` when the achieved goal is within ``tol`` of :math:`g` and
:math:`-1` otherwise. Any other ``reward_fn`` may be passed; it is
called as ``reward_fn(s, a, s_next, g)``.

``m`` is the paper's mapping from a state to the goal it achieves; by
default the state itself (``m = identity``), which is the setting used
throughout the paper's experiments.

This module is the replay-buffer half of Algorithm 1 -- goal sampling
and relabelling -- which is the part that is HER. The other half is
"run any off-policy algorithm A on the buffer", and A is DDPG or DQN,
not something this file should be pretending to own.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hindsr", "her"]

_STRATEGIES = ("future", "final", "episode", "random")


def _as_states(seq, name):
    out = []
    for s in seq:
        v = np.atleast_1d(np.asarray(s, dtype=float))
        out.append([float(t) for t in v])
    if not out:
        raise ValueError("hindsr: %s must be non-empty" % name)
    return out


def _sparse_reward(s, a, s_next, g, tol):
    r"""r(s, a, g) = -[f_g(s') = 0], with f_g the goal test."""
    if len(s_next) != len(g):
        raise ValueError("hindsr: goal has length %d but state has %d; "
                         "pass a state_to_goal mapping" % (len(g),
                                                           len(s_next)))
    for i in range(len(g)):
        if abs(s_next[i] - g[i]) > tol:
            return -1.0
    return 0.0


def hindsr(episodes, actions=None, goals=None, strategy="future", k=4,
           tol=1e-6, reward_fn=None, state_to_goal=None, seed=0,
           history=None):
    r"""Build the HER replay buffer for a batch of episodes.

    Parameters
    ----------
    episodes : sequence
        One entry per episode, each a sequence of ``T + 1`` states
        :math:`s_0, \dots, s_T`.
    actions : sequence, optional
        One entry per episode, each a sequence of ``T`` actions. If
        omitted the action is the transition index, which is enough to
        keep the buffer well-formed when only the relabelling is under
        test.
    goals : sequence, optional
        The original goal of each episode. Defaults to the goal achieved
        by the last state, which makes every episode a success and is
        deliberately a poor default to rely on -- pass real goals.
    strategy : {"future", "final", "episode", "random"}
        :math:`\mathcal{S}`; see the module docstring.
    k : int
        Number of additional goals per transition. Ignored by
        ``"final"``, which always contributes exactly one.
    tol : float
        Tolerance of the goal test :math:`f_g`.
    reward_fn : callable, optional
        ``reward_fn(s, a, s_next, g)``. Defaults to the paper's sparse
        :math:`-[f_g(s') = 0]`.
    state_to_goal : callable, optional
        The paper's :math:`m`. Defaults to the identity.
    seed : int
        Seed for the goal sampling.
    history : sequence, optional
        States "encountered so far in the whole training procedure", for
        ``strategy="random"``. Defaults to every state in ``episodes``.

    Returns
    -------
    RichResult
        ``estimate`` is the buffer size. ``transitions`` is the buffer:
        a list of ``(state, action, reward, next_state, goal, relabelled)``
        tuples in Algorithm 1's order -- the original-goal copy first,
        then its relabelled copies. ``n_original`` and ``n_relabelled``
        split it; ``success_rate`` is the fraction of stored transitions
        whose reward is not :math:`-1`, which is the quantity the paper
        argues HER lifts off the floor.

    References
    ----------
    Andrychowicz et al. (2017) arXiv:1707.01495, Algorithm 1 and
    section 4.5.
    """
    if strategy not in _STRATEGIES:
        raise ValueError("hindsr: strategy must be one of %r, got %r"
                         % (_STRATEGIES, strategy))
    k = int(k)
    if k < 1:
        raise ValueError("hindsr: k must be >= 1")
    tol = float(tol)
    m = state_to_goal if state_to_goal is not None else (lambda s: list(s))
    if not callable(m):
        raise TypeError("hindsr: state_to_goal must be callable")
    rf = reward_fn
    if rf is None:
        def rf(s, a, s_next, g):
            return _sparse_reward(s, a, s_next, g, tol)
    elif not callable(rf):
        raise TypeError("hindsr: reward_fn must be callable")

    eps = [_as_states(e, "episode") for e in episodes]
    for i, e in enumerate(eps):
        if len(e) < 2:
            raise ValueError("hindsr: episode %d has %d states; need at "
                             "least s_0 and s_1" % (i, len(e)))
    n_ep = len(eps)
    if actions is None:
        acts = [list(range(len(e) - 1)) for e in eps]
    else:
        acts = [list(a) for a in actions]
        if len(acts) != n_ep:
            raise ValueError("hindsr: got %d action sequences for %d "
                             "episodes" % (len(acts), n_ep))
        for i in range(n_ep):
            if len(acts[i]) != len(eps[i]) - 1:
                raise ValueError("hindsr: episode %d has %d states but %d "
                                 "actions" % (i, len(eps[i]), len(acts[i])))
    if goals is None:
        gs = [m(e[-1]) for e in eps]
    else:
        gs = [[float(t) for t in np.atleast_1d(np.asarray(g, dtype=float))]
              for g in goals]
        if len(gs) != n_ep:
            raise ValueError("hindsr: got %d goals for %d episodes"
                             % (len(gs), n_ep))

    if history is None:
        pool = [s for e in eps for s in e]
    else:
        pool = _as_states(history, "history")

    rng = np.random.default_rng(seed)
    buf = []
    n_relabelled = 0
    for i in range(n_ep):
        e = eps[i]
        T = len(e) - 1
        for t in range(T):
            s, a, s1 = e[t], acts[i][t], e[t + 1]
            buf.append((s, a, float(rf(s, a, s1, gs[i])), s1, gs[i], False))
            for g2 in _sample_goals(strategy, e, t, T, k, m, pool, rng):
                buf.append((s, a, float(rf(s, a, s1, g2)), s1, g2, True))
                n_relabelled += 1

    rewards = [tr[2] for tr in buf]
    n_success = sum(1 for r in rewards if r > -1.0 + 1e-12)
    return RichResult(payload={
        "estimate": len(buf),
        "transitions": buf,
        "n_transitions": len(buf),
        "n_original": len(buf) - n_relabelled,
        "n_relabelled": n_relabelled,
        "rewards": rewards,
        "success_rate": float(n_success) / len(buf) if buf else 0.0,
        "strategy": strategy,
        "k": k,
        "n_episodes": n_ep,
        "method": "HER (Andrychowicz et al. 2017, Algorithm 1)",
    })


def _sample_goals(strategy, episode, t, T, k, m, pool, rng):
    r"""S(current episode) of Algorithm 1, for the transition at index t."""
    if strategy == "final":
        return [m(episode[T])]
    if strategy == "future":
        # "k random states which come from the same episode ... and were
        # observed after it" -- indices t+1 .. T.
        lo, hi = t + 1, T
        if hi < lo:
            return []
        return [m(episode[lo + int(rng.random() * (hi - lo + 1))])
                for _ in range(k)]
    if strategy == "episode":
        return [m(episode[int(rng.random() * (T + 1))]) for _ in range(k)]
    n = len(pool)
    return [m(pool[int(rng.random() * n)]) for _ in range(k)]


def cheatsheet():
    return ("hindsr: HER (Andrychowicz 2017 Alg. 1). Store each "
            "transition with the original goal, then again for each "
            "g' in S(episode) with the reward RECOMPUTED under g'. "
            "S in {final, future (k, best), episode, random}; "
            "r(s,a,g) = -[f_g(s') = 0].")


# compact alias per ledger/NAMING.md
her = hindsr
