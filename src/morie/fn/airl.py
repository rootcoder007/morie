r"""Adversarial Inverse Reinforcement Learning: recovering a *reward*,
not just a policy.

Fu, J., Luo, K., & Levine, S. (2018) "Learning Robust Rewards with
Adversarial Inverse Reinforcement Learning", *ICLR*, arXiv:1710.11248.

GAIL matches occupancy measures and throws the reward away. AIRL keeps
it, by constraining the discriminator to the form

.. math:: D_{\theta,\phi}(s, a, s') =
          \frac{\exp\{f_{\theta,\phi}(s,a,s')\}}
               {\exp\{f_{\theta,\phi}(s,a,s')\} + \pi(a \mid s)}

with :math:`f` split into a reward approximator and a shaping term
(eq. 4):

.. math:: f_{\theta,\phi}(s,a,s') = g_\theta(s,a)
          + \gamma h_\phi(s') - h_\phi(s).

Algorithm 1 then alternates: train :math:`D_{\theta,\phi}` by binary
logistic regression to separate expert data from policy samples, set

.. math:: r_{\theta,\phi}(s,a,s') = \log D_{\theta,\phi}
          - \log(1 - D_{\theta,\phi}),

and improve :math:`\pi` against that reward with any policy optimiser.
Note the orientation is the opposite of GAIL's: here :math:`D \to 1` on
**expert** data.

Why the shaping term earns its place. At the optimum
:math:`f^*(s,a,s') = A^*(s,a)`, the advantage -- which is entangled with
the dynamics, so it does not transfer to a new environment. Restricting
:math:`g_\theta` to a function of the state alone and giving the
:math:`\gamma h(s') - h(s)` shaping somewhere else to go, Theorem C.1
shows that under deterministic dynamics with a state-only ground-truth
reward,

.. math:: g^*(s) = r(s) + \text{const}, \qquad
          h^*(s) = V^*(s) + \text{const}.

So the shaping term absorbs the value function and what is left in
:math:`g` is the reward itself, up to a constant. That is the claim
that makes AIRL worth using over GAIL, and ``anchor_airl.py`` checks it
against an :math:`r` and a :math:`V^*` computed independently by soft
value iteration rather than read off the fit.

``state_only=True`` (the default) is the :math:`g_\theta(s)`
parameterisation Theorem C.1 requires. ``state_only=False`` gives the
:math:`g_\theta(s,a)` of eq. 4, which fits at least as well but recovers
the advantage rather than a transferable reward -- both are in the
paper, so both are here, with the transferable one as the default.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["airl", "soft_value_iteration"]


def _key(s):
    if isinstance(s, (int, str)):
        return s
    return tuple(float(v) for v in np.atleast_1d(np.asarray(s, dtype=float)))


def _log(x, floor=1e-300):
    return math.log(max(x, floor))


def airl(expert_states, expert_actions, expert_next, expert_log_policy,
         policy_states, policy_actions, policy_next, policy_log_policy,
         gamma=0.99, state_only=True, lr=0.1, epochs=500, l2=0.0):
    r"""Fit the AIRL discriminator and read off the recovered reward.

    Parameters
    ----------
    expert_states, expert_actions, expert_next : array-like
        Expert transitions :math:`(s, a, s')`.
    policy_states, policy_actions, policy_next : array-like
        Transitions sampled from the current policy.
    expert_log_policy, policy_log_policy : array-like
        :math:`\log \pi(a \mid s)` under the **current policy** for each
        transition in the corresponding set. AIRL's discriminator is
        explicitly a function of :math:`\pi`; this is not optional
        bookkeeping, it is what makes :math:`f` come out as the
        advantage rather than as a log density ratio.
    gamma : float
        The discount used in the shaping term of eq. 4.
    state_only : bool
        Parameterise :math:`g` on the state alone (Theorem C.1) or on
        :math:`(s, a)` (eq. 4 as written).
    lr, epochs, l2 : float, int, float
        Full-batch gradient ascent on the logistic log-likelihood, with
        optional ridge penalty.

    Returns
    -------
    RichResult
        ``estimate`` / ``reward`` is
        :math:`\log D - \log(1-D)` for each policy transition (line 6 of
        Algorithm 1). Also ``g`` and ``h`` as dicts over the observed
        states (or ``(s,a)`` pairs), ``f_policy`` / ``f_expert``,
        ``D_policy`` / ``D_expert``, ``accuracy`` and
        ``log_likelihood``.

    References
    ----------
    Fu, Luo & Levine (2018) arXiv:1710.11248, eq. 4, Algorithm 1,
    Theorem C.1.
    """
    def prep(S, A, S1, LP, name):
        S = [_key(s) for s in S]
        S1 = [_key(s) for s in S1]
        A = list(A)
        LP = [float(v) for v in np.atleast_1d(np.asarray(LP, dtype=float))]
        n = len(S)
        if not (len(A) == len(S1) == len(LP) == n) or n == 0:
            raise ValueError("airl: %s states, actions, next states and "
                             "log_policy must be non-empty and the same "
                             "length" % name)
        return list(zip(S, A, S1, LP))

    E = prep(expert_states, expert_actions, expert_next,
             expert_log_policy, "expert")
    P = prep(policy_states, policy_actions, policy_next,
             policy_log_policy, "policy")

    states = sorted(set([t[0] for t in E + P] + [t[2] for t in E + P]),
                    key=repr)
    if state_only:
        gkeys = list(states)
    else:
        gkeys = sorted(set((t[0], t[1]) for t in E + P), key=repr)
    gi = dict((k, i) for i, k in enumerate(gkeys))
    hi = dict((k, i) for i, k in enumerate(states))
    ng, nh = len(gkeys), len(states)
    g = [0.0] * ng
    h = [0.0] * nh
    gamma = float(gamma)
    lr = float(lr)
    l2 = float(l2)

    def gkey(t):
        return t[0] if state_only else (t[0], t[1])

    def f_of(t):
        return g[gi[gkey(t)]] + gamma * h[hi[t[2]]] - h[hi[t[0]]]

    def d_of(t):
        # D = exp(f) / (exp(f) + pi) = sigmoid(f - log pi)
        z = f_of(t) - t[3]
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    # The model is tabular, so identical transitions contribute
    # identical gradients. Collapse them to weighted unique rows: the
    # gradient is unchanged and the fit stops being quadratic in the
    # number of samples.
    def compress(rows):
        counts = {}
        for t in rows:
            counts[t] = counts.get(t, 0) + 1
        n = float(len(rows))
        return [(t, c / n) for t, c in counts.items()]

    Ec = compress(E)
    Pc = compress(P)

    for _ in range(max(1, int(epochs))):
        dg = [0.0] * ng
        dh = [0.0] * nh
        # Binary logistic regression, expert labelled 1, policy 0.
        # d/df log D       =  1 - D    (expert)
        # d/df log (1 - D) =    - D    (policy)
        for t, wgt in Ec:
            c = (1.0 - d_of(t)) * wgt
            dg[gi[gkey(t)]] += c
            dh[hi[t[2]]] += c * gamma
            dh[hi[t[0]]] -= c
        for t, wgt in Pc:
            c = -d_of(t) * wgt
            dg[gi[gkey(t)]] += c
            dh[hi[t[2]]] += c * gamma
            dh[hi[t[0]]] -= c
        for i in range(ng):
            g[i] += lr * (dg[i] - l2 * g[i])
        for i in range(nh):
            h[i] += lr * (dh[i] - l2 * h[i])

    de = [d_of(t) for t in E]
    dp = [d_of(t) for t in P]
    # line 6: r = log D - log(1 - D), which equals f - log pi exactly.
    reward = [_log(v) - _log(1.0 - v) for v in dp]
    ll = (sum(_log(v) for v in de) / len(de)
          + sum(_log(1.0 - v) for v in dp) / len(dp))
    acc = (sum(1.0 for v in de if v > 0.5)
           + sum(1.0 for v in dp if v <= 0.5)) / (len(de) + len(dp))

    return RichResult(payload={
        "estimate": reward,
        "reward": reward,
        "g": dict((k, g[gi[k]]) for k in gkeys),
        "h": dict((k, h[hi[k]]) for k in states),
        "f_policy": [f_of(t) for t in P],
        "f_expert": [f_of(t) for t in E],
        "D_policy": dp,
        "D_expert": de,
        "accuracy": float(acc),
        "log_likelihood": float(ll),
        "gamma": gamma,
        "state_only": bool(state_only),
        "method": "AIRL (Fu, Luo & Levine 2018, eq. 4 + Alg. 1)",
    })


def soft_value_iteration(states, actions, step, reward, gamma=0.9,
                         iters=2000, tol=1e-14):
    r"""MaxEnt (soft) value iteration on a deterministic tabular MDP.

    .. math:: Q(s,a) = r(s) + \gamma V(s'), \qquad
              V(s) = \log \sum_a \exp Q(s,a),

    with the soft-optimal policy :math:`\pi(a \mid s) =
    \exp(Q(s,a) - V(s))`. AIRL is derived in the maximum-entropy IRL
    setting, so this is the :math:`V^*` that Theorem C.1's
    :math:`h^* = V^* + \text{const}` refers to. Provided here so a
    caller -- and the anchors -- can compute the ground truth
    independently of anything AIRL fitted.
    """
    S = list(states)
    A = list(actions)
    V = dict((s, 0.0) for s in S)
    for _ in range(int(iters)):
        newV = {}
        for s in S:
            qs = [reward(s) + gamma * V[step(s, a)] for a in A]
            m = max(qs)
            newV[s] = m + math.log(sum(math.exp(q - m) for q in qs))
        delta = max(abs(newV[s] - V[s]) for s in S)
        V = newV
        if delta < tol:
            break
    pi = {}
    for s in S:
        qs = [reward(s) + gamma * V[step(s, a)] for a in A]
        for i, a in enumerate(A):
            pi[(s, a)] = math.exp(qs[i] - V[s])
    return V, pi


def cheatsheet():
    return ("airl: D = exp(f)/(exp(f)+pi) with f = g(s) + gamma h(s') "
            "- h(s) (Fu 2018 eq. 4); train D to separate EXPERT from "
            "policy, then r = log D - log(1-D) (Alg. 1 line 6). "
            "Thm C.1: deterministic dynamics + state-only reward give "
            "g* = r + const and h* = V* + const, so the reward "
            "transfers where GAIL's occupancy match does not.")
