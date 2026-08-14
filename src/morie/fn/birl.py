# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian inverse reinforcement learning.

Ramachandran, D., & Amir, E. (2007) "Bayesian Inverse Reinforcement
Learning", *IJCAI-07*, 2586-2591.

Inverse reinforcement learning asks what an expert was *trying* to do:
given their behaviour and the dynamics, recover the reward. The problem
is badly underdetermined -- many rewards explain the same behaviour --
so the paper does not pick one. It puts a prior on rewards, writes down
how likely the observed behaviour is under each, and works with the
whole posterior.

**The expert model.** The expert is assumed to prefer high-value actions
but not to be perfect, which gives a Boltzmann likelihood with the
optimal action-value as the energy:

.. math::

   \Pr(O_X \mid R) = \frac{1}{Z}\, e^{\alpha_X E(O_X, R)},
   \qquad
   E(O_X, R) = \sum_i Q^{*}(s_i, a_i, R),

with :math:`1/\alpha_X` the temperature. The normaliser factorises --
:math:`Z = \sum_{O} e^{\alpha E(O, R)} = \prod_i \sum_a
e^{\alpha Q^{*}(s_i, a, R)}` -- so the likelihood is a product of
per-state softmaxes over actions, and that is how it is computed here.
Large :math:`\alpha` means a near-optimal expert, small :math:`\alpha` a
near-random one.

The posterior is then Bayes,
:math:`\Pr(R \mid O_X) \propto \Pr(O_X \mid R) P_R(R)`.

**Sampling it: PolicyWalk** (Figure 3), a random walk on the grid
:math:`\mathbb{R}^{|S|}/\delta`. Its point is the branch in step 3(c).
After proposing a neighbour :math:`\tilde{R}`, it checks whether the
*current* policy is still optimal under it:

* if some action beats :math:`\pi(s)` somewhere, the policy is recomputed
  by policy iteration warm-started from :math:`\pi`, and the pair
  :math:`(\tilde{R}, \tilde{\pi})` is accepted with the usual ratio;
* otherwise :math:`\pi` is already optimal for :math:`\tilde{R}` and no
  policy iteration is run at all.

That test is what makes the walk affordable -- most proposals do not
change the optimal policy, and the expensive step is skipped. The saving
is real and measured in the anchor, not assumed.

**What to report.** Theorem 3: the policy that minimises the expected
policy loss is the optimal policy for :math:`E_P[R]`, the posterior
*mean* reward -- not the posterior mode and not the reward of any single
sample. So the posterior mean is what this returns, and the policy that
goes with it.

Priors offered: ``"uniform"`` (the paper's own experiments),
``"gaussian"``, ``"laplacian"`` (which the paper singles out as mixing
rapidly) and ``"ising"``, the :math:`P_R(R) = \exp(-J\sum R(s)R(s') -
H\sum R(s))` form it gives for rewards with spatial structure.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "birl",
    "bayesian_irl",
    "policy_iteration",
    "q_values",
    "policy_values",
    "log_likelihood",
    "log_prior",
    "policy_walk",
    "PRIORS",
]

PRIORS = ("uniform", "gaussian", "laplacian", "ising")


def _mdp(T, gamma):
    if not T:
        raise ValueError("birl: the transition model is empty")
    nS = len(T)
    nA = len(T[0])
    if nA == 0:
        raise ValueError("birl: there are no actions")
    for s in range(nS):
        if len(T[s]) != nA:
            raise ValueError("birl: every state needs the same actions")
        for a in range(nA):
            row = T[s][a]
            if len(row) != nS:
                raise ValueError("birl: a transition row has the wrong "
                                 "length")
            tot = sum(row)
            if abs(tot - 1.0) > 1e-8 or min(row) < 0:
                raise ValueError("birl: transition rows must be "
                                 "probability distributions")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("birl: gamma must be in [0, 1)")
    return nS, nA


def _solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError("birl: the value system is singular")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def policy_values(T, R, gamma, policy):
    r"""Equation 4: :math:`V^{\pi}(R) = (I - \gamma T^{\pi})^{-1} R`.

    Solved directly rather than iterated, which also makes the linearity
    of :math:`V^{\pi}` in :math:`R` exact rather than approximate.
    """
    nS, _nA = _mdp(T, gamma)
    if len(policy) != nS or len(R) != nS:
        raise ValueError("birl: policy and reward need one entry per state")
    A = [[(1.0 if i == j else 0.0) - gamma * T[i][policy[i]][j]
          for j in range(nS)] for i in range(nS)]
    return _solve(A, [float(v) for v in R])


def q_values(T, R, gamma, V):
    r""":math:`Q(s, a) = R(s) + \gamma \sum_{s'} T(s, a, s') V(s')`."""
    nS, nA = _mdp(T, gamma)
    return [[R[s] + gamma * sum(T[s][a][j] * V[j] for j in range(nS))
             for a in range(nA)] for s in range(nS)]


def policy_iteration(T, R, gamma, policy=None, max_iter=200):
    """Policy iteration, optionally warm-started as PolicyWalk does."""
    nS, nA = _mdp(T, gamma)
    if len(R) != nS:
        raise ValueError("birl: one reward per state is required")
    pi = [0] * nS if policy is None else list(policy)
    if len(pi) != nS:
        raise ValueError("birl: the starting policy has the wrong length")
    sweeps = 0
    for _ in range(int(max_iter)):
        V = policy_values(T, R, gamma, pi)
        Q = q_values(T, R, gamma, V)
        new = [max(range(nA), key=lambda a: Q[s][a]) for s in range(nS)]
        sweeps += 1
        if new == pi:
            return {"policy": pi, "V": V, "Q": Q, "sweeps": sweeps}
        pi = new
    V = policy_values(T, R, gamma, pi)
    return {"policy": pi, "V": V, "Q": q_values(T, R, gamma, V),
            "sweeps": sweeps}


def log_likelihood(Q, observations, alpha=1.0):
    r"""The Boltzmann expert, as a sum of per-state log softmaxes.

    :math:`Z` factorises over the observed states, so
    :math:`\log \Pr(O_X \mid R) = \sum_i \bigl[\alpha Q(s_i, a_i)
    - \log\sum_a e^{\alpha Q(s_i, a)}\bigr]`.
    """
    if alpha <= 0:
        raise ValueError("birl: alpha must be positive")
    if not observations:
        raise ValueError("birl: no observations")
    total = 0.0
    for s, a in observations:
        if not 0 <= s < len(Q) or not 0 <= a < len(Q[s]):
            raise ValueError("birl: an observation is out of range")
        row = [alpha * v for v in Q[s]]
        m = max(row)
        total += row[a] - (m + math.log(sum(math.exp(v - m)
                                            for v in row)))
    return total


def log_prior(R, prior="uniform", scale=1.0, r_max=None, J=0.1, H=0.0,
              neighbours=None):
    """Log prior over reward vectors, up to a constant."""
    if prior not in PRIORS:
        raise ValueError("birl: prior must be one of %s" % (PRIORS,))
    if scale <= 0:
        raise ValueError("birl: scale must be positive")
    if prior == "uniform":
        if r_max is not None and any(abs(v) > r_max + 1e-12 for v in R):
            return float("-inf")
        return 0.0
    if prior == "gaussian":
        return -sum(v * v for v in R) / (2.0 * scale * scale)
    if prior == "laplacian":
        return -sum(abs(v) for v in R) / scale
    pairs = neighbours or [(i, i + 1) for i in range(len(R) - 1)]
    return -(J * sum(R[i] * R[j] for i, j in pairs) + H * sum(R))


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def policy_walk(T, observations, gamma, n_iter=1000, delta=0.25,
                alpha=1.0, prior="uniform", scale=1.0, r_max=1.0,
                J=0.1, H=0.0, burn=None, seed=0, R0=None):
    """Figure 3: PolicyWalk.

    Returns the samples plus a count of how often step 3(c) had to run
    policy iteration, which is the cost the branch exists to avoid.
    """
    nS, nA = _mdp(T, gamma)
    if delta <= 0:
        raise ValueError("birl: delta must be positive")
    if n_iter < 1:
        raise ValueError("birl: n_iter must be positive")
    burn = n_iter // 2 if burn is None else int(burn)
    if not 0 <= burn < n_iter:
        raise ValueError("birl: burn must be less than n_iter")
    rnd = _rng(seed + 3)

    def grid(v):
        return round(v / delta) * delta

    # 1. pick a random reward vector on the grid
    R = [grid((2.0 * rnd() - 1.0) * r_max) for _ in range(nS)] \
        if R0 is None else [grid(float(v)) for v in R0]
    # 2. pi := PolicyIteration(M, R)
    got = policy_iteration(T, R, gamma)
    pi, Q = got["policy"], got["Q"]

    def score(Qm, Rv):
        lp = log_prior(Rv, prior, scale, r_max, J, H, None)
        if lp == float("-inf"):
            return lp
        return log_likelihood(Qm, observations, alpha) + lp

    cur = score(Q, R)
    samples, accepted, repolicy = [], 0, 0
    for it in range(int(n_iter)):
        # 3(a) a uniformly chosen neighbour on the grid
        s = int(rnd() * nS)
        step = delta if rnd() < 0.5 else -delta
        cand = list(R)
        cand[s] = grid(cand[s] + step)
        if r_max is not None and abs(cand[s]) > r_max + 1e-12:
            samples.append(list(R))
            continue
        # 3(b) Q^pi under the proposed reward, with the CURRENT policy
        Vp = policy_values(T, cand, gamma, pi)
        Qp = q_values(T, cand, gamma, Vp)
        # 3(c) is pi still optimal for cand?
        changed = any(Qp[st][pi[st]] < max(Qp[st]) - 1e-12
                      for st in range(nS))
        if changed:
            repolicy += 1
            got2 = policy_iteration(T, cand, gamma, pi)
            newpi, newQ = got2["policy"], got2["Q"]
        else:
            newpi, newQ = pi, Qp
        prop = score(newQ, cand)
        if prop > cur or math.log(max(rnd(), 1e-300)) < prop - cur:
            R, pi, Q, cur = cand, newpi, newQ, prop
            accepted += 1
        if it >= burn:
            samples.append(list(R))
    if not samples:
        samples = [list(R)]
    return {"samples": samples, "acceptance": accepted / float(n_iter),
            "policy_iterations": repolicy, "n_proposals": int(n_iter),
            "final_policy": pi}


def birl(T, observations, gamma=0.9, n_iter=1000, delta=0.25, alpha=1.0,
         prior="uniform", scale=1.0, r_max=1.0, J=0.1, H=0.0, burn=None,
         seed=0, R0=None):
    """Recover the reward and the policy it implies (Theorem 3)."""
    nS, nA = _mdp(T, gamma)
    obs = [(int(s), int(a)) for s, a in observations]
    walk = policy_walk(T, obs, gamma, n_iter, delta, alpha, prior, scale,
                       r_max, J, H, burn, seed, R0)
    S = walk["samples"]
    n = float(len(S))
    mean = [sum(r[i] for r in S) / n for i in range(nS)]
    var = [sum((r[i] - mean[i]) ** 2 for r in S) / max(n - 1.0, 1.0)
           for i in range(nS)]
    # Theorem 3: the policy to report is the optimal one for E_P[R]
    got = policy_iteration(T, mean, gamma)
    return RichResult(payload={
        "estimate": mean,
        "reward_mean": mean,
        "reward_sd": [math.sqrt(v) for v in var],
        "policy": got["policy"],
        "V": got["V"],
        "Q": got["Q"],
        "samples": S,
        "acceptance": walk["acceptance"],
        "policy_iterations": walk["policy_iterations"],
        "n_proposals": walk["n_proposals"],
        "n_samples": len(S),
        "prior": prior,
        "alpha": float(alpha),
        "delta": float(delta),
        "method": ("Bayesian IRL (Ramachandran & Amir 2007): Boltzmann "
                   "expert likelihood, PolicyWalk over the reward grid, "
                   "posterior mean reward per Theorem 3"),
        "note": ("Theorem 3 says the reported policy is the optimal one "
                 "for the posterior MEAN reward, not the mode and not "
                 "any single sample; policy_iterations counts how often "
                 "step 3(c) actually had to recompute the policy"),
    })


bayesian_irl = birl


def cheatsheet():
    return ("birl: Bayesian IRL (Ramachandran & Amir 2007). The expert "
            "is Boltzmann in the optimal action-value, Pr(O|R) = "
            "exp(alpha sum_i Q*(s_i,a_i,R))/Z, whose Z factorises into "
            "per-state softmaxes; the posterior is that times a prior "
            "(uniform, gaussian, laplacian or ising). PolicyWalk walks "
            "the reward grid and only recomputes the policy when the "
            "proposal makes some action beat pi(s). By Theorem 3 the "
            "answer to report is the optimal policy for the posterior "
            "MEAN reward.")

# public names resolved by fn/_lazy_map.json
bayesianirl = birl
