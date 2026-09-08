r"""Generative Adversarial Imitation Learning: the discriminator and the
cost it hands to the policy.

Ho, J., & Ermon, S. (2016) "Generative Adversarial Imitation Learning",
*NeurIPS*, arXiv:1606.03476.

GAIL learns to imitate without ever recovering a reward function. The
learner's occupancy measure :math:`\rho_\pi` plays the role of the
generator's distribution and the expert's :math:`\rho_{\pi_E}` the role
of the true data, and the algorithm looks for a saddle point of

.. math:: \mathbb{E}_{\pi}[\log D(s,a)]
          + \mathbb{E}_{\pi_E}[\log(1 - D(s,a))]
          - \lambda H(\pi) \tag{16}

with :math:`D : S \times A \to (0,1)`. Algorithm 1 alternates an
ascent step on :math:`w` -- the gradient

.. math:: \hat{\mathbb{E}}_{\tau_i}[\nabla_w \log D_w(s,a)]
          + \hat{\mathbb{E}}_{\tau_E}[\nabla_w \log(1 - D_w(s,a))]
          \tag{17}

-- with a TRPO step on :math:`\theta` that *decreases* eq. 16 using the
cost function :math:`c(s,a) = \log D_w(s,a)`, so the policy is pushed
toward the regions the discriminator classifies as expert-like.

Note the orientation, because it is the opposite of the usual GAN
convention and getting it backwards silently inverts the imitation:
here :math:`D \to 1` on **learner** data and :math:`D \to 0` on
**expert** data, and the policy minimises :math:`\log D`.

What this module does is the discriminator half -- fit :math:`D_w`,
report eq. 16, and emit the per-pair cost :math:`\log D_w(s,a)` and its
:math:`Q` aggregation of eq. 18 for the policy step. The policy step
itself is "any policy optimisation method" (TRPO in the paper), which
belongs to the optimiser you are already using, not here. What is here
is the part that is GAIL.

The discriminator is a logistic model on features of :math:`(s, a)`:
a one-hot indicator per pair for discrete spaces (the tabular case,
where eq. 16's saddle point is exactly characterisable and the anchors
check it), or the raw concatenated vector, or your own feature map.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gail", "occupancy_measure"]


def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _pairs(states, actions, name):
    S = [tuple(np.atleast_1d(np.asarray(s, dtype=float)))
         if not isinstance(s, (int, str)) else (s,) for s in states]
    A = list(actions)
    if len(S) != len(A):
        raise ValueError("gail: %s states and actions must have the same "
                         "length" % name)
    if not S:
        raise ValueError("gail: %s must be non-empty" % name)
    return [(S[i], A[i]) for i in range(len(S))]


def occupancy_measure(states, actions):
    r"""The empirical occupancy measure :math:`\hat\rho(s,a)`.

    GAIL's whole framing is that imitation is occupancy-measure
    matching, so this is worth being able to look at directly.
    """
    pr = _pairs(states, actions, "occupancy")
    counts = {}
    for p in pr:
        counts[p] = counts.get(p, 0) + 1
    n = float(len(pr))
    return dict((k, v / n) for k, v in counts.items())


def gail(expert_states, expert_actions, policy_states, policy_actions,
         features=None, lr=0.1, epochs=200, l2=0.0, lam=0.0,
         policy_entropy=0.0, clip=1e-9):
    r"""Fit the GAIL discriminator and return the cost for the policy step.

    Parameters
    ----------
    expert_states, expert_actions : array-like
        The expert demonstrations :math:`\tau_E`.
    policy_states, policy_actions : array-like
        Trajectories :math:`\tau_i` sampled from the current policy.
    features : callable, optional
        ``features(s, a) -> list``. Defaults to a one-hot indicator over
        the observed :math:`(s,a)` pairs, which makes :math:`D_w`
        fully non-parametric and is the setting in which eq. 16's
        optimum is exactly :math:`\rho_\pi / (\rho_\pi + \rho_{\pi_E})`.
    lr, epochs, l2 : float, int, float
        Full-batch gradient ascent on eq. 17, with optional ridge
        penalty.
    lam : float
        :math:`\lambda`, the causal-entropy weight of eq. 16. Only
        enters the reported objective; the entropy of your policy is
        yours to supply.
    policy_entropy : float
        :math:`H(\pi)`, if you have it, for the eq. 16 value.
    clip : float
        Floor on :math:`D` and :math:`1-D` inside logarithms.

    Returns
    -------
    RichResult
        ``estimate`` / ``cost`` is :math:`\log D_w(s,a)` for each
        *policy* sample -- the per-step cost the TRPO step of eq. 18
        consumes. Also ``D_policy`` and ``D_expert``, ``objective``
        (eq. 16), ``accuracy`` (how well the discriminator separates the
        two sets), ``weights``, and ``occupancy_policy`` /
        ``occupancy_expert``.

    References
    ----------
    Ho & Ermon (2016) arXiv:1606.03476, eqs. 16-18 and Algorithm 1.
    """
    E = _pairs(expert_states, expert_actions, "expert")
    P = _pairs(policy_states, policy_actions, "policy")
    if features is None:
        keys = sorted(set(E) | set(P), key=repr)
        index = dict((k, i) for i, k in enumerate(keys))
        nf = len(keys) + 1

        def feat(p):
            v = [0.0] * nf
            v[index[p]] = 1.0
            v[-1] = 1.0                       # bias
            return v
    else:
        if not callable(features):
            raise TypeError("gail: features must be callable")

        def feat(p):
            return [float(x) for x in features(p[0], p[1])] + [1.0]
        nf = len(feat(P[0]))

    XP = [feat(p) for p in P]
    XE = [feat(p) for p in E]
    w = [0.0] * nf
    lr = float(lr)
    l2 = float(l2)

    for _ in range(max(1, int(epochs))):
        g = [0.0] * nf
        # eq. 17: ascend E_pi[log D] + E_piE[log(1 - D)].
        # d/dw log D      = (1 - D) x       (policy samples)
        # d/dw log(1 - D) =    - D  x       (expert samples)
        for x in XP:
            d = _sigmoid(sum(w[j] * x[j] for j in range(nf)))
            c = (1.0 - d) / len(XP)
            for j in range(nf):
                g[j] += c * x[j]
        for x in XE:
            d = _sigmoid(sum(w[j] * x[j] for j in range(nf)))
            c = -d / len(XE)
            for j in range(nf):
                g[j] += c * x[j]
        for j in range(nf):
            w[j] += lr * (g[j] - l2 * w[j])

    def D(x):
        return min(1.0 - clip,
                   max(clip, _sigmoid(sum(w[j] * x[j] for j in range(nf)))))

    dp = [D(x) for x in XP]
    de = [D(x) for x in XE]
    obj = (sum(math.log(v) for v in dp) / len(dp)
           + sum(math.log(1.0 - v) for v in de) / len(de)
           - lam * float(policy_entropy))
    cost = [math.log(v) for v in dp]
    acc = (sum(1.0 for v in dp if v > 0.5)
           + sum(1.0 for v in de if v <= 0.5)) / (len(dp) + len(de))

    # eq. 18's Q(s,a) = E_tau[ log D | s_0 = s, a_0 = a ], estimated by
    # averaging the cost over every occurrence of the pair.
    q = {}
    for i, p in enumerate(P):
        q.setdefault(p, []).append(cost[i])
    q = dict((k, sum(v) / len(v)) for k, v in q.items())

    return RichResult(payload={
        "estimate": cost,
        "cost": cost,
        "Q": q,
        "D_policy": dp,
        "D_expert": de,
        "objective": float(obj),
        "accuracy": float(acc),
        "weights": w,
        "occupancy_policy": occupancy_measure(policy_states, policy_actions),
        "occupancy_expert": occupancy_measure(expert_states, expert_actions),
        "n_policy": len(P),
        "n_expert": len(E),
        "method": "GAIL discriminator (Ho & Ermon 2016, eqs. 16-18)",
    })


def cheatsheet():
    return ("gail: saddle point of E_pi[log D] + E_piE[log(1-D)] - "
            "lambda H(pi) (Ho & Ermon 2016 eq. 16). D -> 1 on LEARNER "
            "data, -> 0 on expert; the policy minimises cost "
            "c(s,a) = log D. At rho_pi = rho_piE the optimum is "
            "D == 1/2 and the objective is -2 log 2. Imitation as "
            "occupancy-measure matching, no reward recovered.")
