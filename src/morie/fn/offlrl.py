r"""Conservative Q-Learning for offline RL.

Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020) "Conservative
Q-Learning for Offline Reinforcement Learning", *NeurIPS*,
arXiv:2006.04779.

Offline RL fails in a specific way: the Bellman backup takes a
:math:`\max` (or an expectation under :math:`\pi`) over actions that
the dataset never contains, so the Q-function is queried out of
distribution, over-estimates there, and the policy chases the
over-estimate. CQL fixes it by learning a Q-function that is a
**lower bound** on the true one, by adding a term that pushes down
Q-values under a distribution :math:`\mu` while pushing *up* the
values actually seen in the data (eq. 2):

.. math:: \hat Q^{k+1} \leftarrow \arg\min_Q\ \alpha\,\Big(
          \mathbb{E}_{s \sim D, a \sim \mu}[Q(s,a)]
          - \mathbb{E}_{s \sim D, a \sim \hat\pi_\beta}[Q(s,a)]\Big)
          + \tfrac12 \mathbb{E}_{s,a,s' \sim D}
          \Big[\big(Q(s,a) - \hat{\mathcal{B}}^\pi \hat Q^k(s,a)\big)^2
          \Big].

The second term is ordinary fitted Q-iteration. The first is the whole
idea, and the asymmetry matters: only the :math:`\mu` term goes down,
the behaviour-policy term goes *up*, so in-distribution actions are not
penalised and the bound stays tight where the data actually is.

Making :math:`\mu` itself adversarial gives the CQL family (eq. 3),
:math:`\min_Q \max_\mu`, regularised by :math:`\mathcal{R}(\mu)`. With
:math:`\mathcal{R}(\mu) = -D_{\mathrm{KL}}(\mu, \rho)` the inner
maximisation has the closed form :math:`\mu \propto \rho \exp Q`, and
two choices of :math:`\rho` are the variants the paper evaluates:

``variant="H"``
    :math:`\rho = \mathrm{Unif}(a)`, so the first term becomes a
    soft-maximum -- eq. 4, the paper's CQL(H):

    .. math:: \min_Q\ \alpha\,\mathbb{E}_{s \sim D}\Big[
              \log \sum_a \exp Q(s,a)
              - \mathbb{E}_{a \sim \hat\pi_\beta}[Q(s,a)]\Big]
              + \tfrac12 \mathbb{E}\big[(Q -
              \hat{\mathcal{B}}^{\pi_k}\hat Q^k)^2\big].

``variant="rho"``
    :math:`\rho = \hat\pi^{k-1}`, the previous policy: an exponentially
    weighted average of Q-values over that policy's actions instead of
    a full soft-max. The paper reports this is more stable in
    high-dimensional action spaces, where estimating
    :math:`\log \sum_a \exp` by sampling is high-variance.

``variant="mu"``
    Eq. 2 directly, with :math:`\mu` supplied by the caller. Theorem
    3.2's tighter bound is stated for :math:`\mu = \pi`.

Note the direction of the guarantee. Theorem 3.1 (eq. 1) gives a
*pointwise* lower bound; Theorem 3.2 (eq. 2) gives the tighter but
weaker statement that the **expected** value lower-bounds,
:math:`\mathbb{E}_{\pi}[\hat Q^\pi(s,a)] \le V^\pi(s)` when
:math:`\mu = \pi`, and explicitly does *not* promise a pointwise bound
-- Q-values for actions likely under :math:`\hat\pi_\beta` may be
over-estimated. That distinction is checked in the anchors rather than
glossed, because getting it backwards would mean claiming a guarantee
the method does not make.

Backups: ``backup="max"`` uses :math:`\mathcal{B}^*` (Q-learning) and
``backup="pi"`` uses :math:`\mathcal{B}^\pi` (policy evaluation); the
paper notes CQL can be instantiated either way.

This is the tabular case, where each Q-value is represented exactly and
the objective can be minimised directly -- which is the setting in
which the theorems are stated, before the function-approximation
extensions of Appendix D.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["offlrl", "offline_rl_cql", "conservative_q_learning"]

_VARIANTS = ("H", "rho", "mu")
_BACKUPS = ("max", "pi")


def _logsumexp(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def _softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def offlrl(dataset, states=None, actions=None, alpha=1.0, gamma=0.99,
           variant="H", backup="max", policy=None, mu=None, lr=0.5,
           iters=2000, tol=1e-12):
    r"""Fit a conservative Q-function to a fixed offline dataset.

    Parameters
    ----------
    dataset : sequence
        Transitions ``(s, a, r, s_next)``, or ``(s, a, r, s_next,
        done)``. This is :math:`D`; nothing else is observed, which is
        the point of offline RL.
    states, actions : sequence, optional
        The full state and action spaces. Inferred from ``dataset`` if
        omitted -- but note that inferring them means the algorithm
        never sees the out-of-distribution actions it is supposed to be
        conservative about, so pass the real action set.
    alpha : float
        :math:`\alpha`, the conservatism weight. ``alpha=0`` reduces to
        ordinary fitted Q-iteration, which is the failure mode CQL
        exists to fix.
    gamma : float
        Discount.
    variant : {"H", "rho", "mu"}
        Eq. 4 (soft-max), eq. 4 with :math:`\rho = \pi^{k-1}`, or eq. 2
        with an explicit :math:`\mu`.
    backup : {"max", "pi"}
        :math:`\mathcal{B}^*` or :math:`\mathcal{B}^\pi`.
    policy : callable or dict, optional
        :math:`\pi(a \mid s)` for ``backup="pi"`` and for
        ``variant="rho"``; ``policy[(s, a)] -> probability``.
    mu : callable or dict, optional
        :math:`\mu(a \mid s)` for ``variant="mu"``.
    lr : float
        Step size for the minimisation of the objective.
    iters : int
        Iterations.
    tol : float
        Convergence tolerance on the largest Q update.

    Returns
    -------
    RichResult
        ``estimate`` / ``q`` is ``{(s, a): value}``; ``value`` the
        greedy state values; ``greedy`` the greedy action per state;
        ``behavior`` the empirical :math:`\hat\pi_\beta` read off the
        dataset; ``counts`` the state-action counts; ``penalty`` the
        value of the CQL term at the solution; ``bellman_error`` the
        fitted-Q term; and ``objective`` their weighted sum.

    References
    ----------
    Kumar, Zhou, Tucker & Levine (2020) arXiv:2006.04779, eqs. 1-4,
    Theorems 3.1-3.2.
    """
    if variant not in _VARIANTS:
        raise ValueError("offlrl: variant must be one of %r, got %r"
                         % (_VARIANTS, variant))
    if backup not in _BACKUPS:
        raise ValueError("offlrl: backup must be 'max' or 'pi', got %r"
                         % (backup,))
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("offlrl: alpha must be >= 0, got %r" % (alpha,))

    D = []
    for t in dataset:
        if len(t) == 4:
            s, a, r, s1 = t
            done = False
        elif len(t) == 5:
            s, a, r, s1, done = t
        else:
            raise ValueError("offlrl: each transition must be (s, a, r, "
                             "s_next) or (s, a, r, s_next, done)")
        D.append((s, a, float(r), s1, bool(done)))
    if not D:
        raise ValueError("offlrl: dataset must be non-empty")

    S = list(states) if states is not None else sorted(
        set([t[0] for t in D] + [t[3] for t in D]), key=repr)
    A = list(actions) if actions is not None else sorted(
        set(t[1] for t in D), key=repr)
    if not S or not A:
        raise ValueError("offlrl: states and actions must be non-empty")
    ai = dict((a, i) for i, a in enumerate(A))

    # Empirical behaviour policy and counts, straight off the data.
    n_sa = {}
    n_s = {}
    for s, a, _r, _s1, _d in D:
        n_sa[(s, a)] = n_sa.get((s, a), 0) + 1
        n_s[s] = n_s.get(s, 0) + 1
    behavior = {}
    for s in S:
        if s not in n_s:
            continue
        for a in A:
            behavior[(s, a)] = n_sa.get((s, a), 0) / float(n_s[s])

    pol = _as_dist(policy, S, A, "policy")
    muu = _as_dist(mu, S, A, "mu")
    if variant == "mu" and muu is None:
        raise ValueError("offlrl: variant='mu' needs mu(a|s)")
    if backup == "pi" and pol is None:
        raise ValueError("offlrl: backup='pi' needs policy(a|s)")
    if variant == "rho" and pol is None:
        raise ValueError("offlrl: variant='rho' needs policy(a|s) to play "
                         "the role of pi^{k-1}")

    Q = dict(((s, a), 0.0) for s in S for a in A)
    data_states = [s for s in S if s in n_s]

    for _ in range(int(iters)):
        # Bellman targets from the dataset only.
        target = {}
        cnt = {}
        for s, a, r, s1, done in D:
            if done:
                t = r
            elif backup == "max":
                t = r + gamma * max(Q[(s1, b)] for b in A)
            else:
                t = r + gamma * sum(pol[(s1, b)] * Q[(s1, b)] for b in A)
            target[(s, a)] = target.get((s, a), 0.0) + t
            cnt[(s, a)] = cnt.get((s, a), 0) + 1
        for k in target:
            target[k] /= cnt[k]

        # Gradient of the CQL objective wrt each Q(s,a).
        grad = dict(((s, a), 0.0) for s in S for a in A)
        for s in data_states:
            w = n_s[s] / float(len(D))
            qs = [Q[(s, b)] for b in A]
            if variant == "H":
                push = _softmax(qs)                   # d/dQ logsumexp
            elif variant == "rho":
                # mu ∝ rho * exp(Q) with rho = pi^{k-1}
                m = max(qs)
                e = [pol[(s, b)] * math.exp(qs[i] - m)
                     for i, b in enumerate(A)]
                z = sum(e)
                push = ([x / z for x in e] if z > 0
                        else [1.0 / len(A)] * len(A))
            else:
                push = [muu[(s, b)] for b in A]
            for i, b in enumerate(A):
                grad[(s, b)] += alpha * w * (push[i] - behavior[(s, b)])
        for k in target:
            grad[k] += (cnt[k] / float(len(D))) * (Q[k] - target[k])

        delta = 0.0
        for k in Q:
            step = lr * grad[k]
            Q[k] -= step
            if abs(step) > delta:
                delta = abs(step)
        if delta < tol:
            break

    value = dict((s, max(Q[(s, a)] for a in A)) for s in S)
    greedy = dict((s, max(A, key=lambda a: Q[(s, a)])) for s in S)

    pen = 0.0
    for s in data_states:
        w = n_s[s] / float(len(D))
        qs = [Q[(s, b)] for b in A]
        if variant == "H":
            first = _logsumexp(qs)
        elif variant == "rho":
            m = max(qs)
            first = m + math.log(sum(pol[(s, b)] * math.exp(qs[i] - m)
                                     for i, b in enumerate(A)))
        else:
            first = sum(muu[(s, b)] * Q[(s, b)] for b in A)
        pen += w * (first - sum(behavior[(s, b)] * Q[(s, b)] for b in A))

    berr = 0.0
    for s, a, r, s1, done in D:
        if done:
            t = r
        elif backup == "max":
            t = r + gamma * max(Q[(s1, b)] for b in A)
        else:
            t = r + gamma * sum(pol[(s1, b)] * Q[(s1, b)] for b in A)
        berr += 0.5 * (Q[(s, a)] - t) ** 2
    berr /= len(D)

    return RichResult(payload={
        "estimate": Q,
        "q": Q,
        "value": value,
        "greedy": greedy,
        "behavior": behavior,
        "counts": n_sa,
        "penalty": float(pen),
        "bellman_error": float(berr),
        "objective": float(alpha * pen + berr),
        "alpha": alpha,
        "variant": variant,
        "backup": backup,
        "n_transitions": len(D),
        "method": "CQL (Kumar et al. 2020, eq. %s)"
                  % ("4" if variant in ("H", "rho") else "2"),
    })


def _as_dist(d, S, A, name):
    if d is None:
        return None
    if callable(d):
        out = dict(((s, a), float(d(s, a))) for s in S for a in A)
    else:
        out = dict(((s, a), float(d[(s, a)])) for s in S for a in A)
    for s in S:
        tot = sum(out[(s, a)] for a in A)
        if abs(tot - 1.0) > 1e-6:
            raise ValueError("offlrl: %s(.|%r) sums to %g, not 1"
                             % (name, s, tot))
    return out


def cheatsheet():
    return ("offlrl: CQL (Kumar 2020). Fitted Q plus alpha*(push DOWN "
            "E_mu[Q] - push UP E_pi_beta[Q]) so the Q-function LOWER "
            "BOUNDS the truth and OOD actions stop being "
            "over-estimated. variant='H' is eq. 4's logsumexp "
            "(rho=Unif); 'rho' uses pi^{k-1}; 'mu' is eq. 2 directly. "
            "Thm 3.2 bounds the EXPECTED value under pi, not "
            "pointwise. alpha=0 is plain fitted Q.")


# compact aliases per ledger/NAMING.md
offline_rl_cql = offlrl
offlinerlcql = offlrl
conservative_q_learning = offlrl
