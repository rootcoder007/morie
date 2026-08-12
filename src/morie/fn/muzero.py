r"""MuZero: MCTS over a learned latent model.

Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L.,
Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T.,
Lillicrap, T., & Silver, D. (2020) "Mastering Atari, Go, Chess and
Shogi by Planning with a Learned Model", *Nature* 588, arXiv:1911.08265.

MuZero plans with a model it learned itself, and that model never
predicts observations -- only the three quantities a search needs:

* the **representation** :math:`s^0 = h_\theta(o_1, \dots, o_t)`,
  embedding the observation history into a latent state;
* the **dynamics** :math:`r^k, s^k = g_\theta(s^{k-1}, a^k)`, giving
  the immediate reward and the next latent state;
* the **prediction** :math:`p^k, v^k = f_\theta(s^k)`, the policy prior
  and value.

The latent states carry no semantics beyond being useful for
predicting reward, value and policy -- which is exactly why the method
works without a simulator.

Search is AlphaZero's, generalised to intermediate rewards, a discount,
and unbounded values. **Selection** uses pUCT (eq. 2):

.. math:: a^k = \arg\max_a\ \Big[ Q(s,a) + P(s,a)\,
          \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}
          \Big(c_1 + \log\frac{\sum_b N(s,b) + c_2 + 1}{c_2}\Big)\Big],

with the paper's :math:`c_1 = 1.25`, :math:`c_2 = 19652`. The
:math:`c_2` term only matters once visit counts approach it, which is
why :math:`c_1` alone reproduces AlphaZero at small budgets.

**Expansion** calls :math:`g_\theta` and :math:`f_\theta` exactly once
per simulation and initialises each new edge to
:math:`N = 0,\ Q = 0,\ P = p^l`.

**Backup** (eqs. 3-4) forms the :math:`(l-k)`-step bootstrapped return

.. math:: G^k = \sum_{\tau=0}^{l-1-k} \gamma^\tau r^{k+1+\tau}
          + \gamma^{l-k} v^l,

and folds it into a running mean:

.. math:: Q(s^{k-1}, a^k) := \frac{N \cdot Q + G^k}{N + 1},
          \qquad N := N + 1.

Because the value is unbounded in general environments, MuZero
normalises :math:`Q` into :math:`[0,1]` using the min and max seen
anywhere in the tree so far (eq. 5):

.. math:: \bar Q(s^{k-1}, a^k) = \frac{Q(s^{k-1},a^k)
          - \min_{s,a \in \mathrm{Tree}} Q(s,a)}
          {\max_{s,a \in \mathrm{Tree}} Q(s,a)
          - \min_{s,a \in \mathrm{Tree}} Q(s,a)},

and it is :math:`\bar Q`, not :math:`Q`, that enters eq. 2. The paper
is explicit that the alternatives -- rescaling by a known maximum score
or tuning the pUCT constants per game -- are game-specific and would be
adding prior knowledge, which is the thing MuZero is trying not to do.

This module is the search. Pass your own :math:`h`, :math:`g`,
:math:`f` as callables -- learned networks, or exact functions when you
have them, which is what makes the search testable against a known
optimum.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["muzero", "mcts_search"]


class _MinMax(object):
    """The tree-wide min/max of eq. 5."""

    def __init__(self):
        self.lo = None
        self.hi = None

    def update(self, v):
        self.lo = v if self.lo is None else min(self.lo, v)
        self.hi = v if self.hi is None else max(self.hi, v)

    def normalize(self, v):
        if self.lo is None or self.hi is None:
            return v
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        # Degenerate tree (every Q identical): eq. 5 is 0/0. Returning
        # the raw value keeps pUCT well-defined and, since every Q is
        # equal, leaves selection driven purely by the prior -- which is
        # the correct behaviour before any distinguishing evidence.
        return v


def muzero(observation, actions, representation, dynamics, prediction,
           simulations=50, gamma=0.997, c1=1.25, c2=19652.0,
           dirichlet_alpha=None, exploration_fraction=0.25,
           temperature=1.0, seed=0):
    r"""Run MuZero's MCTS from one observation and return the search
    policy.

    Parameters
    ----------
    observation : object
        Passed straight to ``representation``.
    actions : sequence
        The action space.
    representation : callable
        :math:`h_\theta(o) \to s^0`.
    dynamics : callable
        :math:`g_\theta(s, a) \to (r, s')`.
    prediction : callable
        :math:`f_\theta(s) \to (p, v)` with ``p`` a sequence of priors
        over ``actions`` and ``v`` a scalar.
    simulations : int
        Number of simulations. Each makes at most one call to
        ``dynamics`` and one to ``prediction``.
    gamma : float
        Discount :math:`\gamma`. Board games use 1.
    c1, c2 : float
        The pUCT constants of eq. 2; the paper's values are the
        defaults.
    dirichlet_alpha : float, optional
        If given, mix Dirichlet noise into the root prior with weight
        ``exploration_fraction``, as AlphaZero and MuZero do at the
        root during self-play. Omitted by default so the search is
        deterministic.
    exploration_fraction : float
        Weight of that noise.
    temperature : float
        Temperature on the visit-count distribution that forms the
        search policy. ``0`` makes it greedy.
    seed : int
        Seed for the root noise.

    Returns
    -------
    RichResult
        ``estimate`` / ``policy`` is the visit-count distribution over
        ``actions`` -- MuZero's search policy, the target its network
        is trained toward. ``action`` is the sampled/greedy choice,
        ``value`` the root value :math:`\sum_a N(a) Q(a) / \sum_a N(a)`,
        ``visits``, ``Q`` and ``prior`` the root statistics, and
        ``n_dynamics_calls`` / ``n_prediction_calls`` the model-call
        counts, which the paper bounds at one each per simulation.

    References
    ----------
    Schrittwieser et al. (2020), arXiv:1911.08265, Appendix B: eqs.
    2-5.
    """
    A = list(actions)
    if not A:
        raise ValueError("muzero: actions must be non-empty")
    for fn, name in ((representation, "representation"),
                     (dynamics, "dynamics"), (prediction, "prediction")):
        if not callable(fn):
            raise TypeError("muzero: %s must be callable" % name)
    simulations = int(simulations)
    if simulations < 1:
        raise ValueError("muzero: simulations must be >= 1")
    if c2 <= 0.0:
        raise ValueError("muzero: c2 must be > 0")

    calls = [0, 0]                     # dynamics, prediction

    def predict(s):
        calls[1] += 1
        p, v = prediction(s)
        p = [float(x) for x in p]
        if len(p) != len(A):
            raise ValueError("muzero: prediction returned %d priors for "
                             "%d actions" % (len(p), len(A)))
        tot = sum(p)
        if tot <= 0.0:
            raise ValueError("muzero: prior must have positive mass")
        return [x / tot for x in p], float(v)

    root = _Node()
    s0 = representation(observation)
    prior, _v0 = predict(s0)
    if dirichlet_alpha is not None:
        prior = _add_noise(prior, float(dirichlet_alpha),
                           float(exploration_fraction), seed)
    root.expand(s0, prior, A)

    mm = _MinMax()
    for _ in range(simulations):
        node = root
        path = [node]
        acts = []
        # --- Selection (eq. 2), descending while the node is expanded
        while node.expanded:
            a = _select(node, A, mm, c1, c2)
            acts.append(a)
            node = node.children[a]
            path.append(node)
        # --- Expansion: exactly one dynamics + one prediction call
        parent = path[-2]
        calls[0] += 1
        r, s = dynamics(parent.state, acts[-1])
        node.reward = float(r)
        p, v = predict(s)
        node.expand(s, p, A)
        # --- Backup (eqs. 3-4)
        _backup(path, float(v), gamma, mm)

    visits = [root.children[a].visits for a in A]
    total = float(sum(visits))
    if total <= 0:
        raise ValueError("muzero: no simulations reached the root's "
                         "children")
    if temperature == 0:
        best = max(range(len(A)), key=lambda i: visits[i])
        policy = [1.0 if i == best else 0.0 for i in range(len(A))]
    else:
        w = [v ** (1.0 / float(temperature)) for v in visits]
        sw = sum(w)
        policy = [x / sw for x in w]
    root_value = sum(root.children[a].visits * root.children[a].value()
                     for a in A) / total

    return RichResult(payload={
        "estimate": policy,
        "policy": policy,
        "action": A[max(range(len(A)), key=lambda i: policy[i])],
        "value": float(root_value),
        "visits": dict((A[i], visits[i]) for i in range(len(A))),
        "Q": dict((a, root.children[a].value()) for a in A),
        "prior": dict((a, root.children[a].prior) for a in A),
        "n_dynamics_calls": calls[0],
        "n_prediction_calls": calls[1],
        "simulations": simulations,
        "method": "MuZero MCTS (Schrittwieser et al. 2020, eqs. 2-5)",
    })


class _Node(object):
    __slots__ = ("visits", "value_sum", "prior", "children", "state",
                 "reward", "expanded")

    def __init__(self, prior=0.0):
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.state = None
        self.reward = 0.0
        self.expanded = False

    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def expand(self, state, prior, actions):
        self.state = state
        self.expanded = True
        for i, a in enumerate(actions):
            self.children[a] = _Node(prior[i])


def _select(node, A, mm, c1, c2):
    """eq. 2, with Q normalised by eq. 5."""
    total = sum(node.children[a].visits for a in A)
    sqrt_total = math.sqrt(total) if total > 0 else 0.0
    best = None
    best_a = A[0]
    for a in A:
        ch = node.children[a]
        explore = (ch.prior * sqrt_total / (1.0 + ch.visits)
                   * (c1 + math.log((total + c2 + 1.0) / c2)))
        q = mm.normalize(ch.value()) if ch.visits > 0 else 0.0
        score = q + explore
        if best is None or score > best:
            best = score
            best_a = a
    return best_a


def _backup(path, value, gamma, mm):
    """eqs. 3-4: G^k accumulated backwards, folded into a running mean."""
    g = value
    for node in reversed(path):
        node.value_sum += g
        node.visits += 1
        mm.update(node.value())
        g = node.reward + gamma * g


def _add_noise(prior, alpha, frac, seed):
    """Dirichlet(alpha) noise at the root, mixed with weight frac."""
    if alpha <= 0.0:
        raise ValueError("muzero: dirichlet_alpha must be > 0")
    if not 0.0 <= frac <= 1.0:
        raise ValueError("muzero: exploration_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    # Dirichlet(alpha,...,alpha) via normalised Gamma(alpha, 1) draws,
    # by Marsaglia-Tsang.
    g = [_gamma(alpha, rng) for _ in prior]
    s = sum(g)
    noise = [x / s for x in g]
    return [(1.0 - frac) * prior[i] + frac * noise[i]
            for i in range(len(prior))]


def _gamma(alpha, rng):
    if alpha < 1.0:
        u = rng.random()
        return _gamma(alpha + 1.0, rng) * (u ** (1.0 / alpha))
    d = alpha - 1.0 / 3.0
    cc = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.standard_normal()
        v = (1.0 + cc * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


def cheatsheet():
    return ("muzero: MCTS over a LEARNED latent model -- h (represent), "
            "g (dynamics -> reward, next latent), f (predict -> prior, "
            "value); no observation is ever reconstructed. pUCT eq. 2 "
            "with c1=1.25, c2=19652; backup eqs. 3-4 form the l-k step "
            "bootstrapped return G^k and fold it into a running mean; "
            "Q is min-max normalised over the whole tree (eq. 5) "
            "because values are unbounded. Search policy = visit "
            "counts. One g and one f call per simulation.")


# compact alias per ledger/NAMING.md
mcts_search = muzero
