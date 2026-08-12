r"""Reward machines and Q-Learning for Reward Machines (QRM).

Toro Icarte, R., Klassen, T. Q., Valenzano, R., & McIlraith, S. A.
(2018) "Using Reward Machines for High-Level Task Specification and
Decomposition in Reinforcement Learning", *ICML*, PMLR 80.

A **reward machine** (Definition 3.1) is a tuple
:math:`\langle U, u_0, \delta_u, \delta_r \rangle` over a set of
propositional symbols :math:`\mathcal{P}`: a finite set of machine
states :math:`U`, an initial state :math:`u_0`, a state-transition
function :math:`\delta_u : U \times 2^{\mathcal{P}} \to U`, and a
reward-transition function :math:`\delta_r : U \times U \to [S \times A
\times S \to \mathbb{R}]`. At each step the machine reads the truth
assignment :math:`\sigma_t = L(s_t)` produced by a labelling function
:math:`L : S \to 2^{\mathcal{P}}`, moves to
:math:`u_{t+1} = \delta_u(u_t, \sigma_t)`, and emits the reward function
:math:`\delta_r(u_t, u_{t+1})`.

A machine is **simple** (Definition 3.2) when every
:math:`\delta_r(u, u')` is a constant. Simple machines are what the
paper's figures use and what ``rmrl`` builds by default: an edge is
written :math:`\langle \varphi, c \rangle`, meaning "take this edge when
the truth assignment satisfies :math:`\varphi`, and pay :math:`c`".

Why bother: the reward may be **non-Markovian in the environment
state**. "Deliver coffee to the office without breaking a decoration"
is not a function of where the agent is standing; it is a function of
where it has been. Folding the history into :math:`U` makes the joint
process Markovian again (Definition 3.3, MDPRM) *and* exposes the task
structure to the learner.

**QRM** (Algorithm 1) is what exploits that structure. It keeps one
q-function per machine state, :math:`\tilde q_j`, and after every real
environment step :math:`(s, a, s')` it updates *all* of them
counterfactually -- for each machine state :math:`u_j` it computes where
the machine *would* have gone, :math:`u_k = \delta_u(u_j, L(s'))`, and
the reward it *would* have paid, :math:`r = \delta_r(u_j, u_k)`, then

.. math:: \tilde q_j(s, a) \xleftarrow{\ \alpha\ } r(s,a,s')
          + \gamma \max_{a'} \tilde q_k(s', a'),

or :math:`\tilde q_j(s,a) \xleftarrow{\alpha} r(s,a,s')` at a dead end.
One transition therefore trains every sub-policy at once. Because the
update is off-policy and no sub-policy is pruned, QRM converges to an
optimal policy in the tabular case -- which the paper contrasts with
hierarchical methods, which do prune.

Three things are implemented here:

``reward_machine(...)``
    Build a machine from an edge list. Formulas are given as sets of
    positive and negative literals, or as ``"true"``; the labelling of
    a state is a set of true propositions.
``rmrl(...)`` / ``qrm(...)``
    Algorithm 1, tabular, over one or many machines simultaneously
    (the paper's multi-task setting), on a user-supplied MDP given as
    a step function.
``qlearn_flat(...)``
    Plain tabular q-learning on the *product* state :math:`(s, u)`,
    which is the honest baseline: it sees the same information as QRM
    but gets only the one update per step. It is here so the
    decomposition claim can be measured rather than asserted, and the
    anchors measure it.

The environment interface is deliberately the ordinary one -- a
``step(s, a) -> (s_next, done)`` callable and a list of actions -- so a
user can put their own gridworld, queueing model or simulator behind it
without adopting anything from this package beyond the machine itself.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rmrl", "reward_machine", "qrm", "reward_machine_run",
           "qlearn_flat"]


class RewardMachine(object):
    r"""A simple reward machine :math:`\langle U, u_0, \delta_u, \delta_r
    \rangle` (Definitions 3.1-3.2).

    Edges are ``(u, formula, u_next, reward)``. ``formula`` is either
    the string ``"true"`` or a pair ``(positive, negative)`` of iterables
    of proposition names, read as "all of *positive* hold and none of
    *negative* does". Edges are tested in the order given and the first
    match wins, so a ``"true"`` edge acts as the default; if none
    matches, the machine stays in ``u`` and pays 0 -- the self-loop the
    paper draws as :math:`\langle \text{true}, 0 \rangle`.
    """

    def __init__(self, edges, u0=0, terminal=()):
        self.u0 = u0
        self.terminal = set(terminal)
        self.edges = {}
        self.states = set([u0]) | self.terminal
        for e in edges:
            if len(e) != 4:
                raise ValueError("reward_machine: each edge must be "
                                 "(u, formula, u_next, reward), got %r" % (e,))
            u, phi, u2, c = e
            self.edges.setdefault(u, []).append((_compile(phi), u2,
                                                 float(c)))
            self.states.add(u)
            self.states.add(u2)

    def step(self, u, sigma):
        r""":math:`(\delta_u(u, \sigma),\ \delta_r(u, \delta_u(u,\sigma)))`."""
        if u in self.terminal:
            return u, 0.0
        for test, u2, c in self.edges.get(u, ()):
            if test(sigma):
                return u2, c
        return u, 0.0

    def __repr__(self):
        return ("RewardMachine(|U|=%d, u0=%r, terminal=%r)"
                % (len(self.states), self.u0, sorted(self.terminal)))


def _compile(phi):
    if phi is None or (isinstance(phi, str) and phi.lower() == "true"):
        return lambda sigma: True
    if isinstance(phi, str):
        pos, neg = {phi}, set()
    else:
        try:
            p, n = phi
        except (TypeError, ValueError):
            raise ValueError("reward_machine: formula must be 'true', a "
                             "proposition name, or (positive, negative), "
                             "got %r" % (phi,))
        pos, neg = set(p), set(n)
    return lambda sigma: pos <= set(sigma) and not (neg & set(sigma))


def reward_machine(edges, u0=0, terminal=()):
    """Construct a :class:`RewardMachine`. See its docstring."""
    return RewardMachine(edges, u0=u0, terminal=terminal)


def reward_machine_run(machine, labels):
    r"""Drive a machine over a sequence of truth assignments.

    ``labels`` is :math:`\sigma_0, \sigma_1, \dots`, i.e. ``L(s)`` for
    each visited state. Returns the machine-state trajectory and the
    rewards emitted, which is what makes the non-Markovian reward
    inspectable without running any learning at all.
    """
    u = machine.u0
    us = [u]
    rs = []
    for sigma in labels:
        u, c = machine.step(u, sigma)
        us.append(u)
        rs.append(c)
    return RichResult(payload={
        "estimate": list(us),
        "states": us,
        "rewards": rs,
        "total_reward": float(sum(rs)),
        "final_state": u,
        "accepted": u in machine.terminal,
        "method": "reward machine run (Icarte et al. 2018 Def. 3.1)",
    })


def rmrl(machines, states, actions, step, label, gamma=0.9, alpha=0.5,
         epsilon=0.1, episodes=500, horizon=100, start=None,
         dead_end=None, seed=0, task_order=None):
    r"""Q-Learning for Reward Machines (Algorithm 1), tabular.

    Parameters
    ----------
    machines : RewardMachine or sequence of them
        :math:`\Sigma`, the list of tasks to learn simultaneously.
    states : sequence
        The (hashable) environment states :math:`S`.
    actions : sequence
        The (hashable) actions :math:`A`.
    step : callable
        ``step(s, a) -> s_next``, or ``-> (s_next, done)``. Must be
        deterministic or draw its own randomness; QRM's counterfactual
        update reuses the same :math:`(s, a, s')` for every machine
        state, which is exactly what makes it sound.
    label : callable
        :math:`L : S \to 2^{\mathcal{P}}`; returns an iterable of the
        propositions true in the state.
    gamma, alpha, epsilon : float
        Discount, learning rate, exploration rate for
        :math:`\varepsilon`-greedy behaviour.
    episodes, horizon : int
        Number of episodes and the per-episode step cap (line 6).
    start : callable or state, optional
        ``EnvInitialState()``. Defaults to ``states[0]``.
    dead_end : callable, optional
        ``EnvDeadEnd(s)`` of lines 7 and 15. Defaults to no dead ends.
    seed : int
        Seed for exploration and any tie-breaking.
    task_order : sequence of int, optional
        ``GetTask``. Defaults to the paper's round-robin over
        :math:`\Sigma`.

    Returns
    -------
    RichResult
        ``estimate`` / ``q`` is a dict ``(task, u, s, a) -> value``;
        ``policy`` is the greedy action for each ``(task, u, s)``;
        ``returns`` the undiscounted return of each training episode;
        ``mean_return_last`` the mean over the final tenth, which is the
        number to compare against ``qlearn_flat``.

    References
    ----------
    Toro Icarte et al. (2018), ICML PMLR 80, Definitions 3.1-3.3 and
    Algorithm 1 (lines 12-20 are the counterfactual update).
    """
    if isinstance(machines, RewardMachine):
        machines = [machines]
    machines = list(machines)
    if not machines:
        raise ValueError("rmrl: need at least one reward machine")
    S = list(states)
    A = list(actions)
    if not S or not A:
        raise ValueError("rmrl: states and actions must be non-empty")
    if not callable(step) or not callable(label):
        raise TypeError("rmrl: step and label must be callable")
    episodes = int(episodes)
    horizon = int(horizon)
    if episodes < 1 or horizon < 1:
        raise ValueError("rmrl: episodes and horizon must be >= 1")
    de = dead_end if callable(dead_end) else (lambda s: False)
    s0 = start if callable(start) else (lambda: S[0] if start is None
                                        else start)
    if task_order is None:
        task_order = [l % len(machines) for l in range(episodes)]
    else:
        task_order = list(task_order)
        if len(task_order) < episodes:
            raise ValueError("rmrl: task_order shorter than episodes")

    rng = np.random.default_rng(seed)
    # q[(task, u)][s][a] -- one q-function per machine state (line 2).
    q = {}
    for i, mm in enumerate(machines):
        for u in mm.states:
            q[(i, u)] = dict((s, dict((a, 0.0) for a in A)) for s in S)

    returns = []
    for l in range(episodes):
        i = task_order[l]
        mm = machines[i]
        u = mm.u0
        s = s0()
        total = 0.0
        for _t in range(horizon):
            if de(s) or u in mm.terminal:
                break
            a = _eps_greedy(q[(i, u)][s], A, epsilon, rng)
            out = step(s, a)
            done = False
            if isinstance(out, tuple):
                s1, done = out[0], bool(out[1])
            else:
                s1 = out
            sigma = set(label(s1))
            dead = de(s1)
            # Lines 12-20: update EVERY q-function of EVERY machine.
            for o, mo in enumerate(machines):
                for uj in mo.states:
                    uk, r = mo.step(uj, sigma)
                    if dead or uk in mo.terminal:
                        target = r
                    else:
                        target = r + gamma * max(q[(o, uk)][s1].values())
                    cur = q[(o, uj)][s][a]
                    q[(o, uj)][s][a] = cur + alpha * (target - cur)
            u, r_real = mm.step(u, sigma)
            total += r_real
            s = s1
            if done or dead:
                break
        returns.append(total)

    policy = {}
    for key, table in q.items():
        for s in S:
            policy[(key[0], key[1], s)] = max(A, key=lambda a: table[s][a])
    tenth = max(1, episodes // 10)
    return RichResult(payload={
        "estimate": q,
        "q": q,
        "policy": policy,
        "returns": returns,
        "mean_return_last": float(sum(returns[-tenth:]) / tenth),
        "mean_return_first": float(sum(returns[:tenth]) / tenth),
        "n_qfunctions": len(q),
        "episodes": episodes,
        "method": "QRM (Icarte et al. 2018, Algorithm 1)",
    })


def qlearn_flat(machine, states, actions, step, label, gamma=0.9, alpha=0.5,
                epsilon=0.1, episodes=500, horizon=100, start=None,
                dead_end=None, seed=0):
    r"""Tabular q-learning on the product state :math:`(s, u)`.

    The same information as QRM -- the machine state is part of the
    observation -- but only the *experienced* :math:`(s,u)` pair is
    updated per step, with no counterfactual sweep. This is the
    baseline the decomposition claim is measured against.
    """
    S = list(states)
    A = list(actions)
    de = dead_end if callable(dead_end) else (lambda s: False)
    s0 = start if callable(start) else (lambda: S[0] if start is None
                                        else start)
    rng = np.random.default_rng(seed)
    q = dict(((u, s), dict((a, 0.0) for a in A))
             for u in machine.states for s in S)
    returns = []
    for _l in range(int(episodes)):
        u = machine.u0
        s = s0()
        total = 0.0
        for _t in range(int(horizon)):
            if de(s) or u in machine.terminal:
                break
            a = _eps_greedy(q[(u, s)], A, epsilon, rng)
            out = step(s, a)
            done = False
            if isinstance(out, tuple):
                s1, done = out[0], bool(out[1])
            else:
                s1 = out
            u1, r = machine.step(u, set(label(s1)))
            if de(s1) or u1 in machine.terminal:
                target = r
            else:
                target = r + gamma * max(q[(u1, s1)].values())
            cur = q[(u, s)][a]
            q[(u, s)][a] = cur + alpha * (target - cur)
            total += r
            s, u = s1, u1
            if done or de(s1):
                break
        returns.append(total)
    tenth = max(1, int(episodes) // 10)
    return RichResult(payload={
        "estimate": q,
        "q": q,
        "returns": returns,
        "mean_return_last": float(sum(returns[-tenth:]) / tenth),
        "mean_return_first": float(sum(returns[:tenth]) / tenth),
        "method": "tabular q-learning on (s, u)",
    })


def _eps_greedy(row, A, epsilon, rng):
    """epsilon-greedy with ties broken UNIFORMLY AT RANDOM.

    Deterministic tie-breaking is not a harmless detail here: the table
    starts all-zero, so every action ties, and always returning
    ``A[0]`` turns the initial behaviour into a systematic drift in one
    direction rather than the random walk it is supposed to be. On any
    task whose reward sits the other way the agent then never sees a
    reward at all and nothing is ever learned.
    """
    if rng.random() < epsilon:
        return A[int(rng.random() * len(A))]
    bv = None
    best = []
    for a in A:
        v = row[a]
        if bv is None or v > bv:
            bv, best = v, [a]
        elif v == bv:
            best.append(a)
    if len(best) == 1:
        return best[0]
    return best[int(rng.random() * len(best))]


def cheatsheet():
    return ("rmrl: reward machine <U, u0, delta_u, delta_r> (Icarte "
            "2018 Def. 3.1) + QRM (Alg. 1): one q-function per machine "
            "state, every step updates ALL of them counterfactually via "
            "u_k = delta_u(u_j, L(s')). qlearn_flat is the (s,u) "
            "baseline. Handles rewards non-Markovian in s.")


# compact aliases per ledger/NAMING.md
qrm = rmrl
rewardmachine = reward_machine
