# morie.fn -- function file (rootcoder007/morie)
r"""Deep Q-networks: two devices that make the divergence go away.

Q-learning with a non-linear function approximator was known to be
unstable. The Nature paper isolates why and fixes it with two
mechanisms, neither of which changes the learning rule itself.

**Experience replay.** Transitions
:math:`e_t = (s_t, a_t, r_t, s_{t+1})` are stored, and updates are
computed on minibatches drawn **uniformly at random** from the pool.
That breaks the correlation in the observation sequence -- consecutive
frames are nearly identical, and training on them in order is training
on a moving, highly dependent distribution. It also smooths over
changes in the data distribution as the policy shifts, and lets each
transition be used many times.

**A target network held fixed.** The loss regresses
:math:`Q(s,a;\theta_i)` onto
:math:`r + \gamma\max_{a'}Q(s',a';\theta_i^-)`, where
:math:`\theta^-` is a *separate* copy updated only every :math:`C`
steps. Without it the target moves with every update -- the network
chases its own output, and an increase in :math:`Q(s,a)` immediately
raises the target for the neighbouring state, which is the feedback
loop that diverges.

**One learning rule, unchanged.** The update is ordinary Q-learning;
what changed is *what data it sees* and *what it regresses onto*. The
anchor exploits that: on a small tabular MDP the fixed point is known
in closed form, so convergence can be checked against the true
:math:`Q^*` rather than against another run.

**A detail that is not incidental**: rewards are clipped to
:math:`[-1,1]` so one learning rate works across games with wildly
different score scales -- at the cost of making the agent indifferent
between rewards of different magnitude.

References
----------
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J.,
Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K.,
Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I.,
King, H., Kumaran, D., Wierstra, D., Legg, S. & Hassabis, D. (2015)
"Human-level control through deep reinforcement learning", *Nature*
518(7540), 529-533, doi:10.1038/nature14236. The two key ideas:
experience replay, which randomises over the data to remove
correlations in the observation sequence and smooth over changes in
the data distribution; and an iterative update towards target values
that are only periodically updated, with theta^- held fixed between
updates and refreshed every C steps. The Methods section gives the
reward clipping to [-1, 1].

Watkins, C. J. C. H. & Dayan, P. (1992) "Q-learning", *Machine
Learning* 8, 279-292, doi:10.1007/BF00992698. The learning rule
itself.

Lin, L.-J. (1992) "Self-improving reactive agents based on
reinforcement learning, planning and teaching", *Machine Learning* 8,
293-321, doi:10.1007/BF00992699. Experience replay.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["ReplayBuffer", "td_target", "clip_reward",
           "q_learning", "bellman_residual"]

_EPS = 1e-12


class ReplayBuffer(object):
    r"""A finite cache; the oldest transitions are discarded.

    Sampling is uniform, which is the point -- sequential sampling
    would reproduce the correlation the buffer exists to break.
    """

    def __init__(self, capacity):
        if int(capacity) < 1:
            raise ValueError("dqnv: the capacity must be at least 1")
        self.capacity = int(capacity)
        self.data = []

    def add(self, s, a, r, s2, done=False):
        self.data.append((s, a, float(r), s2, bool(done)))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, n, rng):
        if not self.data:
            raise ValueError("dqnv: the buffer is empty")
        m = min(int(n), len(self.data))
        return [self.data[int(float(rng.uniform()) * len(self.data))
                          % len(self.data)] for _ in range(m)]

    def __len__(self):
        return len(self.data)


def clip_reward(r, lo=-1.0, hi=1.0):
    r"""Clip to :math:`[-1,1]`.

    One learning rate then works across games, at the cost of making
    the agent indifferent between a reward of 1 and of 100.
    """
    return max(float(lo), min(float(hi), float(r)))


def td_target(r, s2, Q_target, gamma=0.99, done=False):
    r""":math:`r + \gamma\max_{a'}Q(s',a';\theta^-)`, from the FROZEN
    copy."""
    if done:
        return float(r)
    row = Q_target[int(s2)]
    return float(r) + float(gamma) * max(row)


def bellman_residual(Q, P, R, gamma=0.99):
    r""":math:`\max_{s,a}|Q(s,a) - (R + \gamma P\max_a Q)|`.

    Zero exactly at :math:`Q^*`, so it anchors convergence against the
    true fixed point rather than against another run.
    """
    nS, nA = len(Q), len(Q[0])
    worst = 0.0
    for s in range(nS):
        for a in range(nA):
            t = R[s][a] + float(gamma) * sum(
                P[s][a][s2] * max(Q[s2]) for s2 in range(nS))
            worst = max(worst, abs(Q[s][a] - t))
    return worst


def q_learning(P, R, n_states, n_actions, gamma=0.99, alpha=0.1,
               steps=20000, C=100, buffer_size=1000, batch=16,
               seed=0, use_replay=True, use_target=True):
    r"""Tabular Q-learning with both devices switchable.

    Turning either off is what lets the anchor show what each buys.
    """
    nS, nA = int(n_states), int(n_actions)
    if nS < 1 or nA < 1:
        raise ValueError("dqnv: need at least one state and action")
    rng = np.random.default_rng(seed)
    Q = [[0.0] * nA for _ in range(nS)]
    Qt = [list(r) for r in Q]
    buf = ReplayBuffer(buffer_size)
    s = 0
    hist = []
    for t in range(int(steps)):
        a = int(float(rng.uniform()) * nA) % nA
        u, acc, s2 = float(rng.uniform()), 0.0, nS - 1
        for j in range(nS):
            acc += P[s][a][j]
            if u <= acc:
                s2 = j
                break
        r = clip_reward(R[s][a])
        buf.add(s, a, r, s2)
        batchset = (buf.sample(batch, rng) if use_replay
                    else [(s, a, r, s2, False)])
        for (bs, ba, br, bs2, bd) in batchset:
            y = td_target(br, bs2, Qt if use_target else Q, gamma, bd)
            Q[bs][ba] += float(alpha) * (y - Q[bs][ba])
        if use_target and (t + 1) % int(C) == 0:
            Qt = [list(row) for row in Q]
        if (t + 1) % max(1, int(steps) // 20) == 0:
            hist.append(bellman_residual(Q, P, R, gamma))
        s = s2
    return RichResult(payload={
        "estimate": Q, "Q": Q, "residual_history": hist,
        "final_residual": hist[-1] if hist else float("nan"),
        "greedy_policy": [max(range(nA), key=lambda a: Q[s][a])
                          for s in range(nS)],
        "used_replay": bool(use_replay),
        "used_target_network": bool(use_target),
        "C": int(C),
        "method": "Q-learning with experience replay and a frozen "
                  "target network; Mnih et al. (2015)",
    })


def cheatsheet():
    return ("dqnv: the LEARNING RULE is ordinary Q-learning; what "
            "changed is the data and the target. EXPERIENCE REPLAY "
            "samples transitions UNIFORMLY from a finite buffer, "
            "breaking the correlation between consecutive frames and "
            "smoothing distribution shift. The TARGET NETWORK is a "
            "frozen copy refreshed every C steps -- without it the "
            "network chases its own output, since raising Q(s,a) "
            "immediately raises the target at the neighbouring state. "
            "Rewards clipped to [-1,1] so one learning rate spans "
            "games, at the cost of indifference to magnitude.")


# compact alias per ledger/NAMING.md
deepqnetwork = q_learning
