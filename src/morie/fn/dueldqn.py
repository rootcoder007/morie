# morie.fn -- function file (rootcoder007/morie)
r"""Dueling network: separate value and advantage streams.

A Q-function conflates two things: how good a state is, and how much
each action changes that. In most states the second barely matters --
every action leads to roughly the same place -- and forcing one head to
learn :math:`|A|` nearly-identical numbers wastes the samples.

The dueling architecture splits them, :math:`V(s)` and
:math:`A(s,a)`, and recombines. **The naive recombination is broken and
the paper says so plainly:**

.. math:: Q(s,a) = V(s) + A(s,a)

is *unidentifiable* -- add a constant to :math:`V` and subtract it from
:math:`A` and :math:`Q` is unchanged, so nothing pins either stream
down and the network drifts. Two fixes appear in the paper:

.. math:: Q = V + \Big(A(s,a) - \max_{a'} A(s,a')\Big)
          \quad\text{(8)}, \qquad
          Q = V + \Big(A(s,a) - \tfrac{1}{|A|}\sum_{a'}A(s,a')\Big)
          \quad\text{(9)}.

Eq. (8) forces zero advantage at the greedy action, which makes
:math:`V` exactly the value of the best action. Eq. (9) subtracts the
mean instead: it loses that interpretation but is more stable, because
the advantages need only follow the mean rather than chase a maximum
that jumps between actions. Eq. (9) is the default here, as in the
paper.

**The identity worth checking is that the fix actually fixes it.** Under
either aggregation, shifting the whole advantage stream by a constant
must leave :math:`Q` *exactly* unchanged -- that is what makes the
decomposition identified. The anchor shifts it and checks, and checks
the naive form fails the same test.

References
----------
Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M. & de
Freitas, N. (2016) "Dueling Network Architectures for Deep
Reinforcement Learning", *Proceedings of the 33rd International
Conference on Machine Learning*, PMLR 48, 1995-2003,
arXiv:1511.06581. Eq. (7)-(9) and the identifiability argument.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J.,
Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K.,
Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I.,
King, H., Kumaran, D., Wierstra, D., Legg, S. & Hassabis, D. (2015)
"Human-level control through deep reinforcement learning", *Nature*
518(7540), 529-533, doi:10.1038/nature14236. The DQN this restructures.

van Hasselt, H., Guez, A. & Silver, D. (2016) "Deep Reinforcement
Learning with Double Q-learning", *Proceedings of the AAAI Conference
on Artificial Intelligence* 30(1), arXiv:1509.06461. The target the
paper combines with.

Baird, L. C. (1993) "Advantage Updating", Technical Report
WL-TR-93-1146, Wright Laboratory. The advantage function itself.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dueling_aggregate", "dueling_q", "double_q_target",
           "td_error"]

_AGG = ("mean", "max", "naive")


def dueling_aggregate(value, advantage, mode="mean"):
    r"""Eq. (9) by default, eq. (8) with ``mode="max"``.

    ``mode="naive"`` is :math:`V + A` with no correction, which is
    unidentifiable. It is here so that the failure can be demonstrated
    rather than described.
    """
    if mode not in _AGG:
        raise ValueError("duel: mode must be one of %s, got %r"
                         % (", ".join(_AGG), mode))
    a = [float(v) for v in advantage]
    if not a:
        raise ValueError("duel: no actions")
    v = float(value)
    if mode == "mean":
        c = sum(a) / len(a)
    elif mode == "max":
        c = max(a)
    else:
        c = 0.0
    return [v + (a[i] - c) for i in range(len(a))]


def dueling_q(values, advantages, mode="mean"):
    """Aggregate a batch of states."""
    if len(values) != len(advantages):
        raise ValueError("duel: %d values but %d advantage rows"
                         % (len(values), len(advantages)))
    return [dueling_aggregate(values[i], advantages[i], mode=mode)
            for i in range(len(values))]


def double_q_target(reward, gamma, q_online_next, q_target_next,
                    done=False):
    r"""The Double-DQN target: the ONLINE net picks the action, the
    TARGET net values it.

    Using one net for both is what over-estimates: the max of a noisy
    estimate is biased upward, and the same noise then supplies the
    value.
    """
    if len(q_online_next) != len(q_target_next):
        raise ValueError("duel: online and target action counts differ")
    if done:
        return float(reward)
    a = max(range(len(q_online_next)), key=lambda i: q_online_next[i])
    return float(reward) + float(gamma) * q_target_next[a]


def td_error(q_sa, target):
    """The temporal-difference residual."""
    return float(target) - float(q_sa)


def dueling_step(value, advantage, action, reward, gamma,
                 next_value, next_advantage, next_target_value,
                 next_target_advantage, mode="mean", done=False):
    """One dueling + double-Q update's worth of quantities."""
    q = dueling_aggregate(value, advantage, mode=mode)
    if not 0 <= action < len(q):
        raise ValueError("duel: action %d out of range" % action)
    q_next_online = dueling_aggregate(next_value, next_advantage,
                                      mode=mode)
    q_next_target = dueling_aggregate(next_target_value,
                                      next_target_advantage, mode=mode)
    tgt = double_q_target(reward, gamma, q_next_online, q_next_target,
                          done=done)
    return RichResult(payload={
        "estimate": td_error(q[action], tgt), "td_error":
            td_error(q[action], tgt),
        "q": q, "q_taken": q[action], "target": tgt,
        "greedy_action": max(range(len(q)), key=lambda i: q[i]),
        "value": float(value), "advantage": list(advantage),
        "mode": mode, "n_actions": len(q),
        "method": "dueling aggregation eq. (9) with a Double-DQN "
                  "target, Wang et al. (2016)",
    })


def cheatsheet():
    return ("dueldqn: Q = V + (A - mean_a A) [eq. 9, default] or "
            "Q = V + (A - max_a A) [eq. 8]. Q = V + A alone is "
            "UNIDENTIFIABLE -- add c to V, subtract c from A, Q is "
            "unchanged and neither stream is pinned down. Under either "
            "correction a constant shift of the whole advantage stream "
            "leaves Q exactly unchanged; that is the fix working.")


# compact alias per ledger/NAMING.md
duelingq = dueling_q
__all__.append("dueling_step")
