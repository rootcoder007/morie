r"""Deep meta-reinforcement learning: an RL algorithm that learns an RL
algorithm.

Wang, J. X., Kurth-Nelson, Z., Tirumala, D., Soyer, H., Leibo, J. Z.,
Munos, R., Blundell, C., Kumaran, D., & Botvinick, M. (2016) "Learning
to reinforcement learn", arXiv:1611.05763.

The claim, stated plainly by the paper: the system "is trained using
one RL algorithm, but whose recurrent dynamics implement a second,
quite separate RL procedure". The outer algorithm tunes the weights
slowly across many tasks; the inner one lives in the *activations* and
adapts within a single episode. Because the inner procedure is
learned, it is fitted to the structure of the task distribution -- it
can be far more sample-efficient than the outer algorithm that
produced it, on tasks drawn from that distribution.

Formalism (section 2). Let :math:`\mathcal{D}` be a distribution over
MDPs. At the start of each episode a task :math:`m \sim \mathcal{D}`
and an initial state are sampled and **the agent's recurrent state is
reset**. At each step the action is a function of the whole
within-episode history

.. math:: H_t = \{x_0, a_0, r_0, \dots, x_{t-1}, a_{t-1}, r_{t-1},
          x_t\},

and the weights are trained to maximise the sum of rewards over all
steps and episodes. Two consequences carry the method:

* the policy is **history-dependent**, so on a new MDP it can adapt
  without any weight change -- after training the weights are frozen
  and only the activations move;
* the recurrent state must be **reset between episodes**, or the
  agent carries one task's solution into the next. That reset is what
  makes each episode a fresh instance of the inner learning problem.

Because the history includes the previous action and the previous
reward -- not just the observation -- the recurrent net has everything
an RL rule needs as *input*, which is why an RL procedure can emerge
in its dynamics at all. :func:`history_features` builds exactly that
input.

What is implemented here is the meta-RL *loop* and its interfaces, over
a user-supplied task distribution: episodes over sampled MDPs, the
history-conditioned agent, the per-episode reset, and the frozen-weight
evaluation the paper's central claim is stated in terms of. The inner
"algorithm" is whatever your recurrent agent's dynamics amount to;
:func:`bandit_tasks` provides the paper's bandit family so the claim
can be exercised, and a tabular-history agent is supplied as a
reference agent whose adaptation is measurable.

This is not a re-implementation of the paper's A3C/LSTM training --
that is the *outer* algorithm, and the paper is explicit that
"comparisons between specific architectures are outside the scope";
the framework is the contribution.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mtdrl", "meta_rl", "bandit_tasks", "history_features",
           "TabularHistoryAgent"]


def bandit_tasks(n_arms=2, n_tasks=100, seed=0, structure="independent"):
    r"""A distribution :math:`\mathcal{D}` of bandit tasks.

    ``structure="independent"`` draws each arm's payoff probability
    uniformly. ``structure="paired"`` is the paper's dependent-arm
    family: :math:`p_2 = 1 - p_1`, so learning about one arm tells you
    everything about the other -- exactly the structure an adapted
    inner algorithm can exploit and a task-agnostic one cannot.
    """
    if structure not in ("independent", "paired"):
        raise ValueError("mtdrl: structure must be 'independent' or "
                         "'paired', got %r" % (structure,))
    n_arms = int(n_arms)
    if n_arms < 2:
        raise ValueError("mtdrl: need at least 2 arms")
    if structure == "paired" and n_arms != 2:
        raise ValueError("mtdrl: the paired family is defined for 2 arms")
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(int(n_tasks)):
        if structure == "paired":
            p = rng.random()
            tasks.append([p, 1.0 - p])
        else:
            tasks.append([rng.random() for _ in range(n_arms)])
    return tasks


def history_features(history, n_arms):
    r"""The inputs of section 2: the previous action (one-hot), the
    previous reward, and the step index.

    ``history`` is a list of ``(action, reward)`` pairs so far. The
    previous *reward* being an input is not incidental -- without it
    the recurrent dynamics have nothing to learn a reinforcement rule
    from.
    """
    feat = [0.0] * (n_arms + 2)
    if history:
        a, r = history[-1]
        feat[a] = 1.0
        feat[n_arms] = float(r)
    feat[n_arms + 1] = float(len(history))
    return feat


class TabularHistoryAgent(object):
    r"""A reference inner learner: history-dependent, weights fixed.

    Keeps per-arm counts and means *within the episode only* and acts
    :math:`\varepsilon`-greedily on them. Its "weights" are the
    exploration rate; nothing about it is updated across episodes, so
    any within-episode improvement it shows comes entirely from the
    history -- which is the property the paper's LSTM is claimed to
    have, made measurable.
    """

    def __init__(self, n_arms, epsilon=0.1, optimistic=1.0):
        self.n_arms = int(n_arms)
        self.epsilon = float(epsilon)
        self.optimistic = float(optimistic)
        self.reset()

    def reset(self):
        """Called at the start of every episode -- section 2's reset of
        the recurrent state."""
        self.counts = [0] * self.n_arms
        self.means = [self.optimistic] * self.n_arms

    def act(self, features, rng):
        if rng.random() < self.epsilon:
            return int(rng.random() * self.n_arms)
        best = max(self.means)
        cand = [i for i in range(self.n_arms) if self.means[i] >= best]
        return cand[int(rng.random() * len(cand))]

    def observe(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.means[action] += (reward - self.means[action]) / n


def mtdrl(tasks, agent, episode_length=100, n_arms=None, seed=0,
          reset_between_episodes=True):
    r"""Run the meta-RL evaluation loop of section 2.

    Parameters
    ----------
    tasks : sequence
        One entry per episode: the arm probabilities of the MDP drawn
        from :math:`\mathcal{D}`.
    agent : object
        Must provide ``reset()``, ``act(features, rng)`` and
        ``observe(action, reward)``. Weights are never touched here --
        this is the frozen-weight evaluation the paper's claim is
        stated in.
    episode_length : int
        Steps per episode.
    n_arms : int, optional
        Inferred from ``tasks``.
    seed : int
        Seed for arm draws and the agent's exploration.
    reset_between_episodes : bool
        Section 2's reset of the recurrent state. Setting it False is
        the ablation that shows what the reset is for: the agent then
        carries one task's solution into the next.

    Returns
    -------
    RichResult
        ``estimate`` / ``mean_reward`` is the mean reward per step;
        ``regret`` the cumulative regret against each task's best arm;
        ``reward_by_step`` the reward averaged across episodes at each
        within-episode step -- the learning curve of the *inner*
        algorithm, which is where adaptation shows up; and
        ``optimal_action_rate`` the same for the fraction of steps on
        the best arm.

    References
    ----------
    Wang et al. (2016) arXiv:1611.05763, section 2 (Formalism).
    """
    T = [list(map(float, t)) for t in tasks]
    if not T:
        raise ValueError("mtdrl: tasks must be non-empty")
    k = int(n_arms) if n_arms is not None else len(T[0])
    for t in T:
        if len(t) != k:
            raise ValueError("mtdrl: every task must have %d arms" % k)
    L = int(episode_length)
    if L < 1:
        raise ValueError("mtdrl: episode_length must be >= 1")
    for m in ("reset", "act", "observe"):
        if not hasattr(agent, m):
            raise TypeError("mtdrl: agent must provide %s()" % m)

    rng = np.random.default_rng(seed)
    total = 0.0
    regret = 0.0
    by_step = [0.0] * L
    opt_by_step = [0.0] * L
    per_episode = []

    for probs in T:
        if reset_between_episodes:
            agent.reset()
        best_p = max(probs)
        best_arms = set(i for i in range(k) if probs[i] >= best_p)
        hist = []
        ep_reward = 0.0
        for t in range(L):
            feats = history_features(hist, k)
            a = agent.act(feats, rng)
            if not 0 <= a < k:
                raise ValueError("mtdrl: agent chose arm %r outside "
                                 "0..%d" % (a, k - 1))
            r = 1.0 if rng.random() < probs[a] else 0.0
            agent.observe(a, r)
            hist.append((a, r))
            ep_reward += r
            total += r
            regret += best_p - probs[a]
            by_step[t] += r
            opt_by_step[t] += 1.0 if a in best_arms else 0.0
        per_episode.append(ep_reward)

    n_ep = float(len(T))
    return RichResult(payload={
        "estimate": total / (n_ep * L),
        "mean_reward": total / (n_ep * L),
        "total_reward": total,
        "regret": regret,
        "reward_by_step": [v / n_ep for v in by_step],
        "optimal_action_rate": [v / n_ep for v in opt_by_step],
        "episode_reward": per_episode,
        "n_episodes": len(T),
        "episode_length": L,
        "n_arms": k,
        "method": "meta-RL evaluation loop (Wang et al. 2016 sec. 2)",
    })


def cheatsheet():
    return ("mtdrl: deep meta-RL (Wang 2016). Train with one RL "
            "algorithm so the RECURRENT DYNAMICS implement a second, "
            "learned one. Policy conditions on the whole within-episode "
            "history H_t including the previous ACTION and REWARD; the "
            "recurrent state is RESET each episode, and after training "
            "the weights are frozen so all within-episode adaptation "
            "is in the activations. bandit_tasks(structure='paired') "
            "is the dependent-arm family whose structure an adapted "
            "inner algorithm can exploit.")


# compact aliases per ledger/NAMING.md
meta_rl = mtdrl
metarl = mtdrl
