r"""Batch-Constrained Q-learning, discrete form.

Fujimoto, S., Conti, E., Ghavamzadeh, M., & Pineau, J. (2019)
"Benchmarking Batch Deep Reinforcement Learning Algorithms",
arXiv:1910.01708, section 4 (discrete BCQ); the original continuous
method is Fujimoto, van Hoof & Meger (2019), "Off-Policy Deep
Reinforcement Learning without Exploration".

The failure being fixed is *extrapolation error*: a batch algorithm
evaluates :math:`Q(s,a)` for actions the batch does not contain, gets
an arbitrary answer, and then the :math:`\max` in the temporal
difference update selects precisely those arbitrary answers and
propagates them. The result is "extreme overestimation and poor
performance". BCQ's answer is to constrain the action space to actions
the behaviour policy would plausibly have taken.

In continuous control that requires a generative model plus a
perturbation network. In the discrete case it collapses to something
much simpler, because :math:`G_\omega(a \mid s) \approx \pi_b(a \mid
s)` can just be *computed*, and actions are eliminated by a threshold
(eq. 17):

.. math:: \pi(s) = \arg\max_{a\ \mid\
          G_\omega(a \mid s) / \max_{\hat a} G_\omega(\hat a \mid s)
          > \tau} Q_\theta(s, a).

The threshold is *relative* -- scaled by the largest probability the
behaviour model assigns at that state -- so it adapts to how peaked the
behaviour policy is rather than imposing an absolute floor. The same
constrained argmax replaces the max inside the backup (eq. 18):

.. math:: L(\theta) = \ell_\kappa\Big(r + \gamma
          \max_{a'\ \mid\ G_\omega(a' \mid s')/\max_{\hat a}
          G_\omega(\hat a \mid s') > \tau} Q_{\theta'}(s', a')
          - Q_\theta(s, a)\Big).

That single parameter spans the whole range of batch algorithms, and
the paper says so explicitly: **:math:`\tau = 0` returns Q-learning**
(nothing is eliminated) **and :math:`\tau = 1` returns an imitator of
the batch** (only the single most likely action survives). Both ends
are checked in the anchors, because they are the cheapest possible
falsification of an implementation of this method.

One caveat about :math:`\tau = 0`, which matters here and not in the
paper. Eq. 17's inequality is *strict*, and in the deep setting
:math:`G_\omega` is a softmax network that never outputs exactly zero,
so at :math:`\tau = 0` every action survives. In the tabular case the
maximum-likelihood clone is the observed frequency, and an action the
batch never contains has :math:`G = 0` exactly -- so it is eliminated
even at :math:`\tau = 0`. That is the paper's inequality applied
faithfully, and it is the behaviour you want: a zero-frequency action
is precisely the extrapolation risk BCQ exists to remove. The identity
"``tau=0`` is Q-learning" therefore holds exactly when every action is
observed at every state, and otherwise gives Q-learning restricted to
the batch's support. Pass an explicit ``behavior`` with full support to
get the unrestricted version.

:math:`\ell_\kappa` is the Huber loss of the DQN lineage; ``huber_c``
sets :math:`\kappa`, and ``loss="squared"`` recovers plain least
squares.

Tabular and exact: :math:`G_\omega` is the empirical action frequency
at each state, the Q-table is fitted by the eq. 18 backup, and the
"target network" :math:`\theta'` is the previous sweep's table.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bcq", "batch_constrained_q"]


def bcq(dataset, states=None, actions=None, tau=0.3, gamma=0.99, lr=0.5,
        iters=2000, loss="huber", huber_c=1.0, behavior=None,
        tol=1e-12):
    r"""Fit a batch-constrained Q-function to a fixed dataset.

    Parameters
    ----------
    dataset : sequence
        Transitions ``(s, a, r, s_next)`` or ``(s, a, r, s_next,
        done)``.
    states, actions : sequence, optional
        Full state and action spaces. Inferred from the data when
        omitted -- but inferring the action set hides the very actions
        BCQ is meant to exclude, so pass the real one.
    tau : float
        :math:`\tau \in [0, 1]`. ``0`` is unconstrained Q-learning,
        ``1`` is imitation of the batch.
    gamma : float
        Discount.
    lr, iters, tol : float, int, float
        Fitting controls for the tabular backup.
    loss : {"huber", "squared"}
        :math:`\ell_\kappa` or plain squared error.
    huber_c : float
        :math:`\kappa` of the Huber loss.
    behavior : callable or dict, optional
        :math:`G_\omega(a \mid s)`. Defaults to the empirical action
        frequency in the batch, which in the tabular case *is* the
        maximum-likelihood behavioural clone.

    Returns
    -------
    RichResult
        ``estimate`` / ``q`` is ``{(s, a): value}``; ``policy`` the
        constrained argmax of eq. 17; ``allowed`` the surviving action
        set per state; ``behavior`` the fitted :math:`G_\omega`;
        ``value`` the constrained state values; ``n_eliminated`` how
        many state-action pairs the threshold removed; and
        ``bellman_error`` the final loss.

    References
    ----------
    Fujimoto, Conti, Ghavamzadeh & Pineau (2019) arXiv:1910.01708,
    eqs. 17-18.
    """
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("bcq: tau must lie in [0, 1], got %r" % (tau,))
    if loss not in ("huber", "squared"):
        raise ValueError("bcq: loss must be 'huber' or 'squared', got %r"
                         % (loss,))
    huber_c = float(huber_c)
    if huber_c <= 0.0:
        raise ValueError("bcq: huber_c must be > 0")

    D = []
    for t in dataset:
        if len(t) == 4:
            s, a, r, s1 = t
            done = False
        elif len(t) == 5:
            s, a, r, s1, done = t
        else:
            raise ValueError("bcq: each transition must be (s, a, r, "
                             "s_next) or (s, a, r, s_next, done)")
        D.append((s, a, float(r), s1, bool(done)))
    if not D:
        raise ValueError("bcq: dataset must be non-empty")

    S = list(states) if states is not None else sorted(
        set([t[0] for t in D] + [t[3] for t in D]), key=repr)
    A = list(actions) if actions is not None else sorted(
        set(t[1] for t in D), key=repr)
    if not S or not A:
        raise ValueError("bcq: states and actions must be non-empty")

    # G_omega ~ pi_b: the behavioural clone. In the tabular case the
    # maximum-likelihood clone is simply the observed frequency.
    if behavior is None:
        n_sa = {}
        n_s = {}
        for s, a, _r, _s1, _d in D:
            n_sa[(s, a)] = n_sa.get((s, a), 0) + 1
            n_s[s] = n_s.get(s, 0) + 1
        G = {}
        for s in S:
            for a in A:
                G[(s, a)] = (n_sa.get((s, a), 0) / float(n_s[s])
                             if s in n_s else 1.0 / len(A))
    elif callable(behavior):
        G = dict(((s, a), float(behavior(s, a))) for s in S for a in A)
    else:
        G = dict(((s, a), float(behavior[(s, a)])) for s in S for a in A)
    for s in S:
        tot = sum(G[(s, a)] for a in A)
        if tot <= 0.0:
            raise ValueError("bcq: G(.|%r) is all zero" % (s,))

    # eq. 17's constraint set, computed once: the relative threshold
    # depends only on G, not on Q.
    allowed = {}
    for s in S:
        mx = max(G[(s, a)] for a in A)
        keep = [a for a in A if mx > 0 and G[(s, a)] / mx > tau]
        if not keep:
            # tau = 1 eliminates everything under a strict inequality;
            # the paper's stated behaviour at tau = 1 is imitation, so
            # the argmax of G survives.
            keep = [max(A, key=lambda a: G[(s, a)])]
        allowed[s] = keep

    Q = dict(((s, a), 0.0) for s in S for a in A)
    for _ in range(int(iters)):
        target = {}
        cnt = {}
        for s, a, r, s1, done in D:
            if done:
                t = r
            else:
                # eq. 18: the max runs over the CONSTRAINED set.
                t = r + gamma * max(Q[(s1, b)] for b in allowed[s1])
            target[(s, a)] = target.get((s, a), 0.0) + t
            cnt[(s, a)] = cnt.get((s, a), 0) + 1
        delta = 0.0
        for k in target:
            tgt = target[k] / cnt[k]
            err = tgt - Q[k]
            if loss == "huber" and abs(err) > huber_c:
                err = huber_c * (1.0 if err > 0 else -1.0)
            step = lr * err
            Q[k] += step
            if abs(step) > delta:
                delta = abs(step)
        if delta < tol:
            break

    policy = dict((s, max(allowed[s], key=lambda a: Q[(s, a)])) for s in S)
    value = dict((s, max(Q[(s, a)] for a in allowed[s])) for s in S)

    berr = 0.0
    for s, a, r, s1, done in D:
        t = r if done else r + gamma * max(Q[(s1, b)] for b in allowed[s1])
        e = t - Q[(s, a)]
        berr += (0.5 * e * e if loss == "squared" or abs(e) <= huber_c
                 else huber_c * (abs(e) - 0.5 * huber_c))
    berr /= len(D)

    n_elim = sum(len(A) - len(allowed[s]) for s in S)
    return RichResult(payload={
        "estimate": Q,
        "q": Q,
        "policy": policy,
        "allowed": allowed,
        "behavior": G,
        "value": value,
        "n_eliminated": n_elim,
        "bellman_error": float(berr),
        "tau": tau,
        "gamma": float(gamma),
        "n_transitions": len(D),
        "method": "discrete BCQ (Fujimoto et al. 2019, eqs. 17-18)",
    })


def cheatsheet():
    return ("bcq: constrain the argmax to actions the behaviour policy "
            "plausibly took -- pi(s) = argmax_{a: G(a|s)/max G > tau} "
            "Q(s,a) (eq. 17), and the same constrained max inside the "
            "backup (eq. 18). Threshold is RELATIVE to the peak of G, "
            "so it adapts to how peaked pi_b is. tau=0 IS Q-learning, "
            "tau=1 IS imitation of the batch. Kills extrapolation "
            "error on actions the batch never contains.")


# compact alias per ledger/NAMING.md
batch_constrained_q = bcq
