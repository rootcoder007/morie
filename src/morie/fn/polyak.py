# morie.fn -- function file (rootcoder007/morie)
r"""Polyak averaging and soft target updates.

Two related ideas about not trusting the latest iterate.

**Averaging the iterates.** Stochastic approximation with a slowly
decaying step size produces iterates that rattle around the optimum.
Averaging them,
:math:`\bar\theta_T = \frac{1}{T}\sum_{t\le T}\theta_t`, gives an
estimator that is asymptotically optimal -- the same rate a
second-order method achieves, with no second derivatives computed. The
requirement is that the step size decay *slowly* (slower than
:math:`1/t`); with too fast a decay the iterates stop moving before
averaging can help. A running form with a fixed decay is also given,
so nothing needs storing.

**Soft target updates.** Q-learning with a function approximator uses
the network being trained to compute its own regression target, which
makes the update prone to divergence. DQN's answer is a target network
copied every :math:`C` steps. For actor-critic the copy is replaced by
a slow track,

.. math:: \theta' \leftarrow \tau\theta + (1-\tau)\theta',
          \qquad \tau \ll 1,

with :math:`\tau = 10^{-3}` reported. The target values are then
constrained to change slowly, which moves an unstable problem closer
to supervised learning -- where the targets do not move at all. The
price is delay: the target lags the online network by roughly
:math:`1/\tau` steps, and that trade is the point rather than a
side-effect. ``lag_halflife`` computes it, and the anchor checks the
geometric convergence against the closed form.

References
----------
Polyak, B. T. & Juditsky, A. B. (1992) "Acceleration of Stochastic
Approximation by Averaging", *SIAM Journal on Control and
Optimization* 30(4), 838-855, doi:10.1137/0330046. Averaging the
iterates of a slowly-decaying stochastic approximation attains the
asymptotically optimal rate.

Ruppert, D. (1988) *Efficient Estimations from a Slowly Convergent
Robbins-Monro Process*, Technical Report 781, School of Operations
Research and Industrial Engineering, Cornell University. The same
averaging idea, independently.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T.,
Tassa, Y., Silver, D. & Wierstra, D. (2016) "Continuous control with
deep reinforcement learning", *International Conference on Learning
Representations (ICLR 2016)*, arXiv:1509.02971. Sec. 3: the Q update
is prone to divergence because the network being updated also computes
the target; "soft" target updates theta' <- tau theta + (1 - tau)
theta' with tau << 1 replace DQN's periodic copy, constraining target
values to change slowly and moving the problem closer to supervised
learning; the supplementary details give tau = 0.001.

Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015) "Human-level
control through deep reinforcement learning", *Nature* 518, 529-533,
doi:10.1038/nature14236. The periodic-copy target network;
implemented in :mod:`dqnv`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["polyak_average", "running_average", "soft_update",
           "lag_halflife", "hard_update"]

_EPS = 1e-12


def polyak_average(iterates, burn_in=0):
    r"""The running mean of the iterates, discarding a burn-in."""
    X = [[float(v) for v in k.vec(t)] for t in iterates]
    if not X:
        raise ValueError("polyak: no iterates given")
    b = int(burn_in)
    if b >= len(X):
        raise ValueError("polyak: the burn-in of %d discards all %d "
                         "iterates" % (b, len(X)))
    keep = X[b:]
    d = len(keep[0])
    return {"average": [sum(t[f] for t in keep) / len(keep)
                        for f in range(d)],
            "n_averaged": len(keep), "burn_in": b}


def running_average(prev, new, decay=0.999):
    r"""An exponential form, so nothing needs storing."""
    a = float(decay)
    if not 0.0 < a < 1.0:
        raise ValueError("polyak: the decay must lie in (0,1), got %r"
                         % (decay,))
    p = [float(v) for v in k.vec(prev)]
    n = [float(v) for v in k.vec(new)]
    if len(p) != len(n):
        raise ValueError("polyak: the parameter vectors differ in "
                         "length (%d, %d)" % (len(p), len(n)))
    return [a * p[i] + (1.0 - a) * n[i] for i in range(len(p))]


def soft_update(target, online, tau=0.001):
    r""":math:`\theta' \leftarrow \tau\theta + (1-\tau)\theta'`."""
    t = float(tau)
    if not 0.0 < t <= 1.0:
        raise ValueError("polyak: tau must lie in (0,1], got %r"
                         % (tau,))
    a = [float(v) for v in k.vec(target)]
    b = [float(v) for v in k.vec(online)]
    if len(a) != len(b):
        raise ValueError("polyak: the networks differ in size (%d, "
                         "%d)" % (len(a), len(b)))
    return [t * b[i] + (1.0 - t) * a[i] for i in range(len(a))]


def hard_update(target, online, step, C=10000):
    r"""DQN's periodic copy, every :math:`C` steps."""
    if int(step) % int(C) == 0:
        return {"target": [float(v) for v in k.vec(online)],
                "copied": True}
    return {"target": [float(v) for v in k.vec(target)],
            "copied": False}


def lag_halflife(tau):
    r"""How long the target takes to cover half the gap.

    The gap decays as :math:`(1-\tau)^n`, so the half-life is
    :math:`\log(1/2)/\log(1-\tau)` -- about :math:`0.69/\tau` for
    small :math:`\tau`. That lag is the price of the stability.
    """
    t = float(tau)
    if not 0.0 < t < 1.0:
        raise ValueError("polyak: tau must lie in (0,1) for a "
                         "half-life, got %r" % (tau,))
    return {"halflife": math.log(0.5) / math.log(1.0 - t),
            "approx": math.log(2.0) / t, "tau": t,
            "note": "the target lags the online network; that delay "
                    "IS the stabiliser"}


def cheatsheet():
    return ("polyak: (1) averaging the iterates of a SLOWLY decaying "
            "stochastic approximation is asymptotically optimal -- the "
            "second-order rate without second derivatives, provided "
            "the step decays slower than 1/t. (2) Q-learning diverges "
            "because the network computes its own target; DQN copies "
            "the weights every C steps, DDPG instead TRACKS them, "
            "theta' <- tau theta + (1-tau) theta' with tau = 1e-3, so "
            "targets move slowly and the problem resembles supervised "
            "learning. The lag, about 0.69/tau steps, is the price.")


# compact alias per ledger/NAMING.md
polyakaveraging = polyak_average

# public names resolved by fn/_lazy_map.json
polyak_target = polyak_average
polyaktarget = polyak_average
