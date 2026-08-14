# morie.fn -- function file (rootcoder007/morie)
r"""Targeting a simple statistical bandit problem.

An infinite i.i.d. sequence :math:`(W_n, Y_n(0), Y_n(1))` is disclosed
one step at a time. At step :math:`n` the context :math:`W_n` is
revealed, we choose a **randomised** action :math:`A_n \in \{0,1\}`
with probability :math:`g_n(1 \mid W_n)` that we design from
:math:`O_1,\dots,O_{n-1}`, and we receive only :math:`Y_n(A_n)` -- the
other reward is never seen. Contexts may be high-dimensional; rewards
lie in :math:`(0,1)`.

**Two goals that pull apart.** A bandit algorithm usually maximises
cumulative reward. The statistical goal here is *inference*: a
confidence interval for the mean reward under the optimal rule. An
algorithm that converges to always playing the better arm stops
producing the data needed to estimate the other one -- so the design
must keep randomising.

**Which is why the randomisation is bounded away from 0 and 1.**
Choosing :math:`g_n \in [\delta, 1-\delta]` costs some reward and buys
the positivity that both identification and the variance estimate
require. ``design_probability`` enforces it rather than letting a
greedy rule silently destroy the estimator's basis.

**The data are not i.i.d., and the estimator accounts for it.**
:math:`g_n` depends on the past, so :math:`O_1,\dots,O_n` are
dependent. The TMLE's influence terms are nevertheless a **martingale
difference sequence** with respect to the history -- each term has
conditional mean zero given the past *because* the action was
randomised with a known, past-measurable probability. Variance is the
sum of squares and the limit is normal by the martingale central limit
theorem. That known randomisation is what makes an adaptive design
analysable at all, and the anchor checks the martingale property
rather than assuming it.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 24 (Chambaz,
Zheng & van der Laan): an infinite i.i.d. sequence
(W_n, Y_n(0), Y_n(1)) sequentially and partially disclosed; the
context W_n revealed first, then a randomized action A_n carried out
with probability g_n(.|W_n) determined by the observations accrued so
far, and only the reward Y_n = Y_n(A_n) of the action taken granted --
the alternative never observed; a possibly high-dimensional context
set and rewards in the open unit interval.

Chambaz, A., Zheng, W. & van der Laan, M. J. (2017) "Targeted
sequential design for targeted learning inference of the optimal
treatment rule and its mean reward", *Annals of Statistics* 45(6),
2537-2564, doi:10.1214/16-AOS1534.

Lai, T. L. & Robbins, H. (1985) "Asymptotically efficient adaptive
allocation rules", *Advances in Applied Mathematics* 6(1), 4-22,
doi:10.1016/0196-8858(85)90002-8. The reward-maximising tradition
this departs from.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["design_probability", "run_bandit", "martingale_terms",
           "sequential_ci", "regret"]

_EPS = 1e-12


def design_probability(blip_estimate, delta=0.1, greedy=False):
    r"""The randomisation probability, kept inside
    :math:`[\delta, 1-\delta]`.

    ``greedy=True`` returns the unbounded rule, which maximises reward
    and destroys the positivity the inference needs -- offered so the
    trade can be measured.
    """
    d = float(delta)
    if not 0.0 < d < 0.5:
        raise ValueError("tlbandt: delta must lie in (0, 0.5), got %r"
                         % (delta,))
    if greedy:
        return 1.0 if float(blip_estimate) > 0.0 else 0.0
    return 1.0 - d if float(blip_estimate) > 0.0 else d


def run_bandit(W, Y1, Y0, blip_fn, delta=0.1, seed=0, greedy=False,
               burn_in=20):
    r"""Play the sequential design, revealing one reward per step.

    ``blip_fn(history)`` returns the current estimate of the blip
    used to design the next action -- so the design depends on the
    past, which is exactly what makes the data dependent.
    """
    rows = [[float(v) for v in r] for r in k.mat(W)]
    y1 = [float(v) for v in k.vec(Y1)]
    y0 = [float(v) for v in k.vec(Y0)]
    n = len(rows)
    if not (len(y1) == len(y0) == n):
        raise ValueError("tlbandt: the arms differ in length")
    rng = np.random.default_rng(seed)
    hist, A, Y, G = [], [], [], []
    for t in range(n):
        b = 0.0 if t < int(burn_in) else float(blip_fn(hist))
        g = 0.5 if t < int(burn_in) else design_probability(b, delta,
                                                            greedy)
        a = 1.0 if float(rng.uniform()) < g else 0.0
        r = y1[t] if a == 1.0 else y0[t]
        A.append(a)
        Y.append(r)
        G.append(g)
        hist.append({"W": rows[t], "A": a, "Y": r, "g": g})
    return {"A": A, "Y": Y, "g": G, "history": hist,
            "greedy": bool(greedy),
            "min_g": min(G), "max_g": max(G),
            "note": "only the reward of the action TAKEN is observed"}


def martingale_terms(A, Y, g, Q1, Q0, psi):
    r"""The influence terms, which must be a martingale difference
    sequence.

    Conditional mean zero given the past holds *because* the action
    was randomised with a known, past-measurable probability.
    """
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    gg = [float(v) for v in k.vec(g)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    n = len(a)
    if any(v <= 0.0 or v >= 1.0 for v in gg):
        raise ValueError("tlbandt: the design probability left (0,1) "
                         "-- a greedy rule destroys the positivity "
                         "the inference rests on")
    out = []
    for i in range(n):
        qa = q1[i] if a[i] == 1.0 else q0[i]
        h = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
        out.append(h * (y[i] - qa) + q1[i] - q0[i] - float(psi))
    return out


def sequential_ci(D, level=1.96):
    r"""Martingale variance and interval."""
    v = [float(q) for q in k.vec(D)]
    T = len(v)
    if T < 2:
        raise ValueError("tlbandt: at least 2 steps are needed")
    s2 = sum(q * q for q in v) / T
    se = math.sqrt(s2 / T)
    return {"se": se, "half_width": float(level) * se, "T": T,
            "note": "sum of squares, not the i.i.d. variance -- the "
                    "terms are dependent but uncorrelated"}


def regret(Y, Y1, Y0):
    r"""Reward foregone against always playing the better arm.

    Reported because bounded randomisation buys inference *with*
    regret, and hiding the cost would misrepresent the trade.
    """
    y = [float(v) for v in k.vec(Y)]
    a = [float(v) for v in k.vec(Y1)]
    b = [float(v) for v in k.vec(Y0)]
    n = len(y)
    best = [max(a[i], b[i]) for i in range(n)]
    return {"cumulative_regret": sum(best[i] - y[i]
                                     for i in range(n)),
            "mean_regret": sum(best[i] - y[i]
                               for i in range(n)) / n,
            "note": "the price of keeping the design randomised"}


def cheatsheet():
    return ("tlbandt: contexts arrive, we choose a RANDOMISED action "
            "with a probability we design from the past, and only the "
            "reward of the action taken is revealed. The goal is "
            "INFERENCE, not cumulative reward -- and those pull apart, "
            "because an algorithm that converges to one arm stops "
            "generating data about the other. So keep g in "
            "[delta, 1-delta]: it costs regret and buys positivity. "
            "The data are dependent, but the influence terms are a "
            "MARTINGALE difference sequence precisely because the "
            "randomisation probability is known and past-measurable.")


# compact alias per ledger/NAMING.md
statisticalbandit = run_bandit
