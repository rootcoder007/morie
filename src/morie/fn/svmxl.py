# morie.fn -- function file (rootcoder007/morie)
"""Mixed logit spatial vote"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rng import random_normal


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def mixed_logit_vote(
    x, *, ideal_point=None, status_quo=None, beta: float = 1.0, beta_sd: float = 0.5, n_draws: int = 1000, seed: int = 1
):
    r"""Random-coefficient (mixed) logit probability of voting for ``x`` over ``status_quo``.

    The distance sensitivity varies across voters, ``beta_r ~ N(beta,
    beta_sd^2)``; the probability is the simulated average ``(1/R) sum_r
    logit(beta_r (||v - s||^2 - ||v - x||^2))`` over ``n_draws`` Philox
    normal draws (Train 2009, section 6.6). ``beta_sd = 0`` gives the logit
    probability.

    References
    ----------
    Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
    Cambridge University Press, chapter 6.

    Examples
    --------
    >>> r = mixed_logit_vote([1.0], ideal_point=[0.0], status_quo=[2.0], beta_sd=0.0)
    >>> round(r.value, 6)
    0.952574
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    s = [0.0] * len(x) if status_quo is None else _vec(status_quo)
    dd = _d2(v, s) - _d2(v, x)
    z = [float(a) for a in random_normal(n_draws, seed=seed)]
    p = []
    for zr in z:
        u = (beta + beta_sd * zr) * dd
        p.append(1 / (1 + math.exp(-u)) if u >= 0 else math.exp(u) / (1 + math.exp(u)))
    m = math.fsum(p) / n_draws
    se = math.sqrt(math.fsum((a - m) ** 2 for a in p) / (n_draws - 1) / n_draws)
    return DescriptiveResult(name="svmxl", value=m, extra={"simulation_se": se, "n_draws": n_draws})


mixe = mixed_logit_vote


def cheatsheet() -> str:
    return "mixed_logit_vote(x, ideal_point, status_quo, beta, beta_sd) -> simulated mixed logit vote probability"


# compact alias per ledger/NAMING.md
mixedlogitvote = mixed_logit_vote
