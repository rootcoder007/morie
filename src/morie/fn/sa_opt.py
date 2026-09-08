r"""Simulated annealing.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983) "Optimization by
Simulated Annealing", *Science* **220**(4598), 671-680.

The Metropolis criterion (their Sec. "Simulated Annealing", after
Metropolis et al. 1953) accepts a proposed move by

.. math:: P(\text{accept}) = \begin{cases}
          1 & \Delta E \le 0\\
          e^{-\Delta E / T} & \Delta E > 0
          \end{cases}

so uphill moves are taken with a probability that falls as the
temperature does. The whole method is that schedule: at high :math:`T`
the walk is nearly free and explores; as :math:`T \to 0` it becomes
greedy descent and settles. Kirkpatrick's point is that cooling *slowly
enough* leaves the system in a near-ground state rather than the first
local minimum found, which is what plain descent gives.

Routes
------
``schedule`` selects the cooling law, all three in common use and the
first two named in the paper's discussion:

``"geometric"``
    :math:`T_k = T_0 \alpha^k`, the standard choice; ``alpha`` near 1
    cools slowly.
``"linear"``
    :math:`T_k = T_0 (1 - k/K)`.
``"logarithmic"``
    :math:`T_k = T_0 / \ln(k + e)`, the schedule for which convergence
    in probability to the global optimum can be proved (Geman & Geman
    1984); it is impractically slow, and is offered for exactly that
    reason -- it is the one with the guarantee.

Determinism: proposals and acceptances come from the shared RNG, so a
given ``seed`` reproduces the whole trajectory in both language arms.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["simulated_annealing", "sa_opt"]

_SCHEDULES = ("geometric", "linear", "logarithmic")


def _temperature(schedule, T0, k, n_iter, alpha):
    if schedule == "geometric":
        return T0 * (alpha ** k)
    if schedule == "linear":
        frac = 1.0 - (k / float(n_iter))
        return T0 * frac if frac > 0.0 else 0.0
    return T0 / math.log(k + math.e)


def simulated_annealing(fun, x0, step=1.0, T0=1.0, n_iter=1000,
                        schedule="geometric", alpha=0.99, lower=None,
                        upper=None, seed=0):
    r"""Minimise ``fun`` over a continuous box by simulated annealing.

    Parameters
    ----------
    fun : callable
        Objective, called on a list of floats.
    x0 : array-like
        Starting point.
    step : float
        Standard deviation of the Gaussian proposal, per coordinate.
    T0 : float
        Initial temperature. Set it so that a typical uphill move is
        accepted with moderate probability at the start; too cold and
        this is just noisy descent.
    n_iter : int
        Number of proposals.
    schedule : {"geometric", "linear", "logarithmic"}
        Cooling law; see the module docstring.
    alpha : float
        Geometric cooling ratio, ``schedule="geometric"`` only.
    lower, upper : array-like, optional
        Box constraints; proposals outside are clipped.
    seed : int
        RNG seed.

    Returns
    -------
    RichResult
        ``estimate`` is the best point *ever visited*, not the last one
        -- the chain is a random walk and its final state can be worse
        than its best.
    """
    x = [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]
    n = len(x)
    if n == 0:
        raise ValueError("simulated_annealing: x0 must be non-empty")
    sched = str(schedule).lower()
    if sched not in _SCHEDULES:
        raise ValueError(
            "simulated_annealing: schedule must be one of %s, got %r"
            % (", ".join(_SCHEDULES), schedule))
    T0 = float(T0)
    if T0 <= 0.0:
        raise ValueError(
            "simulated_annealing: T0 must be positive, got %r" % (T0,))
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError("simulated_annealing: n_iter must be at least 1")
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            "simulated_annealing: alpha must lie in (0, 1], got %r" % (alpha,))
    lo = None if lower is None else [float(v) for v in
                                     np.atleast_1d(np.asarray(lower, dtype=float))]
    hi = None if upper is None else [float(v) for v in
                                     np.atleast_1d(np.asarray(upper, dtype=float))]

    rng = np.random.default_rng(seed)
    f = float(fun(x))
    best_x, best_f = list(x), f
    n_acc = 0
    n_up = 0
    temps = []
    trace = [f]

    for k in range(1, n_iter + 1):
        T = _temperature(sched, T0, k, n_iter, alpha)
        temps.append(T)
        prop = [x[j] + float(step) * float(rng.standard_normal())
                for j in range(n)]
        if lo is not None:
            prop = [max(prop[j], lo[j]) for j in range(n)]
        if hi is not None:
            prop = [min(prop[j], hi[j]) for j in range(n)]
        fp = float(fun(prop))
        dE = fp - f

        if dE <= 0.0:
            accept = True
        elif T <= 0.0:
            accept = False
        else:
            # Metropolis: exp(-dE/T). Drawn even when it will not be
            # used would desynchronise the two arms, so the draw happens
            # only on this branch in both.
            accept = float(rng.uniform()) < math.exp(-dE / T)
            if accept:
                n_up += 1

        if accept:
            x, f = prop, fp
            n_acc += 1
            if f < best_f:
                best_x, best_f = list(x), f
        trace.append(f)

    return RichResult(payload={
        "estimate": best_x,
        "x": best_x,
        "fun": float(best_f),
        "final_x": x,
        "final_fun": float(f),
        "n_accepted": int(n_acc),
        "n_uphill_accepted": int(n_up),
        "acceptance_rate": n_acc / float(n_iter),
        "temperatures": temps,
        "trace": trace,
        "schedule": sched,
        "T0": T0,
        "n_iter": int(n_iter),
        "method": "Simulated annealing, Metropolis acceptance "
                  "(Kirkpatrick, Gelatt & Vecchi 1983)",
    })


def cheatsheet():
    return ("sa_opt: Metropolis accept exp(-dE/T) for dE>0, always for "
            "dE<=0; schedules geometric T0 a^k, linear, logarithmic "
            "T0/ln(k+e); returns the BEST point visited, not the last.")


sa_opt = simulated_annealing
