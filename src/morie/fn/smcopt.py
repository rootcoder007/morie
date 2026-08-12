r"""SMC for global optimisation (annealed sequence of targets).

Del Moral, P., Doucet, A., & Jasra, A. (2006) "Sequential Monte Carlo
samplers", *JRSS-B* 68(3), 411-436, section 2.3.1(c).

The optimisation route is one of the three sequences the paper lists for
:math:`\{\pi_n\}`: "For global optimization, as in simulated annealing, we
can select" :math:`\pi_n(x) \propto \pi(x)^{\phi_n}` with
:math:`\phi_n` increasing. As :math:`\phi \to \infty` the target concentrates
on the modes of :math:`\pi`, so running the sampler of
:mod:`morie.fn.smcsam` up a rising ladder turns it into an optimiser: the
particles migrate to the maxima and the best point visited is the answer.

The difference from simulated annealing is the population. A single annealed
chain can be stuck in a local mode when the temperature drops; here
:math:`N` particles are annealed together and **resampled**, so the ones in
poor modes are killed off and replaced by copies of the ones in good modes.
That is what the paper means by saying the algorithms "interact".

This module is a front end: the sampler, the incremental weights of equation
31, the ESS criterion and the resampling schemes all live in
:mod:`morie.fn.smcsam` and are not reimplemented here. What is added is the
optimisation view -- a ladder that runs to a caller-chosen inverse
temperature rather than stopping at 1, tracking of the best point seen, and
the objective stated as something to *maximise* rather than as a log density.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

from .smcsam import random_walk_kernel, smcsam

__all__ = ["smcopt", "sequential_mc", "smc_optimise", "annealing_ladder"]


def annealing_ladder(n_steps, phi_max=50.0, phi_min=0.1, kind="geometric"):
    r"""An increasing :math:`\phi_n` for
    :math:`\pi_n \propto \pi^{\phi_n}`.

    ``geometric`` is geometric in :math:`\phi` (equal ratios), which is the
    usual annealing schedule; ``linear`` steps evenly.
    """
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("smcopt: need at least two steps")
    if phi_min <= 0 or phi_max <= phi_min:
        raise ValueError("smcopt: need 0 < phi_min < phi_max")
    if kind == "geometric":
        r = (phi_max / phi_min) ** (1.0 / (n_steps - 1))
        return [phi_min * r ** t for t in range(n_steps)]
    if kind == "linear":
        return [phi_min + (phi_max - phi_min) * t / (n_steps - 1)
                for t in range(n_steps)]
    raise ValueError("smcopt: kind must be 'geometric' or 'linear'")


def smcopt(objective, initial, n_particles=200, n_steps=30, phi_max=50.0,
           phi_min=0.1, kind="geometric", kernel=None, ess_threshold=0.5,
           scheme="systematic", seed=0, maximise=True):
    r"""Global optimisation by an annealed SMC sampler.

    Parameters
    ----------
    objective : callable
        ``objective(x) -> float``. Maximised by default; set
        ``maximise=False`` to minimise. The annealed target is
        :math:`\exp(\phi_n f(x))`, which is
        :math:`\pi^{\phi_n}` for :math:`\pi = e^{f}`.
    initial : callable
        ``initial(rng) -> x``, the starting spread of particles. Make it
        wide: annealing cannot find a mode no particle ever visits.
    n_particles, n_steps : int
        Population size and ladder length.
    phi_max, phi_min, kind : float, float, str
        The inverse-temperature ladder; see :func:`annealing_ladder`.
    kernel : callable, optional
        A :math:`\pi_n`-invariant move. Defaults to a random walk whose
        scale shrinks as :math:`\phi` grows, since the target sharpens.
    ess_threshold, scheme, seed
        Passed to :func:`morie.fn.smcsam.smcsam`.

    Returns
    -------
    RichResult
        ``estimate`` / ``best_x`` is the best point seen and ``best_value``
        its objective; ``particles`` and ``weights`` the final population;
        ``ladder``, ``ess_trace``, ``resampled`` describe the run.

    Examples
    --------
    A bimodal objective whose global maximum is the narrow peak::

        f = lambda x: max(3.0 * math.exp(-20 * (x[0] - 2) ** 2),
                          2.0 * math.exp(-2 * x[0] ** 2))
        smcopt(f, lambda rng: [6.0 * rng.random() - 3.0])["best_x"]

    References
    ----------
    Del Moral, Doucet & Jasra (2006) *JRSS-B* 68(3), 411-436,
    section 2.3.1(c).
    """
    sign = 1.0 if maximise else -1.0
    ladder = annealing_ladder(n_steps, phi_max, phi_min, kind)
    best = {"x": None, "v": float("-inf")}

    def log_gamma(x, phi):
        v = sign * float(objective(x))
        if v > best["v"]:
            best["v"] = v
            best["x"] = list(x)
        return phi * v

    if kernel is None:
        base = random_walk_kernel(scale=1.0)

        def kernel(x, log_target, rng, _b=base):
            return _b(x, log_target, rng)

    fit = smcsam(log_gamma, initial, n_particles=n_particles,
                 ladder=ladder, kernel=kernel,
                 ess_threshold=ess_threshold, scheme=scheme, seed=seed)
    if best["x"] is None:
        raise ValueError("smcopt: the objective was never evaluated")
    return RichResult(payload={
        "estimate": best["x"],
        "best_x": best["x"],
        "best_value": sign * best["v"],
        "particles": fit["particles"],
        "weights": fit["weights"],
        "particle_mean": fit["mean"],
        "ladder": ladder,
        "ess_trace": fit["ess_trace"],
        "resampled": fit["resampled"],
        "accept_trace": fit["accept_trace"],
        "n_particles": int(n_particles),
        "maximise": bool(maximise),
        "note": "annealing concentrates on the modes but cannot find one "
                "no particle visits; widen `initial` before raising "
                "phi_max",
        "method": "annealed SMC optimisation (Del Moral, Doucet & Jasra "
                  "2006, section 2.3.1c)",
    })


def cheatsheet():
    return ("smcopt: SMC as a global optimiser (Del Moral, Doucet & Jasra "
            "2006, sec 2.3.1c). Anneal pi_n = pi^phi_n with phi rising, so "
            "the target concentrates on the modes. Unlike single-chain "
            "simulated annealing the particles INTERACT: resampling kills "
            "the ones in poor modes and copies the ones in good modes. "
            "Shares the sampler, weights and resampling with smcsam.")


# names carried over / compact aliases
smc_optimise = smcopt
sequential_mc = smcopt
