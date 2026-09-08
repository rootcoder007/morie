# morie.fn -- function file (rootcoder007/morie)
"""Uniform-entropy Glivenko-Cantelli theorem."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_uniform"]


def kosorok_ch2_glivenko_cantelli_uniform(N_uniform, envelope_mean, eps_grid=None,
                                          F=None, P=None):
    r"""Uniform-entropy Glivenko-Cantelli theorem: if

    .. math:: \sup_Q N(\epsilon\|F\|_{Q,1}, \mathcal F, L_1(Q))
              < \infty \quad \forall \epsilon > 0
              \qquad\text{and}\qquad P^*F < \infty,

    then F is P-Glivenko-Cantelli. Two conditions, and BOTH are
    checked -- an integrable envelope is not implied by finite
    entropy, and omitting it is the usual way this theorem is
    misapplied.

    Parameters
    ----------
    N_uniform : callable
        eps -> uniform covering number.
    envelope_mean : float
        :math:`P^*F`, the envelope's mean.
    eps_grid : sequence of float, optional
        Radii to check.
    F, P : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``entropy_finite``, ``envelope_integrable``,
        ``conditions_met`` (both), ``covering_numbers``, ``eps_grid``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (uniform-entropy GC).
    """
    if eps_grid is None:
        eps_grid = [0.5, 0.2, 0.1, 0.05, 0.01]
    eps_grid = [float(e) for e in eps_grid]
    if any(e <= 0 for e in eps_grid):
        raise ValueError("eps values must be positive.")
    counts = np.array([float(N_uniform(e)) for e in eps_grid])
    ent = bool(np.all(np.isfinite(counts)))
    env = float(envelope_mean)
    env_ok = bool(np.isfinite(env) and env < np.inf)
    return RichResult(
        payload={"entropy_finite": ent, "envelope_integrable": env_ok,
                 "conditions_met": bool(ent and env_ok),
                 "covering_numbers": counts, "eps_grid": np.array(eps_grid),
                 "envelope_mean": env,
                 "method": "Uniform entropy AND P*F < inf => GC (both required)"}
    )


def cheatsheet():
    return "ksr037: entropy alone is not enough; the envelope must integrate"
