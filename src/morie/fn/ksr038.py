# morie.fn -- function file (rootcoder007/morie)
"""Uniform-entropy Donsker theorem."""

from . import _array_core as np

from scipy import integrate

from ._richresult import RichResult

__all__ = ["kosorok_ch2_donsker_uniform_entropy"]


def kosorok_ch2_donsker_uniform_entropy(N_uniform, envelope_sq_mean, F=None, P=None):
    r"""Uniform-entropy Donsker theorem: if

    .. math:: J(1, \mathcal F, L_2) = \int_0^1
              \sqrt{\log \sup_Q N(\epsilon\|F\|_{Q,2},
              \mathcal F, L_2(Q))}\, d\epsilon < \infty
              \quad\text{and}\quad P^*F^2 < \infty,

    then F is P-Donsker. Note the envelope condition is on the
    SQUARE here, not the first moment as in the GC version
    (:mod:`morie.fn.ksr037`) -- Donsker needs the second moment, and
    carrying the wrong one over is a real trap.

    Parameters
    ----------
    N_uniform : callable
        eps -> uniform covering number.
    envelope_sq_mean : float
        :math:`P^*F^2`.
    F, P : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``J``, ``entropy_integral_finite``,
        ``envelope_sq_integrable``, ``conditions_met``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (uniform-entropy Donsker).
    """
    def integrand(eps):
        n = float(N_uniform(eps))
        return float(np.sqrt(np.log(n))) if n >= 1 else 0.0

    J, _ = integrate.quad(integrand, 1e-10, 1.0, limit=200)
    e2 = float(envelope_sq_mean)
    jf = bool(np.isfinite(J) and J < 1e6)
    ef = bool(np.isfinite(e2) and e2 < np.inf)
    return RichResult(
        payload={"J": float(J), "entropy_integral_finite": jf,
                 "envelope_sq_integrable": ef,
                 "conditions_met": bool(jf and ef), "envelope_sq_mean": e2,
                 "method": "J(1, F, L2) < inf AND P*F^2 < inf => Donsker"}
    )


def cheatsheet():
    return "ksr038: Donsker needs P*F^2, not P*F -- second moment, not first"
