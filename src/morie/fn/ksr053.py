# morie.fn -- function file (rootcoder007/morie)
"""Inverse Kaplan-Meier Hadamard derivative."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_kaplan_meier_inverse"]


def kosorok_ch2_kaplan_meier_inverse(S_0, L, F_0, a, t):
    r"""Inverse of the Kaplan-Meier Hadamard derivative (Kosorok
    Ch. 2):

    .. math:: \dot\Psi^{-1}_{\theta_0}(a)(t) = -S_0(t)\Big\{a(0)
              + \int_0^t \frac{1}{L(u^-)S_0(u^-)}\,da(u)\Big\}.

    Existence of a CONTINUOUS inverse is the condition that turns weak
    convergence of the estimator into weak convergence of the
    functional both ways -- it is what lets confidence bands built on
    one scale transfer to the other. The left limits matter: the
    integrand uses :math:`u^-`, so evaluating at u instead would be
    wrong wherever L or S_0 jumps.

    Parameters
    ----------
    S_0 : callable
        Baseline survival.
    L : callable
        The L function.
    F_0 : ignored
        Interface compatibility.
    a : callable
        The element being inverted.
    t : float or array-like
        Evaluation times.

    Returns
    -------
    RichResult
        keys: ``inverse``, ``integral_term``, ``a_at_zero``, ``t``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (invertibility of the Kaplan-Meier derivative).
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(tt < 0):
        raise ValueError("t must be non-negative.")
    a0 = float(a(0.0))
    eps = 1e-9  # approach from the left, as the formula requires
    out, ints = [], []
    for ti in tt:
        grid = np.linspace(0.0, float(ti), 400)
        Lm = np.array([float(L(max(u - eps, 0.0))) for u in grid])
        Sm = np.array([float(S_0(max(u - eps, 0.0))) for u in grid])
        denom = Lm * Sm
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError("L(u-) S_0(u-) vanishes on [0, t]; inverse undefined.")
        au = np.array([float(a(u)) for u in grid])
        integrand = 1.0 / denom
        ints.append(float(np.sum(0.5 * (integrand[1:] + integrand[:-1])
                                 * np.diff(au))))
        out.append(-float(S_0(ti)) * (a0 + ints[-1]))
    scalar = np.ndim(t) == 0
    return RichResult(
        payload={"inverse": out[0] if scalar else np.array(out),
                 "integral_term": ints[0] if scalar else np.array(ints),
                 "a_at_zero": a0, "t": t,
                 "method": "Psi-dot^-1(a)(t) with LEFT limits in the integrand"}
    )


def cheatsheet():
    return "ksr053: continuous inverse; left limits u- matter at jumps"
