# morie.fn -- function file (rootcoder007/morie)
"""Kaplan-Meier Hadamard derivative."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_kaplan_meier_derivative"]


def kosorok_ch2_kaplan_meier_derivative(S_0, L, G, h, t):
    r"""Hadamard derivative of the Kaplan-Meier map (Kosorok Ch. 2):

    .. math:: \dot\Psi_{\theta_0}(h)(t) = -\int_0^t
              \frac{S_0(t)\,h(u)}{S_0(u)}\,dG(u) - L(t)h(t).

    The derivative of the product-limit functional, which is what
    makes the Kaplan-Meier estimator's weak limit computable by the
    delta method: the survival curve is a smooth functional of the
    cumulative hazard, and this is that smoothness made explicit.

    Parameters
    ----------
    S_0 : callable
        Baseline survival function.
    L : callable
        The L(t) term of the derivative.
    G : callable
        Integrator; its increments dG are used.
    h : callable
        Direction.
    t : float or array-like
        Evaluation times.

    Returns
    -------
    RichResult
        keys: ``derivative``, ``integral_term``, ``boundary_term``,
        ``t``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the Kaplan-Meier example).
    """
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(tt < 0):
        raise ValueError("t must be non-negative.")
    out, ints, bnds = [], [], []
    for ti in tt:
        grid = np.linspace(0.0, float(ti), 400)
        if grid.size < 2:
            ints.append(0.0)
        else:
            Su = np.array([float(S_0(u)) for u in grid])
            if np.any(Su <= 0):
                raise ValueError("S_0 must be strictly positive on [0, t].")
            hu = np.array([float(h(u)) for u in grid])
            Gu = np.array([float(G(u)) for u in grid])
            integrand = float(S_0(ti)) * hu / Su
            ints.append(float(np.sum(0.5 * (integrand[1:] + integrand[:-1])
                                     * np.diff(Gu))))
        bnds.append(float(L(ti)) * float(h(ti)))
        out.append(-ints[-1] - bnds[-1])
    scalar = np.ndim(t) == 0
    return RichResult(
        payload={"derivative": out[0] if scalar else np.array(out),
                 "integral_term": ints[0] if scalar else np.array(ints),
                 "boundary_term": bnds[0] if scalar else np.array(bnds),
                 "t": t,
                 "method": "Psi-dot(h)(t) = -int_0^t S0(t)h(u)/S0(u) dG(u) - L(t)h(t)"}
    )


def cheatsheet():
    return "ksr052: KM Hadamard derivative; integral term plus boundary term"
