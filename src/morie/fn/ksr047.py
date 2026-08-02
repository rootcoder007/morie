# morie.fn -- function file (rootcoder007/morie)
"""Kaplan-Meier Z-estimator map Psi(S)(t)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_survival_psi", "kosorok_ch2_kaplan_meier_self_consistency"]


def kosorok_survival_psi(S, t_grid, S0, L, G):
    r"""The estimating map whose root is the Kaplan-Meier estimator
    (Kosorok Eq. 2.11, p. 26):

    .. math:: \Psi(S)(t) = P\psi_{S,t}
              = S_0(t)L(t)
              + \int_0^t \frac{S_0(u)}{S(u)}\,dG(u)\,S(t) - S(t).

    ``S0``, ``L`` and ``G`` are SUPPLIED. The section fixes them for
    its own censoring model, and the passage stating (2.11) does not
    define them unambiguously enough to reconstruct; substituting
    plausible empirical stand-ins was tried and does NOT make the
    Kaplan-Meier estimator a root of the resulting map, so guessing
    them would have produced a module that looked right and was
    wrong. The printed functional is computed from whatever the
    caller supplies.

    Survival analysis becomes Z-ESTIMATION: instead of writing the
    Kaplan-Meier estimator down and studying it directly, it is
    characterised as the zero of a map between function spaces, and
    the general Z-estimator theory of Chapter 2 then supplies
    consistency, weak convergence and the bootstrap at once.

    The parameter here is a FUNCTION, and the norm is uniform, which
    is why the theory needs empirical processes rather than
    finite-dimensional asymptotics. ``sup_norm`` reports
    :math:`\|\Psi(S)\|_\infty`, and it is near zero exactly when
    ``S`` is close to the Kaplan-Meier solution.

    Parameters
    ----------
    S : array-like
        Candidate survival function on ``t_grid``.
    t_grid : array-like
        Grid everything is supplied on.
    S0 : array-like
        The true survival function on ``t_grid``.
    L : array-like
        The section's ``L`` on ``t_grid``.
    G : array-like
        The section's ``G`` on ``t_grid``.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``psi``, ``sup_norm``, ``parameter_is``,
        ``norm``, ``n``, ``method``.
    References
    ----------
    Kosorok, Ch. 2, Eq. (2.11), p. 26.
    """
    from ._kosorok import survival_psi

    tg = np.atleast_1d(np.asarray(t_grid, dtype=float))
    if tg.size < 2:
        raise ValueError(f"need at least 2 grid points, got {tg.size}.")
    psi = survival_psi(S, tg, S0, L, G)
    return RichResult(payload={
        "t_grid": tg, "psi": psi,
        "sup_norm": float(np.max(np.abs(psi))),
        "parameter_is": "a FUNCTION, so the norm is uniform",
        "norm": "supremum",
        "components_supplied": True,
        "n": int(tg.size),
        "method": "Kaplan-Meier as a Z-estimator (Eq. 2.11); its root is the estimator"})


def cheatsheet():
    return "ksr047: Kaplan-Meier is the ROOT of a map between function spaces"


#: Catalogue alias for :func:`kosorok_survival_psi`.
kosorok_ch2_kaplan_meier_self_consistency = kosorok_survival_psi
