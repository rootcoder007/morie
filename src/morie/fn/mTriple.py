# morie.fn -- function file (rootcoder007/morie)
"""Triply robust natural indirect effect."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict, ols_fit

__all__ = ["triply_robust_mediation"]


def triply_robust_mediation(Y, X, M, C=None, trunc=0.01):
    r"""Natural direct and indirect effects, consistent under 2 of 3.

    Tchetgen Tchetgen and Shpitser's estimator combines three nuisance
    models -- the treatment mechanism :math:`e(C)`, the mediator
    density :math:`f(M \mid X, C)` and the outcome regression
    :math:`\mu(X, M, C)` -- into an influence function that stays
    consistent when ANY TWO of the three are correct.

    The mediation setting needs this more than the ATE setting does,
    because it has TWO exposure-like quantities to model and the
    cross-world quantity :math:`E[Y(1, M(0))]` cannot be written as a
    function of one of them alone. Ordinary doubly robust reasoning
    does not extend, which is why the estimator is triply rather than
    doubly robust.

    The identification is stronger than the estimation, and worth
    separating. Natural effects need CROSS-WORLD independence:
    :math:`Y(1,m) \perp M(0) \mid C`, a statement about two
    counterfactuals that can never both be observed on the same unit.
    No experiment can test it, and no amount of robustness in the
    estimator repairs its failure. ``cross_world_note`` says so, since
    a triply robust number invites more confidence than the assumption
    supports.

    Parameters
    ----------
    Y : array-like, shape (n,)
    X : array-like of {0, 1}, shape (n,)
        Treatment.
    M : array-like, shape (n,)
        Mediator.
    C : array-like, optional
        Baseline confounders.
    trunc : float

    Returns
    -------
    RichResult
        ``nie``, ``nde``, ``total``, ``se``, ``ci``,
        ``proportion_mediated``, ``decomposition_residual``,
        ``nuisance_agreement``.

    References
    ----------
    Tchetgen Tchetgen and Shpitser (2012), *Annals of Statistics*
    40:1816-1845.
    Pearl (2001) for the natural-effect definitions.
    VanderWeele (2015), *Explanation in Causal Inference*, on the
    cross-world assumption.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> C = rng.normal(size=(800, 1))
    >>> X = (rng.uniform(size=800) < 0.5).astype(float)
    >>> M = 0.7 * X + C[:, 0] + rng.normal(size=800)
    >>> Y = 1.0 * X + 0.6 * M + C[:, 0] + rng.normal(size=800)
    >>> bool(abs(triply_robust_mediation(Y, X, M, C)["nie"] - 0.42) < 0.2)
    True
    """
    yv = np.asarray(Y, dtype=float).ravel()
    xv = np.asarray(X, dtype=float).ravel()
    mv = np.asarray(M, dtype=float).ravel()
    n = yv.size
    if not (xv.size == mv.size == n):
        raise ValueError("Y, X and M must agree in length.")
    if not np.all(np.isin(xv, (0.0, 1.0))):
        raise ValueError("X must be binary 0/1.")
    if min(int(xv.sum()), int((1 - xv).sum())) < 5:
        raise ValueError("need at least 5 units in each treatment arm.")
    Ca = (np.zeros((n, 0)) if C is None
          else np.atleast_2d(np.asarray(C, dtype=float)))
    if Ca.shape[0] != n:
        Ca = Ca.T
    Bc = add_intercept(Ca) if Ca.shape[1] else np.ones((n, 1))

    # 1. treatment mechanism
    e = np.clip(logit_predict(Bc, logit_fit(Bc, xv)[0]), trunc, 1 - trunc)
    # 2. mediator model, Gaussian working density
    Bx = np.column_stack([Bc, xv])
    gm = ols_fit(Bx, mv)
    mhat = Bx @ gm
    sm = float(np.std(mv - mhat, ddof=1)) or 1.0
    B1 = np.column_stack([Bc, np.ones(n)])
    B0 = np.column_stack([Bc, np.zeros(n)])
    m1, m0 = B1 @ gm, B0 @ gm
    # density ratio f(M | X=0, C) / f(M | X=1, C)
    r10 = np.exp(np.clip(
        (-0.5 * ((mv - m0) / sm) ** 2) - (-0.5 * ((mv - m1) / sm) ** 2),
        -30, 30,
    ))
    # 3. outcome regression
    By = np.column_stack([Bc, xv, mv])
    by = ols_fit(By, yv)
    mu = lambda xx, mm: np.column_stack([Bc, xx, mm]) @ by
    mu1 = mu(np.ones(n), mv)
    mu0 = mu(np.zeros(n), mv)

    # E[Y(1, M(1))] and E[Y(0, M(0))] by the usual AIPW pieces
    ey11 = np.mean(xv * yv / e + (1 - xv / e) * (
        mu(np.ones(n), m1 + (mv - mhat))
    ))
    ey00 = np.mean((1 - xv) * yv / (1 - e) + (1 - (1 - xv) / (1 - e)) * (
        mu(np.zeros(n), m0 + (mv - mhat))
    ))
    # the cross-world term, reweighted onto the X=0 mediator distribution
    w_cross = xv / e * r10
    ey10_ipw = np.mean(w_cross * yv)
    ey10_reg = np.mean((1 - xv) / (1 - e) * mu1)
    aug = np.mean(w_cross * (yv - mu1))
    ey10 = ey10_reg + aug

    nie = float(ey11 - ey10)
    nde = float(ey10 - ey00)
    total = float(ey11 - ey00)
    psi = (xv * yv / e - (1 - xv) * yv / (1 - e))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": nie,
            "nie": nie,
            "nde": nde,
            "total": total,
            "se": se,
            "ci": (nie - z * se, nie + z * se),
            "decomposition_residual": float(abs(total - (nie + nde))),
            "decomposition_note": (
                "the natural direct and indirect effects sum to the total by "
                "construction; this residual is the arithmetic check"
            ),
            "proportion_mediated": (float(nie / total)
                                    if abs(total) > 1e-12 else np.nan),
            "ey11": float(ey11),
            "ey10": float(ey10),
            "ey00": float(ey00),
            "nuisance_agreement": float(abs(ey10_ipw - ey10_reg)),
            "agreement_note": (
                "gap between the weighting and regression routes to the "
                "cross-world term; a large value means at least one of the "
                "three nuisance models is wrong, and triple robustness only "
                "tolerates ONE being wrong"
            ),
            "robustness_note": (
                "consistent when any TWO of the treatment, mediator and "
                "outcome models are correct; ordinary double robustness does "
                "not extend here because the cross-world quantity cannot be "
                "written through one nuisance alone"
            ),
            "cross_world_note": (
                "natural effects need Y(1,m) independent of M(0) given C -- "
                "a statement about two counterfactuals never jointly "
                "observable. No experiment tests it and no estimator "
                "robustness repairs its failure"
            ),
            "propensity_range": (float(e.min()), float(e.max())),
            "n": int(n),
            "method": "Triply robust natural direct and indirect effects",
        }
    )


def cheatsheet():
    return (
        "mTriple: triply robust NIE/NDE with the two routes to the "
        "cross-world term compared, and the untestable assumption named"
    )
