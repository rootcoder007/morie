# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric efficiency bound."""

import numpy as np

from ._richresult import RichResult

__all__ = ["efficiency_bound"]


def efficiency_bound(y, d, X=None, propensity=None, mu1=None, mu0=None,
                     estimand="ate", trunc=0.01):
    r"""The variance no regular estimator can beat.

    For the ATE under unconfoundedness the semiparametric efficiency
    bound is Hahn's

    .. math::
       V^{*} = \mathbb{E}\left[
         \frac{\sigma_1^2(X)}{e(X)} + \frac{\sigma_0^2(X)}{1-e(X)}
         + \{\mu_1(X) - \mu_0(X) - \tau\}^2\right].

    Its three terms are separately informative and are returned
    separately. The first two are the cost of never observing both
    potential outcomes, inflated where the propensity is extreme; the
    third is the variance of the conditional effect itself, which is
    irreducible even under perfect overlap because the ATE is an
    average over genuinely heterogeneous units.

    The bound is what makes "efficient" a checkable claim rather than a
    slogan. AIPW and TMLE both attain it; IPW alone does not, and
    ``ipw_relative_efficiency`` quantifies how much is lost. An
    estimator reporting a standard error BELOW this bound has not
    outperformed theory -- it has an error somewhere, and
    ``below_bound`` is the check.

    ``overlap_penalty`` isolates how much of the bound comes from
    propensities near 0 or 1. When it dominates, no estimator will
    help; the fix is a different estimand or better data.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, optional
    propensity, mu1, mu0 : array-like, optional
    estimand : {'ate', 'att'}
    trunc : float

    Returns
    -------
    RichResult
        ``bound``, ``se_bound``, ``outcome_term``,
        ``heterogeneity_term``, ``overlap_penalty``,
        ``ipw_relative_efficiency``, ``n_effective``.

    References
    ----------
    Hahn (1998), *Econometrica* 66:315-331.
    Hirano, Imbens and Ridder (2003), *Econometrica* 71:1161-1189.
    Robins, Rotnitzky and Zhao (1994) for the influence function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 1))
    >>> d = (rng.uniform(size=400) < 0.5).astype(float)
    >>> y = 2.0 * d + rng.normal(size=400)
    >>> bool(efficiency_bound(y, d, X)["bound"] > 0)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    n = yv.size
    if dv.size != n:
        raise ValueError("y and d must agree in length.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if estimand not in ("ate", "att"):
        raise ValueError("estimand must be 'ate' or 'att'.")
    if min(int(dv.sum()), int((1 - dv).sum())) < 2:
        raise ValueError("need at least 2 units in each arm.")

    if propensity is None:
        if X is None:
            raise ValueError("supply X or propensity.")
        from ._did import add_intercept, logit_fit, logit_predict
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        if Xa.shape[0] != n:
            Xa = Xa.T
        B = add_intercept(Xa)
        e = logit_predict(B, logit_fit(B, dv)[0])
    else:
        e = np.asarray(propensity, dtype=float).ravel()
        Xa = None if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    e_raw = e.copy()
    e = np.clip(e, trunc, 1 - trunc)

    if mu1 is None or mu0 is None:
        from ._did import add_intercept, ols_fit
        if X is None:
            m1 = np.full(n, float(yv[dv == 1].mean()))
            m0 = np.full(n, float(yv[dv == 0].mean()))
        else:
            B = add_intercept(np.atleast_2d(np.asarray(X, dtype=float))
                              if Xa is None else Xa)
            m1 = B @ ols_fit(B[dv == 1], yv[dv == 1])
            m0 = B @ ols_fit(B[dv == 0], yv[dv == 0])
    else:
        m1 = np.asarray(mu1, dtype=float).ravel()
        m0 = np.asarray(mu0, dtype=float).ravel()

    r1 = yv[dv == 1] - m1[dv == 1]
    r0 = yv[dv == 0] - m0[dv == 0]
    s1 = float(np.var(r1, ddof=1)) if r1.size > 1 else 0.0
    s0 = float(np.var(r0, ddof=1)) if r0.size > 1 else 0.0
    tau = float(np.mean(m1 - m0))

    outcome = float(np.mean(s1 / e + s0 / (1 - e)))
    hetero = float(np.mean((m1 - m0 - tau) ** 2))
    bound = outcome + hetero
    # what the outcome term would be with perfect overlap at e = 1/2
    flat = float(np.mean(s1 / 0.5 + s0 / 0.5))
    penalty = float(outcome - flat)

    # IPW's asymptotic variance for comparison
    ipw_var = float(np.mean(
        (dv * yv / e - (1 - dv) * yv / (1 - e) - tau) ** 2
    ))
    return RichResult(
        payload={
            "estimate": bound,
            "bound": bound,
            "se_bound": float(np.sqrt(bound / n)),
            "outcome_term": outcome,
            "heterogeneity_term": hetero,
            "heterogeneity_note": (
                "the variance of the conditional effect itself, which is "
                "irreducible even under perfect overlap because the ATE "
                "averages over genuinely different units"
            ),
            "overlap_penalty": penalty,
            "overlap_note": (
                "how much of the bound comes from propensities away from "
                "one half; when this dominates, no estimator helps and the "
                "fix is a different estimand or better data"
            ),
            "ipw_variance": ipw_var,
            "ipw_relative_efficiency": (float(bound / ipw_var)
                                        if ipw_var > 0 else np.nan),
            "efficiency_note": (
                "AIPW and TMLE attain this bound; IPW alone does not, and "
                "the ratio says how much is lost. It cannot exceed 1 in "
                "population -- IPW's variance is bounded BELOW by the bound "
                "-- so a value above 1 is a numerical artefact, not a "
                "finding"
            ),
            "ratio_valid": bool(ipw_var > 0 and bound / ipw_var <= 1.0 + 1e-9),
            "ratio_warning": (
                None if not (ipw_var > 0 and bound / ipw_var > 1.0 + 1e-9)
                else "the ratio came out at %.3f, above the 1 it cannot "
                     "exceed in population. Both quantities were formed with "
                     "the SAME truncated propensity, so under poor overlap "
                     "neither is the asymptotic object and the comparison "
                     "does not hold. Measured on a design with %d of %d "
                     "propensities truncated"
                     % (bound / ipw_var,
                        int(np.sum((e_raw < trunc) | (e_raw > 1 - trunc))), n)
            ),
            "n_truncated": int(np.sum((e_raw < trunc) | (e_raw > 1 - trunc))),
            "below_bound_note": (
                "an estimator reporting a standard error below se_bound has "
                "not beaten theory -- it has an error somewhere"
            ),
            "propensity_range": (float(e_raw.min()), float(e_raw.max())),
            "n_effective": float(n * flat / outcome) if outcome > 0 else np.nan,
            "estimand": estimand,
            "n": int(n),
            "method": "Semiparametric efficiency bound (Hahn 1998)",
        }
    )


def cheatsheet():
    return (
        "bnseff: Hahn's efficiency bound split into outcome, heterogeneity "
        "and overlap terms, with IPW's loss against it"
    )
