# morie.fn -- function file (rootcoder007/morie)
"""Value function of a treatment regime."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict, ols_fit

__all__ = ["regime_value", "value_function_eval"]


def regime_value(y, d, X, regime, propensity=None, method="aipw",
                 trunc=0.01):
    r"""Expected outcome if everyone followed a given decision rule.

    .. math::
       V(d) = \mathbb{E}\!\left[
         \frac{\mathbb{1}\{A = d(X)\}}{\pi(A \mid X)} Y\right]
       \;=\;
       \mathbb{E}\big[\mu_{d(X)}(X)\big]

    the first form by weighting, the second by regression, and the
    augmented estimator combining them.

    A regime is a FUNCTION of covariates, so its value is a different
    object from an average treatment effect: it answers what would
    happen under a policy rather than under a uniform intervention.
    The optimal regime's value is bounded below by the better of the
    two static policies, and ``beats_static`` checks that -- a fitted
    regime scoring worse than "treat everyone" has not learned
    anything and should not be deployed.

    The estimate is biased upward when the regime was FITTED on the
    same data, for the same reason a training error understates test
    error: the rule is chosen to look good on these units. That bias
    is exactly what makes reported values of learned policies
    optimistic, and ``fitted_on_same_data`` flags the risk since this
    function cannot tell from its inputs.

    Non-regularity is the deeper problem. When the treatment effect is
    near zero for a subset of covariates, the optimal rule is
    ill-defined there and the value function is non-differentiable, so
    standard confidence intervals undercover. ``near_indifferent``
    reports the share of units whose estimated treatment effect lies
    within one tenth of a standard deviation of zero -- a heuristic for
    how much mass sits near the decision boundary, where the optimal
    rule flips. It is a scale-of-the-effect measure, NOT a per-unit
    standard-error test, and does not shrink with n.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
        Treatment actually received.
    X : array-like, shape (n, p)
    regime : array-like of {0, 1} or callable
        The rule's recommendation per unit.
    propensity : array-like, optional
    method : {'aipw', 'ipw', 'regression'}
    trunc : float

    Returns
    -------
    RichResult
        ``value``, ``se``, ``ci``, ``value_treat_all``,
        ``value_treat_none``, ``beats_static``, ``n_following``,
        ``near_indifferent``.

    References
    ----------
    Zhang, B., Tsiatis, A. A., Davidian, M., Zhang, M., & Laber, E.
    (2012). Estimating optimal treatment regimes from a classification
    perspective. *Stat*, 1(1), 103-114.
    Murphy (2003), *JRSS-B* 65:331-355.
    Luedtke and van der Laan (2016) on non-regularity.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 1))
    >>> d = (rng.uniform(size=600) < 0.5).astype(float)
    >>> y = d * X[:, 0] + rng.normal(size=600)
    >>> out = regime_value(y, d, X, (X[:, 0] > 0).astype(float))
    >>> bool(out["value"] > out["value_treat_none"])
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if dv.size != n or Xa.shape[0] != n:
        raise ValueError("y, d and X must agree in their first dimension.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if method not in ("aipw", "ipw", "regression"):
        raise ValueError(
            "method must be aipw, ipw or regression, got %r." % method
        )
    g = (np.asarray(regime(Xa), dtype=float).ravel() if callable(regime)
         else np.asarray(regime, dtype=float).ravel())
    if g.size != n:
        raise ValueError("regime has %d entries for %d rows." % (g.size, n))
    if not np.all(np.isin(g, (0.0, 1.0))):
        raise ValueError("regime must recommend 0 or 1.")

    B = add_intercept(Xa)
    if propensity is None:
        e = logit_predict(B, logit_fit(B, dv)[0])
    else:
        e = np.asarray(propensity, dtype=float).ravel()
        if e.size != n:
            raise ValueError("propensity has %d entries for %d rows." % (e.size, n))
    e = np.clip(e, trunc, 1 - trunc)
    pi = np.where(dv == 1, e, 1 - e)

    mu1 = B @ ols_fit(B[dv == 1], yv[dv == 1])
    mu0 = B @ ols_fit(B[dv == 0], yv[dv == 0])

    def value(rule):
        follow = (dv == rule).astype(float)
        mu_d = np.where(rule == 1, mu1, mu0)
        if method == "ipw":
            psi = follow / pi * yv
        elif method == "regression":
            psi = mu_d
        else:
            psi = mu_d + follow / pi * (yv - mu_d)
        return psi

    psi = value(g)
    v = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    v_all = float(np.mean(value(np.ones(n))))
    v_none = float(np.mean(value(np.zeros(n))))

    tau = mu1 - mu0
    tau_se = float(np.std(tau, ddof=1)) / np.sqrt(n) if n > 1 else np.nan
    near = float(np.mean(np.abs(tau) < max(tau_se, 1e-12) * np.sqrt(n) * 0.1))
    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": v,
            "value": v,
            "se": se,
            "ci": (v - z * se, v + z * se),
            "value_treat_all": v_all,
            "value_treat_none": v_none,
            "best_static": max(v_all, v_none),
            "beats_static": bool(v >= max(v_all, v_none) - 1e-12),
            "static_note": (
                "the optimal regime's value is bounded below by the better "
                "static policy; a fitted rule scoring worse has learned "
                "nothing and should not be deployed"
            ),
            "gain_over_static": float(v - max(v_all, v_none)),
            "n_following": int(np.sum(dv == g)),
            "following_fraction": float(np.mean(dv == g)),
            "near_indifferent": near,
            "regularity_note": (
                "where the conditional effect is near zero the optimal rule "
                "is ill-defined and the value function is "
                "non-differentiable, so standard intervals undercover; this "
                "is the share of units within 0.1 sd of the boundary"
            ),
            "fitted_on_same_data": None,
            "optimism_note": (
                "if the regime was fitted on these same units the value is "
                "biased upward for the same reason a training error "
                "understates test error; this function cannot tell from its "
                "inputs, so evaluate a learned rule on held-out data"
            ),
            "method_used": method,
            "propensity_range": (float(e.min()), float(e.max())),
            "n": int(n),
            "method": "Value of a treatment regime (%s)" % method.upper(),
        }
    )


def cheatsheet():
    return (
        "vlfctn: regime value against the static policies, with the "
        "non-regularity share and the same-data optimism warning"
    )


#: Catalogue alias for :func:`regime_value`.
value_function_eval = regime_value
