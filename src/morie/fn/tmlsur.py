# morie.fn -- function file (rootcoder007/morie)
"""Targeted maximum likelihood estimation for right-censored survival."""

from . import _array_core as np

from ._richresult import RichResult
from ._survtmle import survival_tmle

__all__ = ["tmle_survival"]


def tmle_survival(time, event, treatment, covariates, tau=None, n_bins=None,
                  trunc=0.025, max_iter=100):
    r"""Treatment-specific survival at a horizon, by hazard-based TMLE.

    The estimand is :math:`\Psi_a(t_0) = E_W[S(t_0 \mid A = a, W)]`, the
    survival probability that WOULD have been observed had everyone been
    assigned arm :math:`a`, and the reported ``estimate`` is
    :math:`\Psi_1(t_0) - \Psi_0(t_0)`.

    Working on the discrete hazard scale rather than on the survival
    scale directly is what makes the censoring adjustment tractable.
    The likelihood factorises into the confounder distribution, the
    propensity score, the failure hazard and the censoring hazard, and
    only the failure hazard is fluctuated -- the other three are tangent
    to the parameter, so touching them would buy nothing.

    The clever covariate is

    .. math::
       h_{t_0}(k, A, W) = -\,
       \frac{I(A = a)\, I(k \le t_0)}
            {g(a \mid W)\, S_{A^c}(k^- \mid A, W)}\,
       \frac{S_N(t_0 \mid A, W)}{S_N(k \mid A, W)},

    and the hazard is updated along
    :math:`\mathrm{logit}\,\lambda^{*} = \mathrm{logit}\,\lambda +
    \epsilon h`, with :math:`\epsilon` fitted by a no-intercept logistic
    regression of the failure indicator on :math:`h` carrying the
    initial hazard as an offset.

    Three details are not optional and are reported rather than assumed.
    First, the targeting is ITERATED: one fluctuation does not in
    general solve the efficient influence curve equation, and the loop
    here continues until :math:`|P_n D^{*}| \le 1/n`. ``converged``
    says whether it got there. Second, :math:`S_{A^c}(k^-)` is the
    censoring survival strictly BEFORE :math:`k`; using
    :math:`S_{A^c}(k)` instead divides by a quantity that already
    reflects the censoring at :math:`k` and biases every weight.
    Third, because this is a substitution estimator, the survival
    probabilities cannot leave :math:`[0, 1]` however badly the
    nuisance fits behave -- which an estimating-equation correction
    built on the same influence curve can and does.

    Positivity is the failure mode. ``n_truncated`` and
    ``propensity_range`` show how much of the sample sat near the
    boundary; heavy truncation there means the difference is being
    extrapolated rather than estimated.

    Parameters
    ----------
    time : array-like, shape (n,)
        Observed follow-up time, :math:`\tilde T = \min(T, C)`.
    event : array-like of {0, 1}, shape (n,)
        1 if the failure was observed, 0 if right-censored.
    treatment : array-like of {0, 1}, shape (n,)
        Treatment arm.
    covariates : array-like, shape (n, p) or (n,)
        Baseline confounders.
    tau : float, optional
        Horizon :math:`t_0` on the original time scale. The last time
        bin by default.
    n_bins : int, optional
        Number of bins for the discrete time grid. When the observed
        times take 50 or fewer distinct values they ARE the grid and
        nothing is coarsened.
    trunc : float
        Propensity truncation bound.
    max_iter : int
        Cap on targeting iterations per arm.

    Returns
    -------
    RichResult
        ``estimate`` (:math:`\Psi_1 - \Psi_0`), ``se``, ``ci``,
        ``s1``, ``s0`` with their own standard errors, ``curve1`` and
        ``curve0`` (the whole targeted survival curves), ``eif_mean``,
        ``epsilon``, ``iterations``, ``converged``, ``horizon``,
        ``n_truncated``, ``propensity_range``, ``n_events``,
        ``n_censored``.

    References
    ----------
    Cai and van der Laan (2020), "One-step targeted maximum likelihood
    estimation for time-to-event outcomes", *Biometrics* 76:722-733.
    Preprint arXiv:1802.09479; their equations (1)-(3) and (11) are
    what is implemented here.
    Hubbard, van der Laan and Robins (2000), in *Statistical Models in
    Epidemiology, the Environment, and Clinical Trials*, Springer,
    pp. 135-177 -- the efficient influence curve, reproduced as
    equation (2) of Cai and van der Laan.
    Moore and van der Laan (2009), *Journal of Biopharmaceutical
    Statistics* 19:1099-1131, on covariate adjustment with
    right-censored outcomes. Their 2009 *Statistics in Medicine*
    28:39-64 paper is sometimes cited for this estimator by mistake; it
    is about BINARY outcomes.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> n = 400
    >>> W = rng.normal(size=(n, 1))
    >>> A = (rng.uniform(size=n) < 0.5).astype(float)
    >>> T = rng.integers(1, 9, size=n) + 3 * A
    >>> C = rng.integers(2, 12, size=n)
    >>> out = tmle_survival(np.minimum(T, C), (T <= C).astype(int), A, W)
    >>> bool(out["s1"] > out["s0"])
    True
    """
    res = survival_tmle(
        time, event, treatment, covariates, t0=tau, n_bins=n_bins,
        trunc=trunc, max_iter=max_iter,
    )
    z = 1.959963984540054
    est, se = res["estimate"], res["se"]
    g = res["propensity"]
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - z * se, est + z * se),
            "ci_lower": est - z * se,
            "ci_upper": est + z * se,
            "s1": res["s1"],
            "s0": res["s0"],
            "se_s1": res["se_s1"],
            "se_s0": res["se_s0"],
            "curve1": res["curve1"],
            "curve0": res["curve0"],
            "time_grid": res["time_grid"],
            "horizon": res["horizon"],
            "eif_mean": res["eif_mean"],
            "eif_note": (
                "the targeting step drives the empirical mean of the "
                "efficient influence curve to zero; this is the check that "
                "it converged, not a formality"
            ),
            "epsilon": res["epsilon"],
            "iterations": res["iterations"],
            "converged": res["converged"],
            "convergence_note": (
                None if res["converged"] else
                "targeting stopped at the iteration cap with |P_n D*| still "
                "above 1/n; the standard error is not trustworthy here"
            ),
            "propensity": g,
            "propensity_range": (float(g.min()), float(g.max())),
            "n_truncated": res["n_truncated"],
            "positivity_note": (
                "propensities were truncated for %d of %d subjects; a large "
                "share means the contrast is extrapolated, not estimated"
                % (res["n_truncated"], res["n"])
            ),
            "separated": res["separated"],
            "n_events": res["n_events"],
            "n_censored": res["n_censored"],
            "n_bins": res["tmax"],
            "n": res["n"],
            "method": (
                "Hazard-based TMLE of the treatment-specific survival "
                "difference"
            ),
        }
    )


def cheatsheet():
    return (
        "tmlsur: survival TMLE on the discrete hazard scale -- iterated "
        "targeting along the Cai-van der Laan clever covariate, with the "
        "influence-curve convergence check reported"
    )
