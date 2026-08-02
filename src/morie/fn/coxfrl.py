# morie.fn -- function file (rootcoder007/morie)
"""Shared-frailty Cox model."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["cox_frailty"]


def cox_frailty(time, event, X, cluster, theta=None, max_iter=30, tol=1e-6,
                ties="efron"):
    r"""Cox model with a shared gamma frailty per cluster.

    Each cluster :math:`k` carries a latent multiplier :math:`w_k` on its
    hazard,

    .. math::
        \lambda_{ik}(t) = w_k \, \lambda_0(t)\, e^{\beta^\top x_i},
        \qquad w_k \sim \mathrm{Gamma}(1/\theta, 1/\theta),

    with mean 1 so that :math:`\theta` alone carries the between-cluster
    variance. Fitted by penalised partial likelihood: the frailties enter as
    an offset :math:`\log w_k`, are updated in closed form from the gamma
    conjugacy, and the two steps alternate.

    Frailty is what makes correlated survival data honest. Siblings, repeat
    admissions of the same patient, or subjects within a hospital are not
    independent, and ignoring that leaves the coefficients roughly right but
    the standard errors **too small** -- the usual consequence of pretending
    clustered data is i.i.d.

    :math:`\theta = 0` recovers the ordinary Cox model. Larger
    :math:`\theta` means more heterogeneity between clusters; the Kendall's
    tau induced within a cluster is :math:`\theta/(\theta+2)`, which is
    returned as an interpretable scale.

    Unshared frailty -- one cluster per subject -- is not identifiable and is
    rejected rather than silently fitted.

    Parameters
    ----------
    time, event, X : array-like
        Survival data.
    cluster : array-like
        Cluster label per subject.
    theta : float, optional
        Fix the frailty variance instead of estimating it.
    max_iter, tol
        Outer-loop controls.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``theta``, ``kendall_tau``, ``frailty``,
        ``n_clusters``, ``loglik``, ``converged``.

    References
    ----------
    Clayton, D. G. (1978). A model for association in bivariate life tables.
        *Biometrika*, 65(1), 141-151.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    With genuine cluster heterogeneity the frailty variance is estimated well
    above zero, and the coefficient is still recovered.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> k = np.repeat(np.arange(60), 10)
    >>> w = rng.gamma(2.0, 0.5, 60)[k]              # cluster multipliers
    >>> X = rng.normal(size=(600, 1))
    >>> T = rng.exponential(1 / (w * np.exp(X[:, 0] * 0.8)))
    >>> C = rng.exponential(2.0, 600)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_frailty(t, e, X, k)
    >>> bool(r["theta"] > 0.1)
    True
    >>> bool(abs(r["beta"][0] - 0.8) < 0.3)
    True

    Frailties are centred at 1 by construction, so they multiply the baseline
    rather than shifting it.

    >>> bool(abs(float(np.mean(r["frailty"])) - 1.0) < 0.15)
    True

    Ignoring the clustering understates the standard error, which is the whole
    reason to fit this.

    >>> from morie.fn.efrnt import efron_tie_correction
    >>> bool(r["se"][0] > efron_tie_correction(t, e, X)["se"][0])
    True

    One cluster per subject carries no shared information and is refused.

    >>> cox_frailty(t, e, X, np.arange(600))
    Traceback (most recent call last):
        ...
    ValueError: every cluster has one member, so a shared frailty is not identifiable
    """
    t, e, Xm = prepare(time, event, X)
    cl = np.asarray(cluster).ravel()
    if cl.size != t.size:
        raise ValueError(f"cluster has {cl.size} entries but time has {t.size}")
    levels, idx = np.unique(cl, return_inverse=True)
    K = levels.size
    if K == t.size:
        raise ValueError(
            "every cluster has one member, so a shared frailty is not identifiable"
        )

    from ._sci_core import minimize_scalar
    from ._sci_core import gammaln

    from ._surv import baseline_hazard

    def _inner(th_val):
        """Alternate Cox fit and closed-form frailty update at fixed theta."""
        logw = np.zeros(t.size)
        beta_l = np.zeros(Xm.shape[1])
        frail_l = np.ones(K)
        conv_l = False
        ll_l = -np.inf
        it_l = 0
        for it_l in range(1, max_iter + 1):
            beta_new, ll_l, _, _, _, _ = cox_fit(t, e, Xm, ties=ties, offset=logw)
            times, _, Hc = baseline_hazard(t, e, Xm, beta_new, offset=logw)
            pos = np.searchsorted(times, t, side="right") - 1
            H_at = np.where(pos >= 0, Hc[np.clip(pos, 0, max(Hc.size - 1, 0))], 0.0)
            risk = np.exp(np.clip(Xm @ beta_new, -500, 500)) * H_at
            d_k = np.bincount(idx, weights=e, minlength=K)
            r_k = np.bincount(idx, weights=risk, minlength=K)
            frail_new = (1.0 / th_val + d_k) / (1.0 / th_val + r_k)
            delta = max(float(np.max(np.abs(beta_new - beta_l))),
                        float(np.max(np.abs(frail_new - frail_l))))
            beta_l, frail_l = beta_new, frail_new
            logw = np.log(np.maximum(frail_l[idx], 1e-12))
            if delta < tol:
                conv_l = True
                break
        return beta_l, frail_l, d_k, r_k, ll_l, it_l, conv_l

    def _marginal(th_val):
        """Closed-form gamma-frailty marginal log-likelihood, up to a constant.

        The posterior means of w are shrunk toward 1, so their sample variance
        is a badly biased estimator of theta -- it collapses to zero and takes
        the frailty with it. Profiling the marginal likelihood is what actually
        identifies theta.
        """
        _, _, d_k, r_k, ll_l, _, _ = _inner(th_val)
        a = 1.0 / th_val
        return float(np.sum(
            gammaln(a + d_k) - gammaln(a) - (a + d_k) * np.log1p(th_val * r_k)
            + d_k * np.log(th_val)
        ) + ll_l)

    if theta is None:
        opt = minimize_scalar(lambda lg: -_marginal(np.exp(lg)),
                              bounds=(np.log(1e-4), np.log(10.0)),
                              method="bounded", options={"xatol": 1e-3})
        th = float(np.exp(opt.x))
    else:
        th = float(theta)
    beta, frail, d_k, r_k, ll, it, converged = _inner(th)
    logw = np.log(np.maximum(frail[idx], 1e-12))

    from ._stats_core import norm

    _, _, I, _, _, _ = cox_fit(t, e, Xm, ties=ties, offset=logw, max_iter=1)
    # Cluster-robust variance: the frailty inflates uncertainty relative to an
    # i.i.d. fit, which is the point of fitting it.
    try:
        Iinv = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        Iinv = np.linalg.pinv(I)
    scale = 1.0 + th * float(np.mean(np.bincount(idx, minlength=K)) - 1.0)
    se = np.sqrt(np.clip(np.diag(Iinv) * max(scale, 1.0), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return RichResult(
        title="Shared-frailty Cox model",
        summary_lines=[("n", int(t.size)), ("clusters", int(K)),
                       ("theta", float(th)), ("loglik", ll)],
        warnings=[] if converged else ["the frailty loop did not converge"],
        payload={
            "beta": beta, "se": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
            "hazard_ratio": np.exp(beta), "theta": float(th),
            "kendall_tau": float(th / (th + 2.0)),
            "frailty": frail, "clusters": levels, "n_clusters": int(K),
            "loglik": ll, "n": int(t.size), "n_iter": it,
            "converged": converged, "method": "cox_frailty",
        },
    )


def cheatsheet():
    return "coxfrl: gamma frailty per cluster; ignoring clustering leaves SEs too SMALL. tau = theta/(theta+2)"
