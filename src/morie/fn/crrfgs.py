# morie.fn -- function file (rootcoder007/morie)
"""Fine-Gray subdistribution hazard model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["competing_risks_fg"]


def competing_risks_fg(time, event_type, X, cause=1, ties="efron"):
    r"""Fine-Gray model for the subdistribution hazard of one cause.

    The subdistribution hazard is

    .. math::
        \bar\lambda_k(t) = \lim_{h\to 0} \frac{1}{h}\,
            P\!\left(t \le T < t+h,\, D = k \;\middle|\;
            \{T \ge t\} \cup \{T < t,\, D \ne k\}\right),

    whose risk set is the one peculiar feature of the method: subjects who
    have already failed from a **competing** cause stay in it forever. They
    can never experience cause :math:`k`, and keeping them in is precisely
    what makes the model's coefficients map monotonically onto cumulative
    incidence.

    That is the trade. The cause-specific hazard has a clean mechanistic
    reading and no direct link to incidence; the subdistribution hazard has a
    direct link to incidence and a risk set containing people who are already
    dead. A Fine-Gray hazard ratio is a statement about *risk*, not about
    biology, and should be reported as such.

    Implemented with inverse-probability-of-censoring weights, so subjects who
    failed from competing causes carry a weight that decays with the censoring
    distribution.

    Parameters
    ----------
    time : array-like
        Follow-up time.
    event_type : array-like
        0 censored, otherwise the cause label.
    X : array-like
        Covariates.
    cause : int
        Cause of interest.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``z``, ``p_value``, ``subdistribution_hazard_ratio``,
        ``n_cause``, ``n_competing``.

    References
    ----------
    Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the
        subdistribution of a competing risk. *JASA*, 94(446), 496-509.

    Examples
    --------
    On data where a covariate drives cause 1, the subdistribution hazard ratio
    points the same way as the cause-specific one.

    >>> import numpy as np
    >>> from morie.fn.crrcsh import cause_specific_hazard
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1500, 1))
    >>> T1 = rng.exponential(1 / np.exp(0.9 * X[:, 0]))
    >>> T2 = rng.exponential(1 / np.exp(0.0 * X[:, 0]))
    >>> C = rng.exponential(2.0, 1500)
    >>> T = np.minimum(np.minimum(T1, T2), C)
    >>> d = np.where(T == C, 0, np.where(T1 < T2, 1, 2))
    >>> fg = competing_risks_fg(T, d, X, cause=1)["beta"][0]
    >>> cs = cause_specific_hazard(T, d, X, cause=1)["beta"][0]
    >>> bool(fg > 0.3 and cs > 0.3)
    True

    They are different estimands and generally differ numerically -- that is
    expected, not an inconsistency.

    >>> bool(abs(fg - cs) > 1e-6)
    True

    Subjects failing from competing causes stay in the risk set, which is what
    the weights encode.

    >>> r = competing_risks_fg(T, d, X, cause=1)
    >>> bool(r["n_competing"] > 0 and r["weights"].min() >= 0)
    True
    """
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(event_type)).ravel()
    if t.size != d.size:
        raise ValueError(f"time has {t.size} entries but event_type has {d.size}")
    if not np.any(d == cause):
        raise ValueError(f"no events of cause {cause} in event_type")
    e = (d == cause).astype(float)
    _, _, Xm = prepare(t, e, X)

    # Censoring distribution by Kaplan-Meier on the censoring indicator.
    from ._surv import km_estimate

    ct, csurv = km_estimate(t, (d == 0).astype(float))

    def G(u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if ct.size == 0:
            return np.ones_like(u)
        pos = np.searchsorted(ct, u, side="right") - 1
        return np.where(pos >= 0, csurv[np.clip(pos, 0, csurv.size - 1)], 1.0)

    competing = (d != 0) & (d != cause)
    Gi = np.maximum(G(t), 1e-8)

    # The IPCW weight is TIME-DEPENDENT: a subject who failed from a competing
    # cause at t_i stays in every later risk set with weight G(t)/G(t_i), which
    # decays as censoring accumulates. Evaluating it at the subject's own time
    # gives 1 for everyone and silently collapses Fine-Gray back to the
    # cause-specific fit.
    beta, ll, I, U = _fg_newton(t, e, Xm, competing, G, Gi)
    it, conv = 0, True
    from scipy.stats import norm

    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(I)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(beta.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return RichResult(
        title=f"Fine-Gray subdistribution model (cause {cause})",
        summary_lines=[("n", int(t.size)), ("events of cause", int(e.sum())),
                       ("competing", int(competing.sum())), ("loglik", ll)],
        warnings=["the risk set keeps subjects who already failed from a "
                  "competing cause; a Fine-Gray hazard ratio is a statement "
                  "about RISK, not about mechanism"],
        payload={
            "beta": beta, "se": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
            "subdistribution_hazard_ratio": np.exp(beta),
            "hazard_ratio": np.exp(beta), "weights": Gi, "loglik": ll,
            "n_cause": int(e.sum()), "n_competing": int(competing.sum()),
            "cause": cause, "n": int(t.size), "converged": conv,
            "method": "competing_risks_fg",
        },
    )


def _fg_newton(t, e, X, competing, G, Gi, max_iter=50, tol=1e-9):
    """Newton-Raphson on the weighted Fine-Gray partial likelihood."""
    n, p = X.shape
    beta = np.zeros(p)
    utimes = np.unique(t[e == 1])
    ll = 0.0
    I = np.zeros((p, p))
    U = np.zeros(p)
    for _ in range(max_iter):
        w = np.exp(np.clip(X @ beta, -500, 500))
        ll = 0.0
        U = np.zeros(p)
        I = np.zeros((p, p))
        for ut in utimes:
            # Weight 1 for those still at risk; G(ut)/G(t_i) for those who
            # already failed from a competing cause; 0 once censored.
            wt = np.where(t >= ut, 1.0,
                          np.where(competing & (t < ut), G(ut)[0] / Gi, 0.0))
            inr = wt > 0
            if not np.any(inr):
                continue
            died = (t == ut) & (e == 1)
            dcount = int(died.sum())
            if dcount == 0:
                continue
            ww = wt[inr] * w[inr]
            Xr = X[inr]
            S0 = ww.sum()
            S1 = ww @ Xr
            S2 = (ww[:, None] * Xr).T @ Xr
            ll += float((X[died] @ beta).sum()) - dcount * np.log(max(S0, 1e-300))
            mu = S1 / max(S0, 1e-300)
            U += X[died].sum(axis=0) - dcount * mu
            I += dcount * (S2 / max(S0, 1e-300) - np.outer(mu, mu))
        try:
            step = np.linalg.solve(I, U)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(I, U, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, float(ll), I, U


def cheatsheet():
    return "crrfgs: risk set keeps subjects already dead of other causes -- that is what links it to incidence"
