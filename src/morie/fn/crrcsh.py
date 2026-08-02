# morie.fn -- function file (rootcoder007/morie)
"""Cause-specific hazard for competing risks."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["cause_specific_hazard"]


def cause_specific_hazard(time, event_type, X, cause=1, ties="efron"):
    r"""Cox model for one cause, treating competing events as censored.

    .. math::
        \lambda_k(t \mid x) = \lim_{h \to 0}
            \frac{P(t \le T < t+h,\; D = k \mid T \ge t, x)}{h}

    -- the rate of failing from cause :math:`k` among those still event-free.
    Fitted by censoring everyone who fails from a competing cause at their
    failure time.

    This answers an **aetiological** question: does the covariate drive this
    particular mechanism? It does *not* answer the prognostic one. Because
    competing events are censored, the implied "survival curve" is the one that
    would obtain if the competing causes were abolished, which is usually
    counterfactual and always optimistic. Converting a cause-specific hazard
    into a cumulative incidence by :math:`1 - e^{-\Lambda_k}` **overstates
    risk**, sometimes grossly, and is the standard error in this area. Use
    :func:`~morie.fn.crrfgs.competing_risks_fg` when the question is about
    actual incidence.

    Parameters
    ----------
    time : array-like
        Follow-up time.
    event_type : array-like
        0 for censored, otherwise the cause label.
    X : array-like
        Covariates ``(n, p)``.
    cause : int
        The cause of interest.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``z``, ``p_value``, ``hazard_ratio``, ``loglik``,
        ``n_cause``, ``n_competing``, ``n_censored``.

    References
    ----------
    Prentice, R. L., Kalbfleisch, J. D., Peterson, A. V., et al. (1978). The
        analysis of failure times in the presence of competing risks.
        *Biometrics*, 34(4), 541-554.

    Examples
    --------
    A covariate driving cause 1 but not cause 2 shows up in the cause-1 fit
    and not the cause-2 one.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1500, 1))
    >>> T1 = rng.exponential(1 / np.exp(0.9 * X[:, 0]))
    >>> T2 = rng.exponential(1 / np.exp(0.0 * X[:, 0]))
    >>> C = rng.exponential(2.0, 1500)
    >>> T = np.minimum(np.minimum(T1, T2), C)
    >>> d = np.where(T == C, 0, np.where(T1 < T2, 1, 2))
    >>> a = cause_specific_hazard(T, d, X, cause=1)["beta"][0]
    >>> b = cause_specific_hazard(T, d, X, cause=2)["beta"][0]
    >>> bool(abs(a - 0.9) < 0.25 and abs(b) < 0.25)
    True

    Competing events are censored, not dropped, so all subjects contribute.

    >>> r = cause_specific_hazard(T, d, X, cause=1)
    >>> int(r["n_cause"] + r["n_competing"] + r["n_censored"])
    1500

    >>> cause_specific_hazard(T, d, X, cause=9)
    Traceback (most recent call last):
        ...
    ValueError: no events of cause 9 in event_type
    """
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(event_type)).ravel()
    if t.size != d.size:
        raise ValueError(f"time has {t.size} entries but event_type has {d.size}")
    if not np.any(d == cause):
        raise ValueError(f"no events of cause {cause} in event_type")
    e = (d == cause).astype(float)
    _, _, Xm = prepare(t, e, X)
    beta, ll, I, _, it, conv = cox_fit(t, e, Xm, ties=ties)

    from scipy.stats import norm

    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(I)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(beta.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return RichResult(
        title=f"Cause-specific hazard (cause {cause})",
        summary_lines=[("n", int(t.size)), ("events of cause", int(e.sum())),
                       ("competing", int(np.sum((d != 0) & (d != cause)))),
                       ("loglik", ll)],
        warnings=["competing events are censored here, so 1 - exp(-Lambda) "
                  "OVERSTATES incidence; use the Fine-Gray model for actual risk"],
        payload={
            "beta": beta, "se": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
            "hazard_ratio": np.exp(beta), "loglik": ll, "information": I,
            "n_cause": int(e.sum()),
            "n_competing": int(np.sum((d != 0) & (d != cause))),
            "n_censored": int(np.sum(d == 0)), "cause": cause,
            "n": int(t.size), "converged": conv,
            "method": "cause_specific_hazard",
        },
    )


def cheatsheet():
    return "crrcsh: aetiological question; 1-exp(-Lambda) OVERSTATES incidence -- use Fine-Gray for risk"
