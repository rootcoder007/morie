# morie.fn -- function file (rootcoder007/morie)
"""Cause-specific hazards for every cause at once."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .crrcsh import cause_specific_hazard

__all__ = ["cause_specific_hazard_all"]


def cause_specific_hazard_all(time, cause, X, ties="efron"):
    r"""Fit a cause-specific Cox model for each observed cause.

    Convenience over :func:`~morie.fn.crrcsh.cause_specific_hazard`, returning
    one fit per cause so the mechanisms can be compared side by side.

    The set of cause-specific hazards is a **complete** description of a
    competing-risks process: all-cause hazard is their sum, and cumulative
    incidence for any cause is recoverable from all of them together. That
    completeness is what a single Fine-Gray subdistribution fit gives up in
    exchange for interpretability -- so fitting every cause here and reading
    them jointly is the more informative default when the number of causes is
    small.

    A covariate can raise the cause-specific hazard for one cause and still
    *lower* its cumulative incidence, if it raises a competing hazard more.
    That is not a contradiction; it is why both views exist.

    Parameters
    ----------
    time : array-like
        Follow-up time.
    cause : array-like
        0 for censored, otherwise the cause label.
    X : array-like
        Covariates.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``causes``, ``beta`` ``(n_causes, p)``, ``se``, ``hazard_ratio``,
        ``n_events``, ``fits``.

    References
    ----------
    Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics:
        Competing risks and multi-state models. *Statistics in Medicine*,
        26(11), 2389-2430.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1500, 1))
    >>> T1 = rng.exponential(1 / np.exp(0.9 * X[:, 0]))
    >>> T2 = rng.exponential(1 / np.exp(-0.6 * X[:, 0]))
    >>> C = rng.exponential(2.0, 1500)
    >>> T = np.minimum(np.minimum(T1, T2), C)
    >>> d = np.where(T == C, 0, np.where(T1 < T2, 1, 2))
    >>> r = cause_specific_hazard_all(T, d, X)
    >>> [int(c) for c in r["causes"]]
    [1, 2]

    Opposite effects on the two mechanisms are recovered separately.

    >>> bool(r["beta"][0, 0] > 0.5 and r["beta"][1, 0] < -0.3)
    True

    Event counts across causes and censoring account for everyone.

    >>> int(r["n_events"].sum() + int(np.sum(d == 0)))
    1500
    """
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(cause)).ravel()
    causes = np.array([c for c in np.unique(d) if c != 0])
    if causes.size == 0:
        raise ValueError("no events: every entry of cause is 0")
    fits = [cause_specific_hazard(t, d, X, cause=c, ties=ties) for c in causes]
    beta = np.vstack([f["beta"] for f in fits])
    se = np.vstack([f["se"] for f in fits])
    return RichResult(
        title="Cause-specific hazards (all causes)",
        summary_lines=[("n", int(t.size)), ("causes", int(causes.size))],
        warnings=["a covariate can raise a cause-specific hazard while "
                  "lowering that cause's cumulative incidence, if it raises a "
                  "competing hazard more"],
        payload={
            "causes": causes, "beta": beta, "se": se,
            "hazard_ratio": np.exp(beta),
            "n_events": np.array([f["n_cause"] for f in fits]),
            "loglik": np.array([f["loglik"] for f in fits]),
            "fits": fits, "n": int(t.size),
            "method": "cause_specific_hazard_all",
        },
    )


def cheatsheet():
    return "csrh: all causes together are a COMPLETE description; Fine-Gray trades that for interpretability"
