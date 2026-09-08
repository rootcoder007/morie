# morie.fn -- function file (rootcoder007/morie)
"""Fine-Gray subdistribution hazard, alternative front-end."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .crrfgs import competing_risks_fg

__all__ = ["fine_gray_subdistribution_hazard"]


def fine_gray_subdistribution_hazard(time, cause, X, of_cause=1, **kwargs):
    r"""Subdistribution hazard with the cumulative incidence it implies.

    Wraps :func:`~morie.fn.crrfgs.competing_risks_fg` and adds the object the
    model exists to produce: the **cumulative incidence function**

    .. math::
        F_k(t \mid x) = 1 - \exp\!\left(-\bar\Lambda_k(t)\, e^{\beta^\top x}\right),

    which is a genuine probability of having failed from cause :math:`k` by
    time :math:`t`, accounting for the competing causes.

    The corresponding formula applied to a *cause-specific* hazard is not a
    probability and can exceed the truth substantially. The whole reason the
    subdistribution hazard is worth its strange risk set is that this
    transformation is valid for it.

    Cumulative incidence is bounded above by the probability of any event, so
    the curves for all causes sum to the all-cause failure probability -- a
    property worth checking against.

    Parameters
    ----------
    time : array-like
        Follow-up time.
    cause : array-like
        0 censored, otherwise cause label.
    X : array-like
        Covariates.
    of_cause : int
        Cause of interest.
    **kwargs
        Passed through.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``times``, ``cumulative_incidence``,
        ``baseline_cif``.

    References
    ----------
    Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for
        the subdistribution of a competing risk. *Journal of the American
        Statistical Association*, 94(446), 496-509.

    Examples
    --------
    Cumulative incidence is a probability: bounded in [0, 1] and
    non-decreasing.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1200, 1))
    >>> T1 = rng.exponential(1 / np.exp(0.8 * X[:, 0]))
    >>> T2 = rng.exponential(1.0, 1200)
    >>> C = rng.exponential(2.0, 1200)
    >>> T = np.minimum(np.minimum(T1, T2), C)
    >>> d = np.where(T == C, 0, np.where(T1 < T2, 1, 2))
    >>> r = fine_gray_subdistribution_hazard(T, d, X, of_cause=1)
    >>> cif = r["baseline_cif"]
    >>> bool(cif.min() >= 0 and cif.max() <= 1 and np.all(np.diff(cif) >= -1e-12))
    True

    It stays strictly below 1, since competing causes claim some subjects.

    >>> bool(cif.max() < 1.0)
    True
    """
    fit = competing_risks_fg(time, cause, X, cause=of_cause, **kwargs)
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(cause)).ravel()
    e = (d == of_cause).astype(float)
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != t.size:
        Xm = Xm.T

    from ._surv import baseline_hazard

    times, dH, H = baseline_hazard(t, e, Xm, fit["beta"],
                                   offset=np.log(np.maximum(fit["weights"], 1e-12)))
    base_cif = 1.0 - np.exp(-H)
    lin = np.exp(np.clip(Xm @ fit["beta"], -500, 500))
    cif = 1.0 - np.exp(-np.outer(lin, H))
    return RichResult(
        title=f"Fine-Gray cumulative incidence (cause {of_cause})",
        summary_lines=[("n", int(t.size)), ("event times", int(times.size)),
                       ("max baseline CIF", float(base_cif[-1]) if base_cif.size else float("nan"))],
        warnings=list(fit.warnings),
        payload={
            "beta": fit["beta"], "se": fit["se"], "p_value": fit["p_value"],
            "subdistribution_hazard_ratio": fit["subdistribution_hazard_ratio"],
            "times": times, "baseline_cif": base_cif,
            "cumulative_incidence": cif, "cumhazard": H,
            "cause": of_cause, "n": int(t.size),
            "method": "fine_gray_subdistribution_hazard",
        },
    )


def cheatsheet():
    return "fgsbh: 1-exp(-Lambda) IS a probability here, unlike for cause-specific hazards"
