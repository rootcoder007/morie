# morie.fn -- function file (rootcoder007/morie)
"""Martingale residuals for a Cox model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import baseline_hazard

__all__ = ["cox_martingale_residuals"]


def _unpack(fit):
    for k in ("time", "event", "X", "beta"):
        if fit.get(k) is None:
            raise ValueError(
                f"fit is missing {k!r}; pass a result from efron_tie_correction "
                "or breslow_tie_correction"
            )
    return (np.asarray(fit["time"], dtype=float), np.asarray(fit["event"], dtype=float),
            np.atleast_2d(np.asarray(fit["X"], dtype=float)),
            np.asarray(fit["beta"], dtype=float).ravel())


def cox_martingale_residuals(fit):
    r"""Martingale residuals :math:`\hat M_i = \delta_i - \hat\Lambda_i(t_i)`.

    The observed event indicator minus the cumulative hazard actually
    accumulated by that subject. Under a correct model these have mean zero.

    Their shape is deliberately awkward and worth knowing: they are bounded
    above by 1 (a subject cannot have more than one event) but unbounded below,
    so the distribution is severely **left-skewed** and they should never be
    read like ordinary regression residuals. A large negative value means a
    subject survived far longer than the model expected.

    Their canonical use is functional form: plotting martingale residuals from
    a *null* model against a candidate covariate reveals the transformation
    that covariate needs -- the smooth of the plot estimates
    :math:`f(x)` up to a constant.

    Parameters
    ----------
    fit : mapping
        A result from :func:`~morie.fn.efrnt.efron_tie_correction` or
        :func:`~morie.fn.breslot.breslow_tie_correction`.

    Returns
    -------
    RichResult
        ``residuals``, ``expected`` (cumulative hazard per subject),
        ``event``, ``mean``.

    References
    ----------
    Barlow, W. E., & Prentice, R. L. (1988). Residuals for relative risk
        regression. *Biometrika*, 75(1), 65-74.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    Martingale residuals sum to approximately zero under a correct model.

    >>> import numpy as np
    >>> from morie.fn.efrnt import efron_tie_correction
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(2.0, 400)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_martingale_residuals(efron_tie_correction(t, e, X))
    >>> bool(abs(r["mean"]) < 0.02)
    True

    Bounded above by 1, unbounded below -- the left skew that makes them
    unlike ordinary residuals.

    >>> bool(r["residuals"].max() <= 1.0 + 1e-9)
    True
    >>> bool(r["residuals"].min() < -1.0)
    True

    >>> cox_martingale_residuals({"beta": [1.0]})
    Traceback (most recent call last):
        ...
    ValueError: fit is missing 'time'; pass a result from efron_tie_correction or breslow_tie_correction
    """
    t, e, X, beta = _unpack(fit)
    times, dH, H = baseline_hazard(t, e, X, beta)
    w = np.exp(np.clip(X @ beta, -500, 500))
    idx = np.searchsorted(times, t, side="right") - 1
    H_at = np.where(idx >= 0, H[np.clip(idx, 0, max(H.size - 1, 0))], 0.0)
    expected = w * H_at
    resid = e - expected
    return RichResult(
        title="Cox martingale residuals",
        summary_lines=[("n", int(t.size)), ("mean", float(resid.mean())),
                       ("min", float(resid.min()))],
        payload={
            "residuals": resid, "expected": expected, "event": e,
            "mean": float(resid.mean()), "cumhazard": H, "times": times,
            "method": "cox_martingale_residuals",
        },
    )


def cheatsheet():
    return "coxmgr: bounded above by 1, unbounded below -- left-skewed; use vs a covariate for functional form"
