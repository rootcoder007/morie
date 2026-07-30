# morie.fn -- function file (rootcoder007/morie)
"""Inverse-probability-of-censoring weights."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["censoring_at_risk_weight"]


def censoring_at_risk_weight(time, censor, at=None, stabilize=True):
    r"""Weights that restore the population a censored sample would have been.

    .. math::
        w_i(t) = \\frac{\\mathbb{1}\\{C_i > t\\}}{\\hat G(t \\mid x_i)},

    where :math:`\\hat G` is the censoring survivor function. Each uncensored
    subject stands in for those like them who were censored, so an analysis on
    the weighted sample estimates what a complete-data analysis would have.

    IPCW is what makes complete-case analysis honest under **informative**
    censoring. If censoring depends on covariates -- sicker patients leaving
    the study earlier -- then dropping censored subjects biases everything,
    and no amount of care with the remainder repairs it.

    The weights inherit propensity weighting's pathology: as
    :math:`\\hat G \\to 0` at long follow-up, weights explode and a few
    late survivors dominate. ``max_weight_share`` is reported for that
    reason, and administrative censoring at the study end should be handled by
    truncating the time axis rather than by weighting.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    censor : array-like
        1 if censored, 0 if the event was observed.
    at : float, optional
        Time at which to evaluate the weights. Defaults to each subject's own
        time.
    stabilize : bool
        Multiply by the marginal censoring survivor to stabilise the weights.

    Returns
    -------
    RichResult
        ``weights``, ``G``, ``max_weight_share``, ``ess``,
        ``n_censored``.

    References
    ----------
    Robins, J. M., & Finkelstein, D. M. (2000). Correcting for noncompliance
        and dependent censoring in an AIDS clinical trial with IPCW log-rank
        tests. *Biometrics*, 56(3), 779-788.

    Examples
    --------
    Censored subjects get zero weight and the uncensored are up-weighted to
    stand in for them.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = rng.exponential(2.0, 500)
    >>> C = rng.exponential(3.0, 500)
    >>> t = np.minimum(T, C)
    >>> cen = (C < T).astype(float)
    >>> r = censoring_at_risk_weight(t, cen)
    >>> bool(np.all(r["weights"][cen == 1] == 0))
    True
    >>> bool(r["weights"][cen == 0].min() > 0)
    True

    Weights grow with follow-up time, since fewer subjects survive censoring
    to stand in for the rest.

    >>> obs = np.flatnonzero(cen == 0)
    >>> late = obs[np.argmax(t[obs])]
    >>> early = obs[np.argmin(t[obs])]
    >>> bool(r["weights"][late] > r["weights"][early])
    True

    >>> censoring_at_risk_weight([1.0, 2.0], [0.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: censor must be 0/1
    """
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    c = np.atleast_1d(np.asarray(censor, dtype=float)).ravel()
    if t.size != c.size:
        raise ValueError(f"time has {t.size} entries but censor has {c.size}")
    if not np.all((c == 0) | (c == 1)):
        raise ValueError("censor must be 0/1")
    from ._surv import km_estimate

    ct, csurv = km_estimate(t, c)

    def G(u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if ct.size == 0:
            return np.ones_like(u)
        pos = np.searchsorted(ct, u, side="right") - 1
        return np.where(pos >= 0, csurv[np.clip(pos, 0, csurv.size - 1)], 1.0)

    eval_t = t if at is None else np.full(t.size, float(at))
    g = np.maximum(G(eval_t), 1e-8)
    w = np.where(c == 0, 1.0 / g, 0.0)
    if stabilize:
        w = w * float(np.mean(G(eval_t)))
    tot = float(w.sum())
    share = float(w.max() / tot) if tot > 0 else float("nan")
    ess = float(tot**2 / max(float(np.sum(w**2)), 1e-300))
    return RichResult(
        title="IPCW weights",
        summary_lines=[("n", int(t.size)), ("censored", int(c.sum())),
                       ("ESS", ess), ("max weight share", share)],
        warnings=(["weights explode as the censoring survivor approaches zero "
                   "at long follow-up; truncate the time axis rather than "
                   "weighting through administrative censoring"]
                  + ([f"one subject carries {share:.1%} of the weight"]
                     if share > 0.1 else [])),
        payload={
            "weights": w, "G": g, "max_weight_share": share, "ess": ess,
            "n_censored": int(c.sum()), "n": int(t.size),
            "method": "censoring_at_risk_weight",
        },
    )


def cheatsheet():
    return "chrwgt: restores the censored population; weights explode at long follow-up -- truncate instead"
