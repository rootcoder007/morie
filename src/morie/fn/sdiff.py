# morie.fn -- function file (rootcoder007/morie)
"""Synthetic difference-in-differences estimator."""

import numpy as np

from ._did import as_panel, simplex_lstsq
from ._richresult import RichResult

__all__ = ["synthetic_did"]


def _unit_weights(Y0_pre, Y1_pre_mean, zeta):
    """Donor weights with an intercept and the ridge term of Algorithm 1."""
    w, w0, _ = simplex_lstsq(Y0_pre.T, Y1_pre_mean, zeta=zeta, intercept=True)
    return w, w0


def _time_weights(Y0_pre, Y0_post_mean):
    """Pre-period weights: which past periods look like the post-period."""
    lam, l0, _ = simplex_lstsq(Y0_pre, Y0_post_mean, zeta=0.0, intercept=True)
    return lam, l0


def _sdid_point(Y, treated, t0, zeta):
    ctrl = ~treated
    Y0pre = Y[ctrl][:, :t0]
    w, _ = _unit_weights(Y0pre, Y[treated][:, :t0].mean(axis=0), zeta)
    lam, _ = _time_weights(Y0pre, Y[ctrl][:, t0:].mean(axis=1))
    tr_post = Y[treated][:, t0:].mean()
    tr_pre = float(Y[treated][:, :t0].mean(axis=0) @ lam)
    co_post = float(w @ Y[ctrl][:, t0:].mean(axis=1))
    co_pre = float(w @ (Y[ctrl][:, :t0] @ lam))
    return (tr_post - tr_pre) - (co_post - co_pre), w, lam


def synthetic_did(Y, unit_id, time_id, treated, treatment_time, zeta=None,
                  n_boot=0, seed=None):
    r"""Reweight units AND periods, then take a difference-in-differences.

    Arkhangelsky, Athey, Hirshberg, Imbens and Wager's estimator sits
    between the two methods it is named after, and fixes a weakness in
    each. Synthetic control requires the synthetic unit to match the
    treated unit's pre-treatment LEVEL exactly, which is often
    impossible and is not what identification needs. Plain DiD uses
    every control unit and every pre-period with equal weight, which
    is only sensible when trends really are parallel across all of
    them. Synthetic DiD keeps the difference-in-differences structure
    but chooses both sets of weights:

    .. math:: \hat\tau^{sdid} = \Big(\bar Y^{post}_{tr}
              - \sum_t \lambda_t \bar Y_{tr,t}\Big)
              - \Big(\sum_j \omega_j \bar Y^{post}_{j}
              - \sum_j \sum_t \omega_j \lambda_t Y_{jt}\Big).

    Two differences from synthetic control matter. The unit weights
    include an INTERCEPT, so the synthetic unit need only be parallel
    to the treated unit, not equal to it -- level differences are
    absorbed by the DiD structure. And a ridge penalty
    :math:`\zeta` spreads weight across donors rather than letting a
    couple of units carry the estimate; the default is the paper's
    :math:`\zeta = (N_{tr}T_{post})^{1/4}\hat\sigma` with
    :math:`\hat\sigma` the standard deviation of first differences
    among control units in the pre-period.

    Time weights are the piece plain DiD lacks: they downweight
    pre-periods that look nothing like the post-period, which is
    exactly when the parallel-trends extrapolation is least credible.

    Parameters
    ----------
    Y : array-like
        Outcomes: long format with ``unit_id``/``time_id``, or a
        (n_units, n_periods) matrix.
    unit_id, time_id : array-like or None
        Identifiers for long format; ``None`` if ``Y`` is a matrix.
    treated : array-like
        Treated unit identifiers, or a boolean mask over the rows of
        a matrix ``Y``.
    treatment_time : scalar
        First treated period; adoption must be simultaneous.
    zeta : float, optional
        Ridge parameter for the unit weights. The paper's default is
        used when omitted, and the value used is reported.
    n_boot : int
        Jackknife-free placebo replications for the standard error.
        Zero returns the leave-one-out jackknife of Algorithm 3.
    seed : int, optional
        Seed for the placebo draws.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``unit_weights``,
        ``time_weights``, ``zeta``, ``n_donors_used``,
        ``n_periods_used``, ``did_estimate``, ``sc_estimate``.

    References
    ----------
    Arkhangelsky, Athey, Hirshberg, Imbens and Wager (2021), *AER*
    111:4088-4118, Algorithms 1-3.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> f = np.cumsum(rng.normal(size=16))
    >>> load = np.concatenate([[1.0, 1.1], rng.uniform(0.4, 1.6, size=10)])
    >>> Y = np.outer(load, f) + rng.normal(scale=0.05, size=(12, 16))
    >>> Y[:2, 10:] += 2.0
    >>> out = synthetic_did(Y, None, None, [0, 1], 10)
    >>> bool(abs(out["estimate"] - 2.0) < 0.25)
    True
    """
    Ya = np.asarray(Y, dtype=float)
    if unit_id is None and time_id is None:
        M = np.atleast_2d(Ya)
        units = np.arange(M.shape[0])
        periods = np.arange(M.shape[1])
    else:
        M, units, periods = as_panel(Ya, unit_id, time_id)
    n_u, T = M.shape

    tr = np.asarray(treated)
    if tr.dtype == bool and tr.size == n_u:
        mask = tr
    else:
        mask = np.isin(units, tr)
    if mask.sum() < 1:
        raise ValueError("no treated unit was found in the unit set.")
    if (~mask).sum() < 2:
        raise ValueError(
            "need at least 2 control units, got %d." % int((~mask).sum())
        )
    tt = np.nonzero(periods >= treatment_time)[0]
    if tt.size == 0:
        raise ValueError("treatment_time is after the last period.")
    t0 = int(tt[0])
    if t0 < 2:
        raise ValueError(
            "only %d pre-treatment period(s); both weight sets are fitted on "
            "the pre-period." % t0
        )
    if t0 >= T:
        raise ValueError("no post-treatment period.")

    ctrl = ~mask
    if zeta is None:
        d = np.diff(M[ctrl][:, :t0], axis=1)
        sigma = float(np.std(d, ddof=1)) if d.size > 1 else 1.0
        zeta = float((mask.sum() * (T - t0)) ** 0.25 * sigma)
    est, w, lam = _sdid_point(M, mask, t0, float(zeta))

    # the two estimators it sits between, on the same data
    did = float(
        (M[mask][:, t0:].mean() - M[mask][:, :t0].mean())
        - (M[ctrl][:, t0:].mean() - M[ctrl][:, :t0].mean())
    )
    wsc, _, _ = simplex_lstsq(M[ctrl][:, :t0].T, M[mask][:, :t0].mean(axis=0))
    sc = float(M[mask][:, t0:].mean() - wsc @ M[ctrl][:, t0:].mean(axis=1))

    # leave-one-unit-out jackknife (Algorithm 3); needs >1 treated unit
    if n_boot:
        rng = np.random.default_rng(seed)
        donors = np.nonzero(ctrl)[0]
        reps = []
        for _ in range(int(n_boot)):
            # placebo: pretend a random set of donors was treated
            pick = rng.choice(donors, size=int(mask.sum()), replace=False)
            pm = np.zeros(n_u, dtype=bool)
            pm[pick] = True
            sub = np.zeros(n_u, dtype=bool)
            sub[donors] = True
            keep = sub
            Ms = M[keep]
            pms = pm[keep]
            if pms.sum() < 1 or (~pms).sum() < 2:
                continue
            reps.append(_sdid_point(Ms, pms, t0, float(zeta))[0])
        se = float(np.std(reps, ddof=1)) if len(reps) > 1 else np.nan
        se_method = "placebo (Algorithm 4)"
    elif mask.sum() > 1:
        jk = []
        idx = np.nonzero(mask)[0].tolist() + np.nonzero(ctrl)[0].tolist()
        for i in idx:
            keep = np.ones(n_u, dtype=bool)
            keep[i] = False
            if mask[keep].sum() < 1 or (~mask[keep]).sum() < 2:
                continue
            jk.append(_sdid_point(M[keep], mask[keep], t0, float(zeta))[0])
        jk = np.array(jk)
        n_j = jk.size
        se = float(np.sqrt((n_j - 1) / n_j * np.sum((jk - jk.mean()) ** 2)))
        se_method = "leave-one-unit-out jackknife (Algorithm 3)"
    else:
        se = np.nan
        se_method = (
            "unavailable: the jackknife needs more than one treated unit, "
            "and with a single treated unit placebo inference (n_boot > 0) "
            "is the option the paper leaves open"
        )

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": float(est),
            "att": float(est),
            "se": se,
            "ci": ((est - z * se, est + z * se) if np.isfinite(se)
                   else (np.nan, np.nan)),
            "unit_weights": w,
            "time_weights": lam,
            "donors": units[ctrl],
            "n_donors_used": int((w > 1e-8).sum()),
            "n_periods_used": int((lam > 1e-8).sum()),
            "zeta": float(zeta),
            "zeta_note": (
                "(N_tr * T_post)^(1/4) * sd(first differences of control "
                "pre-period outcomes), the paper's default; it spreads "
                "weight across donors instead of concentrating it"
            ),
            "did_estimate": did,
            "sc_estimate": sc,
            "comparison_note": (
                "sdid keeps the DiD double difference but chooses both unit "
                "and period weights; the unit weights carry an intercept, so "
                "the synthetic unit must be PARALLEL to the treated unit, not "
                "equal to it"
            ),
            "se_method": se_method,
            "t0_index": t0,
            "n_treated": int(mask.sum()),
            "n_control": int(ctrl.sum()),
            "n_periods": int(T),
            "method": "Synthetic difference-in-differences (Arkhangelsky et al. 2021)",
        }
    )


def cheatsheet():
    return (
        "sdiff: synthetic DiD -- simplex unit weights with an intercept and "
        "ridge, plus time weights, reported next to plain DiD and SC"
    )
