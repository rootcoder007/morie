# morie.fn -- function file (rootcoder007/morie)
"""Causal survival forest for heterogeneous time-to-event effects."""

from . import _array_core as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["causal_survival_forest"]


def _rmst(time, event, horizon):
    """Restricted mean survival time up to `horizon` via Kaplan-Meier."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    surv, prev, n_at_risk = 1.0, 0.0, t.size
    area = 0.0
    for i in range(t.size):
        if t[i] > horizon:
            break
        if e[i] == 1:
            area += surv * (t[i] - prev)
            prev = t[i]
            surv *= 1 - 1 / max(n_at_risk, 1)
        n_at_risk -= 1
    return area + surv * (horizon - prev)


def causal_survival_forest(time, event, D, X, horizon=None, n_trees=200, min_leaf=15, seed=0):
    r"""Heterogeneous effects on restricted mean survival time.

    Censoring makes the raw event time unusable as a forest outcome.
    This uses the inverse-probability-of-censoring-weighted pseudo
    outcome

    .. math:: \tilde Y_i = \frac{\delta_i \min(T_i, \tau)}
              {\hat G(\min(T_i, \tau)^-)}
              + \frac{(1-\delta_i)\,\tau}{\hat G(\tau)}\,
                \mathbb{1}\{T_i > \tau\},

    with :math:`\hat G` the Kaplan-Meier estimate of the censoring
    distribution, so :math:`E[\tilde Y] = E[\min(T, \tau)]` -- the
    RMST at horizon :math:`\tau`. The honest causal forest then runs
    on that pseudo outcome, and :math:`\hat\tau(x)` is a difference in
    restricted mean survival months, not a hazard ratio.

    Parameters
    ----------
    time : array-like, shape (n,)
        Observed follow-up times (positive).
    event : array-like of {0, 1}, shape (n,)
        1 = event, 0 = censored.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    horizon : float, optional
        Restriction time tau; default the 90th percentile of ``time``.
    n_trees, min_leaf, seed :
        Forest hyperparameters.

    Returns
    -------
    RichResult
        keys: ``cate`` (RMST difference per unit), ``cate_oob``,
        ``ate``, ``horizon``, ``pseudo_outcome``, ``n``, ``forest``,
        ``method``.

    References
    ----------
    Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S. & Zhu, R. (2023).
    Estimating heterogeneous treatment effects with right-censored
    data via causal survival forests. *Journal of the Royal
    Statistical Society Series B*, 85(2), 179-211.
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    n = time.size
    if not (event.size == n and D.size == n):
        raise ValueError("time, event, D must have equal length.")
    if np.any(time <= 0):
        raise ValueError("time must be positive.")
    for v, name in ((event, "event"), (D, "D")):
        if not np.all(np.isin(v, (0.0, 1.0))):
            raise ValueError(f"{name} must be binary 0/1.")
    tau = float(np.percentile(time, 90)) if horizon is None else float(horizon)
    if tau <= 0:
        raise ValueError(f"horizon must be positive, got {tau}.")

    # Kaplan-Meier for the censoring distribution (event indicator flipped)
    order = np.argsort(time)
    ts, cs = time[order], 1 - event[order]
    G, at_risk = 1.0, n
    grid, vals = [0.0], [1.0]
    for i in range(n):
        if cs[i] == 1:
            G *= 1 - 1 / max(at_risk, 1)
            grid.append(ts[i])
            vals.append(G)
        at_risk -= 1
    grid, vals = np.array(grid), np.maximum(np.array(vals), 1e-3)

    def Ghat(t):
        return vals[np.searchsorted(grid, t, side="right") - 1]

    tmin = np.minimum(time, tau)
    pseudo = np.where(
        (event == 1) & (time <= tau),
        time / Ghat(np.maximum(time - 1e-12, 0.0)),
        np.where(time > tau, tau / Ghat(tau), 0.0),
    )

    f = CausalForest(n_trees=n_trees, min_leaf=min_leaf, seed=seed)
    f.fit(X, pseudo, D)
    cate = f.predict()
    return RichResult(
        payload={
            "cate": cate,
            "cate_oob": f.predict(oob=True),
            "ate": float(np.nanmean(cate)),
            "horizon": tau,
            "pseudo_outcome": pseudo,
            "n": int(n),
            "forest": f,
            "method": "Causal survival forest on IPC-weighted RMST pseudo outcomes",
        }
    )


def cheatsheet():
    return "csfgrf: IPCW RMST pseudo outcome, then an honest causal forest"
