# morie.fn -- function file (rootcoder007/morie)
"""Multiplicative intensity model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch1_multiplicative_intensity"]


def kosorok_ch1_multiplicative_intensity(time, event, Z, beta=None, t=None):
    r"""Aalen multiplicative intensity model (Kosorok Ch. 1):

    .. math:: E[N(t) \mid Z] = \int_0^t E[Y(s)\mid Z]\,
              e^{\beta' Z}\, d\Lambda(s),

    with N the counting process, Y the at-risk indicator and Lambda
    the baseline cumulative hazard. Lambda is an infinite-dimensional
    nuisance; it is estimated here by the Breslow estimator given
    beta, which is exactly the profiling step that makes the Cox
    partial likelihood work.

    Parameters
    ----------
    time, event : array-like, shape (n,)
        Follow-up times and 0/1 indicators.
    Z : array-like, shape (n,) or (n, p)
        Covariates.
    beta : array-like, optional
        Coefficients; zeros if omitted.
    t : float or array-like, optional
        Times at which to report the cumulative hazard; the event
        times if omitted.

    Returns
    -------
    RichResult
        keys: ``t``, ``cumulative_hazard`` (Breslow),
        ``expected_counts`` (n x len(t)), ``beta``, ``n_events``,
        ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 1 (counting processes; the multiplicative intensity model).
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = Z.shape
    if time.size != n or event.size != n:
        raise ValueError("time and event must match the rows of Z.")
    if not np.all(np.isin(event, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    beta = np.zeros(p) if beta is None else np.atleast_1d(
        np.asarray(beta, dtype=float)
    )
    if beta.size != p:
        raise ValueError(f"beta must have {p} entries.")

    w = np.exp(Z @ beta)
    ev_times = np.sort(time[event == 1])
    if ev_times.size == 0:
        raise ValueError("no events; the baseline hazard is unidentified.")
    # Breslow increments dLambda = dN / sum_{at risk} exp(beta'Z)
    inc = []
    for ti in ev_times:
        denom = float(w[time >= ti].sum())
        inc.append(1.0 / denom if denom > 0 else 0.0)
    cum = np.cumsum(inc)
    tt = ev_times if t is None else np.atleast_1d(np.asarray(t, dtype=float))
    idx = np.searchsorted(ev_times, tt, side="right") - 1
    Lam = np.where(idx >= 0, cum[np.clip(idx, 0, cum.size - 1)], 0.0)
    at_risk = (time[:, None] >= tt[None, :]).astype(float)
    return RichResult(
        payload={"t": tt, "cumulative_hazard": Lam,
                 "expected_counts": at_risk * (w[:, None] * Lam[None, :]),
                 "beta": beta, "n_events": int(event.sum()), "n": int(n),
                 "method": "Aalen multiplicative intensity; Lambda by Breslow"}
    )


def cheatsheet():
    return "ksr022: E[N(t)|Z] with Breslow baseline; Lambda profiled, not assumed"
