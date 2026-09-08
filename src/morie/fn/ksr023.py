# morie.fn -- function file (rootcoder007/morie)
"""Cox partial-likelihood score process."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_cox_score_process", "kosorok_ch1_cox_estimating_equation"]


def kosorok_cox_score_process(beta, z, time, event, t_grid=None):
    r"""Cox partial-likelihood score PROCESS (Kosorok Eq. 1.4, p. 5):

    .. math:: U_n(t,\beta) = n^{-1}\sum_i \int_0^t
              \big[Z_i - E_n(s,\beta)\big]\,dN_i(s),
              \qquad
              E_n(s,\beta) = \frac{\sum_i Z_i Y_i(s)e^{\beta'Z_i}}
                                   {\sum_i Y_i(s)e^{\beta'Z_i}}.

    Indexed by :math:`t`, not just evaluated at the end. That is the
    point of the example: the score is a stochastic PROCESS in time,
    and its weak convergence -- not merely the asymptotic normality
    of :math:`U_n(\infty,\beta)` -- is what licences the usual Cox
    inference. :math:`E_n(s,\beta)` is the risk-set weighted average
    covariate, so each event contributes how far its covariate sits
    from the average among those still at risk.

    :math:`U_n(\infty,\hat\beta) = 0` defines the estimator, which
    makes this a Z-estimation problem and connects Chapter 1 to the
    machinery of Chapter 2.

    Parameters
    ----------
    beta : array-like, shape (p,)
        Coefficients.
    z : array-like, shape (n,) or (n, p)
        Covariates.
    time : array-like, shape (n,)
        Follow-up times.
    event : array-like of {0, 1}, shape (n,)
        Event indicators.
    t_grid : array-like, optional
        Times at which to report the process.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``U``, ``U_final``, ``E_bar``,
        ``is_process`` (True), ``root_defines_estimator`` (True),
        ``n_events``, ``n``, ``method``.
    References
    ----------
    Kosorok, Ch. 1, Eq. (1.4), p. 5.
    """
    b = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != tv.size:
        Z = Z.T
    if Z.shape[0] != tv.size:
        raise ValueError("z must have one row per follow-up time.")
    if ev.size != tv.size:
        raise ValueError(f"event has {ev.size} entries for {tv.size} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if b.size != Z.shape[1]:
        raise ValueError(f"beta has {b.size} entries for {Z.shape[1]} columns.")
    n, p = Z.shape
    w = np.exp(Z @ b)
    et = np.sort(tv[ev == 1.0])
    if et.size == 0:
        raise ValueError("no events: the score process is identically zero.")
    tg = et if t_grid is None else np.atleast_1d(np.asarray(t_grid, dtype=float))

    contrib = np.zeros((et.size, p))
    ebar = np.zeros((et.size, p))
    for k, s in enumerate(et):
        at = tv >= s
        sw = float(w[at].sum())
        if sw <= 0:
            continue
        e = (w[at, None] * Z[at]).sum(axis=0) / sw
        ebar[k] = e
        # every event exactly at s contributes
        for i in np.nonzero((tv == s) & (ev == 1.0))[0]:
            contrib[k] += Z[i] - e
    cum = np.cumsum(contrib, axis=0) / n
    U = np.array([cum[max(np.searchsorted(et, v, side="right") - 1, 0)]
                  if v >= et[0] else np.zeros(p) for v in tg])
    return RichResult(payload={
        "t_grid": tg, "U": U, "U_final": cum[-1], "E_bar": ebar,
        "is_process": True, "root_defines_estimator": True,
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Cox score process (Eq. 1.4); indexed by t, so its weak convergence is what matters"})


def cheatsheet():
    return "ksr023: the score is a PROCESS in t -- its weak convergence licences Cox inference"


#: Catalogue alias for :func:`kosorok_cox_score_process`.
kosorok_ch1_cox_estimating_equation = kosorok_cox_score_process
