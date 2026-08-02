# morie.fn -- function file (rootcoder007/morie)
"""BvM for Cox model: semiparametric efficient estimation of regression coefficient."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_cox_bvm"]


def ghosal_cox_bvm(x, time=None, event=None, beta_grid=None):
    r"""Semiparametric Bernstein-von Mises theorem for the Cox model
    (Ghosal Sec. 13.6.2):

    .. math:: \sqrt n\,(\beta_n - \beta_0) \rightsquigarrow
              N\big(0, I_{\beta|\lambda}^{-1}\big),

    where :math:`I_{\beta|\lambda}` is the EFFICIENT information --
    the information for :math:`\beta` after projecting out the
    infinite-dimensional baseline hazard.

    The theorem says the posterior for :math:`\beta`, marginalised
    over a nonparametric prior on the baseline hazard, is
    asymptotically normal and centred at an efficient estimator, with
    variance equal to the semiparametric efficiency bound. Two
    consequences follow and are the practical content:

    * Bayesian credible intervals for :math:`\beta` are
      asymptotically valid CONFIDENCE intervals. That is not
      automatic in semiparametric problems -- a BvM theorem can and
      does fail for other functionals, where credible sets have the
      wrong coverage.
    * nothing is lost by not knowing the baseline hazard: the
      variance is the same as if it were known up to the efficient
      projection, which is what ``efficient`` records.

    The module fits the Cox partial likelihood, reports the observed
    efficient information, and evaluates the implied normal
    approximation on a grid so the claim is checkable.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p)
        Covariates.
    time : array-like, optional
        Follow-up times; taken from ``x`` when it is 1-D and ``time``
        is absent is not allowed, so supply it.
    event : array-like of {0, 1}, optional
        Event indicators; all events otherwise.
    beta_grid : array-like, optional
        Grid for the posterior approximation of the first
        coefficient.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``efficient_information``,
        ``beta_grid``, ``posterior_normal``, ``efficient`` (True),
        ``credible_equals_confidence`` (True), ``n_events``, ``n``,
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 13.6.2 (Bernstein-von Mises) and
    Sec. 13.7.2 (Cox proportional hazards).
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if time is None:
        raise ValueError("time is required: the Cox partial likelihood needs "
                         "follow-up times as well as covariates.")
    tv = np.asarray(time, dtype=float).ravel()
    if X.shape[0] != tv.size:
        X = X.T
    if X.shape[0] != tv.size:
        raise ValueError("x must have one row per follow-up time.")
    n, p = X.shape
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    ev = np.ones(n) if event is None else \
        np.asarray(event, dtype=float).ravel()
    if ev.size != n:
        raise ValueError(f"event has {ev.size} entries for {n} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if ev.sum() < 2:
        raise ValueError("need at least 2 events to fit the Cox model.")

    def nll_grad_hess(b):
        eta = X @ b
        w = np.exp(eta - eta.max())
        ll = 0.0
        gr = np.zeros(p)
        he = np.zeros((p, p))
        for i in np.nonzero(ev == 1.0)[0]:
            at = tv >= tv[i]
            sw = float(w[at].sum())
            if sw <= 0:
                continue
            xb = (w[at, None] * X[at]).sum(axis=0) / sw
            ll += eta[i] - np.log(sw)
            gr += X[i] - xb
            xx = (w[at, None, None] * (X[at][:, :, None] * X[at][:, None, :])
                  ).sum(axis=0) / sw
            he -= xx - np.outer(xb, xb)
        return -ll, -gr, -he

    b = np.zeros(p)
    for _ in range(50):
        _, gr, he = nll_grad_hess(b)
        step = np.linalg.solve(he + 1e-10 * np.eye(p), gr)
        b = b - step
        if np.max(np.abs(step)) < 1e-10:
            break
    _, _, info = nll_grad_hess(b)
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    bg = np.linspace(b[0] - 4 * se[0], b[0] + 4 * se[0], 101) \
        if beta_grid is None else \
        np.atleast_1d(np.asarray(beta_grid, dtype=float))
    post = np.exp(-0.5 * ((bg - b[0]) / max(se[0], 1e-12)) ** 2) / \
        (max(se[0], 1e-12) * np.sqrt(2 * np.pi))
    return RichResult(payload={
        "beta": b, "se": se, "efficient_information": info,
        "beta_grid": bg, "posterior_normal": post,
        "efficient": True, "credible_equals_confidence": True,
        "caveat": "BvM can FAIL for other semiparametric functionals; "
                  "validity here is a theorem about this one",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Cox partial likelihood with the semiparametric BvM of Sec. 13.6.2"})


def cheatsheet():
    return "gh_c13_15: BvM is what makes credible intervals valid confidence intervals -- it can fail elsewhere"
