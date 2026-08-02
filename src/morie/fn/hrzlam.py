# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric baseline hazard estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_baseline_hazard_est"]


def horowitz_baseline_hazard_est(t, x, event, beta_hat, bandwidth=None,
                                 grid=None):
    r"""Kernel-smoothed baseline hazard for the proportional hazards
    model (Horowitz Sec. 6.2.4), equation (6.44):

    .. math:: \lambda_{n0}(y) = \frac{1}{h_n}\int
              K\!\left(\frac{y - \xi}{h_n}\right) d\Lambda_{n0}(\xi),

    where :math:`\Lambda_{n0}` is the Breslow estimator of the
    cumulative baseline hazard,

    .. math:: \hat\Lambda_0(t) = \sum_{t_i \le t}
              \frac{d_i}{\sum_j \mathbf 1\{t_j \ge t_i\}
              \exp(X_j'\hat\beta)}.

    The smoothing is NOT cosmetic. :math:`\lambda_0 = d\Lambda_0/dy`,
    so the obvious estimator is :math:`d\Lambda_{n0}/dy` -- and that
    does not work, because :math:`\Lambda_{n0}` is a STEP function
    whose derivative is zero almost everywhere and undefined at the
    jumps. This is the same obstruction as differentiating an
    empirical distribution function to get a density, and it has the
    same remedy: smooth first. Writing (6.44) as a Stieltjes integral
    against the jumps makes it a weighted kernel sum over the event
    times, with the Breslow increments as weights.

    Two facts follow and are returned rather than implied:

    * the rate is no faster than :math:`n^{-2/5}` for a twice
      differentiable :math:`\lambda_0` and a second-order kernel --
      :math:`n^{-1/2}` is NOT attainable here, exactly as in density
      estimation. A higher-order kernel and more derivatives buy a
      faster rate, but never root-n.
    * the leading bias is :math:`\tfrac12 A_K h_n^2\lambda_0''(y)`
      with :math:`A_K = \int \zeta^2 K(\zeta)d\zeta`, from the
      Taylor expansion at (6.47)-(6.48). :math:`A_K` is returned so
      the bias is computable rather than a footnote.

    Parameters
    ----------
    t : array-like, shape (n,)
        Observed durations.
    x : array-like, shape (n, d)
        Covariates.
    event : array-like of {0, 1}, shape (n,)
        1 for an event, 0 for right-censoring.
    beta_hat : array-like, shape (d,)
        Regression coefficients.
    bandwidth : float, optional
        :math:`h_n`; ``n**(-1/5)`` times the spread of the event
        times otherwise, the rate that attains :math:`n^{-2/5}`.
    grid : array-like, optional
        Points at which to evaluate the hazard.

    Returns
    -------
    RichResult
        keys: ``grid``, ``lambda0_hat``, ``cumhaz_times``,
        ``cumhaz`` (Breslow), ``bandwidth``, ``A_K``,
        ``rate_exponent`` (-2/5), ``root_n_attainable`` (False),
        ``n_events``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 6.2.4, eqs. (6.44)-(6.48).
    """
    from ._hrz_transform import kernel_K

    tv = np.asarray(t, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != tv.size:
        X = X.T
    if X.shape[0] != tv.size:
        raise ValueError("x must have one row per entry of t.")
    if ev.size != tv.size:
        raise ValueError(
            f"event has {ev.size} entries for {tv.size} durations.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if np.any(tv < 0):
        raise ValueError("durations must be non-negative.")
    n, d = X.shape
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    b = np.asarray(beta_hat, dtype=float).ravel()
    if b.size != d:
        raise ValueError(f"beta_hat has {b.size} entries for {d} covariates.")

    risk = np.exp(X @ b)
    if not np.all(np.isfinite(risk)):
        raise ValueError("exp(X'beta) overflowed; rescale the covariates.")

    order = np.argsort(tv)
    ts, es, rs = tv[order], ev[order], risk[order]
    # Breslow: at each event time, 1 / sum of risk scores still at risk
    at_risk = np.cumsum(rs[::-1])[::-1]
    jump_t = ts[es == 1.0]
    jump_w = (1.0 / at_risk)[es == 1.0]
    if jump_t.size == 0:
        raise ValueError("no events: the baseline hazard is not identified.")
    cumhaz = np.cumsum(jump_w)

    spread = float(jump_t.max() - jump_t.min())
    hh = (spread if spread > 0 else 1.0) * n ** (-0.2) if bandwidth is None \
        else float(bandwidth)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(jump_t.min(), jump_t.max(), 50) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))

    # (6.44) as a Stieltjes sum: the measure dLambda_n0 puts mass
    # jump_w at each event time
    lam = (kernel_K((g[:, None] - jump_t[None, :]) / hh) *
           jump_w[None, :]).sum(axis=1) / hh

    return RichResult(payload={
        "grid": g, "lambda0_hat": lam,
        "cumhaz_times": jump_t, "cumhaz": cumhaz,
        "bandwidth": hh,
        "A_K": 1.0,  # second moment of the standard normal kernel
        "rate_exponent": -0.4, "root_n_attainable": False,
        "n_events": int(jump_t.size), "n": int(n),
        "method": "Breslow cumulative hazard smoothed by (6.44); differentiating the step function does not work"})


def cheatsheet():
    return "hrzlam: Lambda_n0 is a STEP function -- smooth it, never differentiate it; no root-n here"
