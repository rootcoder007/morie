# morie.fn -- function file (rootcoder007/morie)
"""Bayesian mediation with conjugate normal priors."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bayes_mediation"]


def bayes_mediation(x, m, y, prior_sd=10.0, n_draws=4000, seed=0, c=None):
    r"""Posterior for the indirect effect under a normal-normal model.

    With a normal prior :math:`\beta \sim N(0, \tau^2 I)` on each
    equation's coefficients and a known-variance normal likelihood, the
    posterior for each regression is conjugate,

    .. math:: \beta \mid y \sim N\!\left(
              (X'X/\hat\sigma^2 + I/\tau^2)^{-1} X'y/\hat\sigma^2,\;
              (X'X/\hat\sigma^2 + I/\tau^2)^{-1} \right),

    so draws of the a- and b-paths give draws of :math:`ab` directly.
    The posterior for the *product* is skewed even when each path's is
    symmetric -- the reason Yuan and MacKinnon recommend Bayesian (or
    bootstrap) intervals over the normal-theory Sobel interval.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    prior_sd : float, default 10.0
        Prior standard deviation tau (weakly informative by default).
    n_draws : int, default 4000
        Posterior draws.
    seed : int, default 0
        RNG seed.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``indirect_mean``, ``indirect_median``, ``indirect_ci``
        (2.5/97.5 percentiles), ``p_direction`` (posterior mass on the
        sign of the mean), ``direct_mean``, ``draws`` (n_draws,),
        ``n``, ``method``.

    References
    ----------
    Yuan, Y. & MacKinnon, D. P. (2009). Bayesian mediation analysis.
    *Psychological Methods*, 14(4), 301-322.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n):
        raise ValueError("x, m, y must have equal length.")
    tau = float(prior_sd)
    if tau <= 0:
        raise ValueError(f"prior_sd must be positive, got {tau}.")
    nd = int(n_draws)
    if nd < 100:
        raise ValueError(f"n_draws must be at least 100, got {nd}.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 6:
        raise ValueError("too few observations for the two regressions.")

    rng = np.random.default_rng(seed)

    def posterior(D, t):
        b_ols, *_ = np.linalg.lstsq(D, t, rcond=None)
        s2 = float(((t - D @ b_ols) ** 2).sum() / max(t.size - D.shape[1], 1))
        prec = D.T @ D / s2 + np.eye(D.shape[1]) / tau**2
        cov = np.linalg.inv(prec)
        mean = cov @ (D.T @ t / s2)
        return rng.multivariate_normal(mean, cov, size=nd)

    one = np.ones(n)
    a_draws = posterior(np.column_stack([one, x, C]), m)[:, 1]
    yb = posterior(np.column_stack([one, x, m, C]), y)
    c_draws, b_draws = yb[:, 1], yb[:, 2]
    ab = a_draws * b_draws

    mean = float(ab.mean())
    pd = float((ab > 0).mean() if mean >= 0 else (ab < 0).mean())
    return RichResult(
        payload={
            "indirect_mean": mean,
            "indirect_median": float(np.median(ab)),
            "indirect_ci": (float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))),
            "p_direction": pd,
            "direct_mean": float(c_draws.mean()),
            "draws": ab,
            "n": int(n),
            "method": "Bayesian mediation (conjugate normal posteriors, product draws)",
        }
    )


def cheatsheet():
    return "baymed: conjugate posteriors for a and b; posterior of ab is skewed"
