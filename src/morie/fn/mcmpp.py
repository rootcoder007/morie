# morie.fn -- function file (rootcoder007/morie)
"""Bayesian probit IRT via Albert's Gibbs sampler (data augmentation)."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["mcmcpack_irt", "_irt_gibbs"]


def _truncnorm(rng, mean, positive):
    """Draw from N(mean, 1) truncated to the requested sign."""
    u = rng.random(mean.shape)
    if positive:
        lo = stats.norm.cdf(-mean)
        return mean + stats.norm.ppf(lo + u * (1 - lo))
    hi = stats.norm.cdf(-mean)
    return mean + stats.norm.ppf(u * hi)


def _irt_gibbs(V, n_iter, burnin, seed, prior_sd=5.0, polarity_idx=None):
    """Albert (1992) data-augmentation Gibbs for the 2-parameter probit IRT."""
    rng = np.random.default_rng(seed)
    n, q = V.shape
    obs = ~np.isnan(V)
    tau2 = prior_sd**2

    # initialise ideal points from row yea-rates
    x = np.nan_to_num(np.nanmean(V, axis=1))
    x = (x - x.mean()) / max(x.std(), 1e-8)
    a = np.zeros(q)  # item difficulty
    b = np.ones(q)  # item discrimination
    ystar = np.zeros((n, q))

    kept_x = []
    for it in range(n_iter):
        # 1. latent utilities: y* ~ N(b_j x_i - a_j, 1) truncated by the vote
        mean = x[:, None] * b[None, :] - a[None, :]
        pos = V == 1
        neg = V == 0
        ystar[pos] = _truncnorm(rng, mean[pos], True)
        ystar[neg] = _truncnorm(rng, mean[neg], False)
        # 2. item parameters (a_j, b_j) | y*, x  -- Bayesian regression on [-1, x]
        for j in range(q):
            m = obs[:, j]
            if m.sum() < 2:
                continue
            D = np.column_stack([-np.ones(m.sum()), x[m]])
            prec = D.T @ D + np.eye(2) / tau2
            cov = np.linalg.inv(prec)
            mu = cov @ (D.T @ ystar[m, j])
            a[j], b[j] = rng.multivariate_normal(mu, cov)
        # 3. ideal points x_i | y*, items -- N(0,1) prior
        for i in range(n):
            m = obs[i]
            if not m.any():
                continue
            prec = (b[m] ** 2).sum() + 1.0
            mu = (b[m] * (ystar[i, m] + a[m])).sum() / prec
            x[i] = rng.normal(mu, np.sqrt(1 / prec))
        # identify: mean 0 / sd 1, absorb into items; optional polarity
        muX, sdX = x.mean(), max(x.std(), 1e-8)
        x = (x - muX) / sdX
        a = a - b * muX
        b = b * sdX
        if polarity_idx is not None and x[polarity_idx] > 0:
            x = -x
            b = -b
        if it >= burnin:
            kept_x.append(x.copy())
    return np.array(kept_x), a, b


def mcmcpack_irt(votes, n_iter=2000, burnin=500, seed=0, polarity_idx=None):
    r"""One-dimensional Bayesian probit IRT for roll-call data.

    Albert's data-augmentation Gibbs sampler for

    .. math:: P(y_{ij} = 1) = \Phi(\beta_j x_i - \alpha_j),

    the model behind MCMCpack's ``MCMCirt1d`` and pscl's ``ideal``:
    draw truncated-normal latent utilities given the votes, then
    conjugate normal updates for the item parameters and ideal
    points. The scale is re-identified each sweep (mean 0, sd 1,
    optional polarity), so the reported posterior is over the
    identified parameterisation.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Binary roll-call matrix (1 = yea, 0 = nay, NaN = missing).
    n_iter, burnin, seed :
        Sampler controls.
    polarity_idx : int, optional
        Legislator forced to the negative side.

    Returns
    -------
    RichResult
        keys: ``ideal_points`` (posterior means), ``ideal_ci``
        (2, n), ``alpha`` (q,), ``beta`` (q,), ``n_kept``, ``n``,
        ``q``, ``method``.

    References
    ----------
    Albert, J. H. (1992). Bayesian estimation of normal ogive item
    response curves using Gibbs sampling. *Journal of Educational
    Statistics*, 17(3), 251-269.

    Clinton, J., Jackman, S. & Rivers, D. (2004). The statistical
    analysis of roll call data. *APSR*, 98(2), 355-370.
    """
    V = np.asarray(votes, dtype=float)
    if V.ndim != 2:
        raise ValueError("votes must be 2-D.")
    ok = ~np.isnan(V)
    if not np.all(np.isin(V[ok], (0.0, 1.0))):
        raise ValueError("votes must be binary 0/1 (NaN for missing).")
    n_iter, burnin = int(n_iter), int(burnin)
    if n_iter <= burnin:
        raise ValueError("n_iter must exceed burnin.")

    draws, a, b = _irt_gibbs(V, n_iter, burnin, seed, polarity_idx=polarity_idx)
    return RichResult(
        payload={
            "ideal_points": draws.mean(axis=0),
            "ideal_ci": np.percentile(draws, [2.5, 97.5], axis=0),
            "alpha": a,
            "beta": b,
            "n_kept": int(draws.shape[0]),
            "n": int(V.shape[0]),
            "q": int(V.shape[1]),
            "method": "Bayesian probit IRT (Albert 1992 data-augmentation Gibbs)",
        }
    )


def cheatsheet():
    return "mcmpp: truncated-normal augmentation, conjugate item/ability updates (Albert 1992)"
