# morie.fn -- function file (rootcoder007/morie)
"""Bayesian probit IRT via Albert's Gibbs sampler (data augmentation)."""

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

__all__ = ["mcmcpack_irt", "_irt_gibbs"]


def _truncnorm(rng, mean, positive):
    """One draw from N(mean, 1) truncated to the requested sign.

    Inverse-cdf in the stable form: for y > 0,
    y = mean - Phi^{-1}(u Phi(mean)); for y < 0,
    y = mean + Phi^{-1}(u Phi(-mean)).  Both evaluate Phi^{-1} in its
    lower tail, which stays finite when the truncation point is many sds
    from the mean (the textbook form hits Phi^{-1}(1) = inf there).
    """
    u = max(float(rng.random()), 1e-300)
    if positive:
        return mean - stats.norm.ppf(u * stats.norm.cdf(mean))
    return mean + stats.norm.ppf(u * stats.norm.cdf(-mean))


def _irt_gibbs(V, n_iter, burnin, seed, prior_sd=5.0, polarity_idx=None):
    """Albert (1992) data-augmentation Gibbs for the 2-parameter probit IRT."""
    import math

    rng = np.random.default_rng(seed)
    Vl = [[float(v) for v in row] for row in np.asarray(V, dtype=float).tolist()]
    n, q = len(Vl), len(Vl[0])
    obs = [[v == v for v in row] for row in Vl]
    tau2 = prior_sd ** 2

    # initialise ideal points from row yea-rates
    x = []
    for i in range(n):
        vals = [Vl[i][j] for j in range(q) if obs[i][j]]
        x.append(sum(vals) / len(vals) if vals else 0.0)
    mx = sum(x) / n
    sx = max((sum((v - mx) ** 2 for v in x) / n) ** 0.5, 1e-8)
    x = [(v - mx) / sx for v in x]
    a = [0.0] * q  # item difficulty
    b = [1.0] * q  # item discrimination
    ystar = [[0.0] * q for _ in range(n)]
    kept_x = []
    for it in range(n_iter):
        # 1. latent utilities: y* ~ N(b_j x_i - a_j, 1) truncated by the vote
        for i in range(n):
            for j in range(q):
                if obs[i][j]:
                    ystar[i][j] = _truncnorm(rng, x[i] * b[j] - a[j], Vl[i][j] == 1.0)
        # 2. item parameters (a_j, b_j) | y*, x -- Bayesian regression of
        #    y*_j on [-1, x] with a N(0, tau^2 I) prior
        for j in range(q):
            rows = [i for i in range(n) if obs[i][j]]
            m = len(rows)
            if m < 2:
                continue
            s1 = sum(x[i] for i in rows)
            s2 = sum(x[i] * x[i] for i in rows)
            p11, p12, p22 = m + 1.0 / tau2, -s1, s2 + 1.0 / tau2
            r1 = -sum(ystar[i][j] for i in rows)
            r2 = sum(x[i] * ystar[i][j] for i in rows)
            det = p11 * p22 - p12 * p12
            c11, c12, c22 = p22 / det, -p12 / det, p11 / det
            mu1 = c11 * r1 + c12 * r2
            mu2 = c12 * r1 + c22 * r2
            # draw from N(mu, cov) through the Cholesky factor of cov
            l11 = math.sqrt(c11)
            l21 = c12 / l11
            l22 = math.sqrt(max(c22 - l21 * l21, 0.0))
            z1, z2 = float(rng.standard_normal()), float(rng.standard_normal())
            a[j] = mu1 + l11 * z1
            b[j] = mu2 + l21 * z1 + l22 * z2
        # 3. ideal points x_i | y*, items -- N(0, 1) prior
        for i in range(n):
            cols = [j for j in range(q) if obs[i][j]]
            if not cols:
                continue
            prec = sum(b[j] * b[j] for j in cols) + 1.0
            mu = sum(b[j] * (ystar[i][j] + a[j]) for j in cols) / prec
            x[i] = mu + math.sqrt(1.0 / prec) * float(rng.standard_normal())
        # identify: mean 0 / sd 1, absorb into items; optional polarity
        muX = sum(x) / n
        sdX = max((sum((v - muX) ** 2 for v in x) / n) ** 0.5, 1e-8)
        x = [(v - muX) / sdX for v in x]
        a = [a[j] - b[j] * muX for j in range(q)]
        b = [b[j] * sdX for j in range(q)]
        if polarity_idx is not None and x[polarity_idx] > 0:
            x = [-v for v in x]
            b = [-v for v in b]
        if it >= burnin:
            kept_x.append(list(x))
    return np.array(kept_x), np.array(a), np.array(b)


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


# compact alias per ledger/NAMING.md
mcmcpackirt = mcmcpack_irt
